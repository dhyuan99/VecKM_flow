import importlib.resources as pkg_resources
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from . import models  # Import the models subpackage
from .estimator import NormalEstimator

class VecKMNormalFlowEstimator(nn.Module):
    def __init__(self, training_set='UNION'):
        assert training_set in ['UNION', 'EVIMO', 'DSEC', 'MVSEC'], "Invalid training set"
        if training_set == 'UNION':
            from .params import UNIONParams as P
        elif training_set == 'EVIMO':
            from .params import EVIMOParams as P
        elif training_set == 'DSEC':
            from .params import DSECParams as P
        elif training_set == 'MVSEC':
            from .params import MVSECParams as P
        else:
            raise ValueError("Invalid training set")
        self.P = P

        super(VecKMNormalFlowEstimator, self).__init__()
        self.estimator = self.init_model()
        self.load_model(self.estimator, training_set)
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.estimator = self.estimator.cuda()
    
    @staticmethod
    def load_model(estimator, training_set):
        # Load the .pth file from the models directory
        with pkg_resources.open_binary(models, f"{training_set}.pth") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"), weights_only=True)
        estimator.load_state_dict(state_dict)
        estimator.eval()
        return estimator
    
    def init_model(self):
        return NormalEstimator(self.P.d, self.P.alpha)
    
    def inference(self, events_t, events_xy, ensemble=3):
        assert len(events_t.shape) == 1 and len(events_xy.shape) == 2, "Invalid input shape"
        assert events_t.shape[0] == events_xy.shape[0], "Inconsistent input shape"
        assert torch.all(events_t[1:] >= events_t[:-1]), "Events are not sorted"
        
        events_t = events_t / self.P.t_radius
        events_xy = events_xy / self.P.pxl_radius
        flow_predictions = torch.zeros_like(events_xy)
        flow_uncertainty = torch.zeros_like(events_xy[:, 0]) + 9999
        
        progress_bar = tqdm(total=events_t.shape[0], desc="computing per-event normal flow")
        for start, end, events_txy in self.slice_events(events_t, events_xy):
            with torch.no_grad():
                if self.cuda_available:
                    events_txy = events_txy.cuda()
                flow_pred, flow_uncert = self.estimator.inference(
                    events_txy, ensemble=ensemble)
                flow_predictions[start:end] = flow_pred
                flow_uncertainty[start:end] = flow_uncert
            progress_bar.update((end-start).item())
        
        return flow_predictions, flow_uncertainty
        
    def slice_events(self, events_t, events_xy):
        t_min, t_max = events_t.min(), events_t.max()
        num_steps = int((t_max - t_min) / 2)                          # after normalizing, the step size will be 2*1=2
        t_grid = np.linspace(
            t_min, 
            t_min + num_steps * 2, 
            num_steps, False)
        for i in range(num_steps-1):
            start, end = t_grid[i], t_grid[i+1]
            start_idx = torch.searchsorted(events_t, start)
            end_idx = torch.searchsorted(events_t, end)
            events = torch.cat([
                events_t[start_idx:end_idx, None], 
                events_xy[start_idx:end_idx]], dim=-1)
            yield start_idx, end_idx, events
        