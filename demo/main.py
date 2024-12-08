import torch
import numpy as np
from VecKM_flow.inference import VecKMNormalFlowEstimator
from VecKM_flow.visualize import gen_flow_video

""" Initialize the normal flow estimator.
There are four options for the training_set: 'UNION', 'MVSEC', 'DSEC', 'EVIMO'.
The 'UNION' option is recommended for general use, which is trained on the union of MVSEC, DSEC, and EVIMO.
"""
estimator = VecKMNormalFlowEstimator(training_set='UNION')


""" Load demo data.
WARNING: It is crutial to convert the raw pixel coordinates to the normalized coordinates. See the README for more details.
"""
events_t = np.load('/fs/nexus-projects/VecKM_flow/data/all_eval/MVSEC/indoor_flying1/dataset_events_t.npy')
events_xy = np.load('/fs/nexus-projects/VecKM_flow/data/all_eval/MVSEC/indoor_flying1/undistorted_events_xy.npy')
events_t = torch.tensor(events_t)
events_xy = torch.tensor(events_xy).float()
print(f"events_t: {events_t.shape}, events_xy: {events_xy.shape}")


""" Perform inference.
The 'ensemble' parameter controls the number of ensemble members for the inference. 
Usually higher ensemble number leads to better performance, if computation is not a concern.
"""
flow_predictions, flow_uncertainty = estimator.inference(events_t, events_xy, ensemble=10)
print(f"flow_predictions: {flow_predictions.shape}, flow_uncertainty: {flow_uncertainty.shape}")


""" Mask out the uncertain flow predictions. """
flow_predictions[flow_uncertainty > 0.3] = np.nan


""" generate the flow video for visualization. """
gen_flow_video(
    events_t.numpy(), 
    events_xy.numpy(), 
    flow_predictions.numpy(), 
    './frames', './output.mp4', fps=30)

