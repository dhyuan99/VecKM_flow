import numpy as np
import os
from tqdm import tqdm

def density_mining(scenes, norm_range):
    bins = np.linspace(
        np.log(norm_range[0]), np.log(norm_range[1]), 50)
    bins_all = np.zeros((len(bins)-1))
    for scene in scenes:
        flow = scene.events_flow
        flow_norm = np.linalg.norm(flow, axis=1)
        log_flow_norm = np.log(flow_norm[flow_norm>0])
        hist, bin_edges = np.histogram(log_flow_norm, bins)
        bins_all = bins_all + hist
    bins = np.exp(bins)
    inverse_density = bins_all.max() / bins_all
    return bins, inverse_density

class Scene:
    def __init__(self, scene_path, pxl_radius, t_radius, t_scale=1):
        events_t  = np.load(os.path.join(scene_path, 'dataset_events_t.npy'))
        events_xy = np.load(os.path.join(scene_path, 'undistorted_events_xy.npy')).astype(np.float32)
        events_p  = np.load(os.path.join(scene_path, 'dataset_events_p.npy')).astype(np.float32)
        
        pos_events = events_p == events_p.max()
        neg_events = events_p == events_p.min()
        events_p[pos_events] = 1
        events_p[neg_events] = -1
        self.events_p = events_p

        events_t = events_t - events_t[0]                                       # prevent overflow.
        events_t = events_t / t_radius * t_scale
        events_t = events_t.astype(np.float32)
        self.events_t = events_t
        
        events_xy = events_xy / pxl_radius
        self.events = np.concatenate((events_t[:,None], events_xy), axis=1)
        
        self.t_min, self.t_max = np.min(self.events_t), np.max(self.events_t)
        num_steps = int((self.t_max - self.t_min) / 2)                          # after normalizing, the step size will be 2*1=2
        self.t_grid = np.linspace(
            self.t_min, 
            self.t_min + num_steps * 2, 
            num_steps, False)
        
        if os.path.exists(os.path.join(scene_path, 'undistorted_optical_flow.npy')):
            events_flow = np.load(
                os.path.join(scene_path, 'undistorted_optical_flow.npy')
            ).astype(np.float32)
            self.events_flow = events_flow
            self.flow_norm = np.linalg.norm(self.events_flow, axis=1)

    def compute_sample_probability(self, density_grid, inverse_density):
        self.probability = np.zeros_like(self.flow_norm)
        indices = np.searchsorted(density_grid, self.flow_norm) - 1
        self.valid_indices = np.where(
            np.logical_and(
                indices >= 0, 
                indices < len(inverse_density) - 1
            )
        )[0]

        self.probability[self.valid_indices] = inverse_density[
            indices[self.valid_indices]
        ]
        
        self.valid_prob = self.probability[self.valid_indices]
        
        return self.valid_indices, self.valid_prob

    def __getitem__(self, idx):
        """ used for inference.
        """
        t_center = (self.t_grid[idx] + self.t_grid[idx+1]) / 2
        slice_indices = np.logical_and(
            self.events_t>self.t_grid[idx], self.events_t<self.t_grid[idx+1]
        )
        events = self.events[slice_indices]
        events_p = self.events_p[slice_indices]
        events[:,0] -= t_center                                                 # optional, just to make it numeric stable.
        return events, events_p, slice_indices

    def __len__(self):
        return len(self.t_grid)-1

    def slice(self, event_idx, random_rotation=True, random_scaling=True):
        """ used for training.
        """
        
        if random_scaling:
            scale = np.random.uniform(0.8, 1.2)
        else:
            scale = 1

        t_center = self.events_t[event_idx]
        start_idx = np.searchsorted(self.events_t, t_center - scale)
        end_idx = np.searchsorted(self.events_t, t_center + scale)
        slice_indices = np.arange(start_idx, end_idx)
        events = self.events[slice_indices] / scale
        events_p = self.events_p[slice_indices]
        flows = self.events_flow[slice_indices]

        if random_rotation:
            alpha = np.random.uniform(0, 2*np.pi)
            random_xy_rotation = np.array([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)],
            ])
            random_xy_rotation_2d = np.array([
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)]
            ])
            events = events @ (random_xy_rotation)
            flows = flows @ (random_xy_rotation_2d)

        return events, events_p, flows

class FlowDataset:
    """ Dataset for training and validation, NOT for inference.
    """
    def __init__(self, 
                 directory, 
                 pxl_radius, t_radius, 
                 norm_range,
                 random_rotation=True, 
                 random_scaling=True):
        """
        Args:
            directory: str, directory storing all scenes.
            pxl_radius: float, radius of the local region.
            t_radius: float, radius of the temporal region.
            norm_range: tuple, (min, max), range of the norm of the optical flow.
            K: tuple, camera matrix. Will convert the normalized xy to this target camera.
            random_rotation: bool, whether to apply random rotation.
            random_scaling: bool, whether to apply random scaling.
        """
        self.random_rotation = random_rotation
        self.random_scaling  = random_scaling
        print(f'Loading dataset from {directory}')
        
        self.scenes = []
        for scene in tqdm(os.listdir(directory)):
            scene_dir = os.path.join(directory, scene)
            if not os.path.isdir(scene_dir):
                continue
            self.scenes.append(
                Scene(scene_dir, pxl_radius, t_radius)
            )

        grid, inverse_density = density_mining(self.scenes, norm_range)

        self.scene_split = []
        self.valid_index_list = []
        self.probability_list = []
        for scene in self.scenes:
            valid_indices, valid_prob = scene.compute_sample_probability(
                grid, inverse_density
            )
            self.scene_split.append(
                len(valid_prob)
            )
            self.valid_index_list.append(
                valid_indices
            )
            self.probability_list.append(
                valid_prob
            )
        self.scene_split = np.cumsum(self.scene_split)
        self.valid_index_list = np.concatenate(self.valid_index_list)
        self.probability_list = np.concatenate(self.probability_list)
        self.probability_list /= self.probability_list.sum()
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        r_idx = np.random.choice(len(self.probability_list), p=self.probability_list)
        scene_idx = np.searchsorted(self.scene_split, r_idx)
        event_idx = self.valid_index_list[r_idx]
        return self.scenes[scene_idx].slice(
            event_idx, 
            random_rotation=self.random_rotation, 
            random_scaling=self.random_scaling
        )

    def __len__(self):
        return 20