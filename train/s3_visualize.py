from s0_params import dataset_path
from VecKM_flow.visualize import gen_flow_video
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='EVIMO')
parser.add_argument('--model_name', type=str, default='EVIMO')
parser.add_argument('--split', type=str, default='eval')
args = parser.parse_args()
print(args)

eval_path = os.path.join(dataset_path, args.dataset, args.split)
scene_names = os.listdir(eval_path)
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(os.path.join(eval_path, scene_name))]

for scene_name in scene_names:
    events_t = np.load(os.path.join(eval_path, scene_name, 'dataset_events_t.npy'))
    events_xy = np.load(os.path.join(eval_path, scene_name, 'undistorted_events_xy.npy'))
    flow_predictions = np.load(os.path.join(eval_path, scene_name, f'dataset_pred_flow_{args.model_name}.npy'))
    flow_uncertainty = np.load(os.path.join(eval_path, scene_name, f'dataset_angle_vars_flow_{args.model_name}.npy'))
    flow_predictions[flow_uncertainty > 0.3] = np.nan
    gen_flow_video(
        events_t, 
        events_xy, 
        flow_predictions, 
        './frames', f'./videos/{scene_name}.mp4', fps=30)