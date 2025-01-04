import numpy as np
import torch
import os
from tqdm import tqdm

from s0_dataset import Scene
from s0_model import NormalEstimator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='EVIMO')
parser.add_argument('--model_name', type=str, default='EVIMO')
parser.add_argument('--split', type=str, default='eval')
parser.add_argument('--t_scale', type=float, default=1.)
parser.add_argument('--num_ensembles', type=int, default=10)
args = parser.parse_args()
print(args)

from s0_params import dataset_path
if args.model_name == 'EVIMO':
    from s0_params import EVIMOParams as P
elif args.model_name == 'DSEC':
    from s0_params import DSECParams as P
elif args.model_name == 'MVSEC':
    from s0_params import MVSECParams as P
elif args.model_name == 'UNION':
    from s0_params import UNIONParams as P
else:
    raise ValueError(f'Unknown model {args.model_name}.')
print(P())

model = NormalEstimator(P.d, P.alpha).cuda()
model.load_state_dict(torch.load(f'model_checkpoints/{args.model_name}.pth'))
model.eval()

eval_path = os.path.join(dataset_path, args.dataset, args.split)
scene_names = os.listdir(eval_path)
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(os.path.join(eval_path, scene_name))]
print(f"scenes: {scene_names}")

for scene_name in scene_names:
    print(scene_name)
    cur_dir = os.path.join(eval_path, scene_name)
    if not os.path.isdir(cur_dir):
        continue
    scenes = Scene(
        cur_dir, P.pxl_radius, P.t_radius, args.t_scale)
    pred_flows = np.zeros((scenes.events.shape[0], 2))
    angle_vars = np.zeros((scenes.events.shape[0])) + 999
    for scene_idx, scene in tqdm(enumerate(scenes)):
        events, events_p, slice_idx = scene
        pred_flow_out = np.zeros((events.shape[0], 2))
        angle_var_out = np.zeros((events.shape[0])) + 999
        
        if len(events) > 80000:
            r = np.random.permutation(len(events))[:80000]
        else:
            r = np.arange(len(events))

        pred_flow, angle_var = model.inference(
            torch.tensor(events[r]).float().cuda(),
            ensemble=args.num_ensembles
        )
        pred_flow = pred_flow.detach().cpu().numpy()
        angle_var = angle_var.detach().cpu().numpy()
        
        pred_flow_out[r] = pred_flow
        angle_var_out[r] = angle_var

        pred_flows[slice_idx] = pred_flow_out
        angle_vars[slice_idx] = angle_var_out

    pred_flows = pred_flows * args.t_scale
    
    np.save(
        os.path.join(
            cur_dir, f'dataset_pred_flow_{args.model_name}.npy'
        ), pred_flows) 
    np.save(
        os.path.join(
            cur_dir, f'dataset_angle_vars_flow_{args.model_name}.npy'
        ), angle_vars) 

