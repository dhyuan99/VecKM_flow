import torch
import torch.optim

import os
from s0_dataset import FlowDataset
from s0_model import NormalEstimator, MotionFieldLoss, EndPointLoss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='EVIMO')
parser.add_argument('--model_name', type=str, default='EVIMO')
parser.add_argument('--random_scaling', action='store_true')
args = parser.parse_args()
print(args)

from s0_params import dataset_path
if args.dataset == 'MVSEC':
    from s0_params import MVSECParams as P
elif args.dataset == 'EVIMO':
    from s0_params import EVIMOParams as P
elif args.dataset == 'DSEC':
    from s0_params import DSECParams as P
else:
    raise ValueError(f'Unknown dataset {args.dataset}.')
print(P())

print(f"radius in terms of normalized pixel:      {P.pxl_radius}.")
print(f"radius in terms of original coordinates: ({P.pxl_radius * P.K[0]}, {P.pxl_radius * P.K[1]}).")

import time
cur_time_str = time.strftime("%Y%m%d_%H%M%S")
os.system(f'mkdir -p checkpoints/{cur_time_str}')
print(f'Saving checkpoints to checkpoints/{cur_time_str}')

trainset = FlowDataset(
    os.path.join(dataset_path, args.dataset, "train"), 
    P.pxl_radius, P.t_radius, P.train_norm_range,
    random_rotation=True, random_scaling=args.random_scaling)
valset = FlowDataset(
    os.path.join(dataset_path, args.dataset, "eval"),
    P.pxl_radius, P.t_radius, P.val_norm_range,
    random_rotation=False, random_scaling=False)

model = NormalEstimator(P.d, P.alpha).cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn = MotionFieldLoss if P.loss_fn == 'mfl' else EndPointLoss

min_eval_total_loss = 1e9
for epoch in range(5000):
    total_loss = 0
    model.train()
    for events, events_p, flows in trainset:
        events = torch.tensor(events).float().cuda()
        events_p = torch.tensor(events_p).float().cuda()
        flows = torch.tensor(flows).float().cuda()
        
        if len(events) > 80000:
            r = torch.randperm(len(events))[:80000]
            events = events[r]
            events_p = events_p[r]
            flows = flows[r]

        pred_flows = model(events)
        valid_indices = torch.norm(flows, dim=-1) > 0
        loss = loss_fn(
            pred_flows[valid_indices], 
            flows[valid_indices],
            P.mfl_loss_lambda
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
 
    count = len(trainset)
    print(f'epoch {epoch}. '
          f'total_loss: {total_loss/count:.5f}. '
          f'norm_pred: {torch.norm(pred_flows, dim=-1)[0].item():.5f}; '
          f'norm_opti: {torch.norm(flows, dim=-1)[0].item():.5f} [maybe nan, does not matter].')

    if (epoch+1) % 20 == 0:
        model.eval()
        total_sign_correct_rate = 0
        total_norm_error_abs = 0
        total_end_point_error = 0
        count = 0
        for _ in range(10):
            for events, events_p, flows in valset:
                events = torch.tensor(events).float().cuda()
                events_p = torch.tensor(events_p).float().cuda()
                flows = torch.tensor(flows).float().cuda()
                
                if len(events) > 80000:
                    r = torch.randperm(len(events))[:80000]
                    events = events[r]
                    events_p = events_p[r]
                    flows = flows[r]

                with torch.no_grad():
                    pred_flows = model(events)
                    
                    valid_indices = torch.logical_and(
                        torch.norm(flows, dim=-1) > 1e-6,
                        torch.norm(pred_flows, dim=-1) > 1e-6
                    )
                    pred_flows = pred_flows[valid_indices]
                    flows = flows[valid_indices]

                    dot = torch.sum(pred_flows * flows, dim=-1)
                    pred_norm = torch.norm(pred_flows, dim=-1)

                    total_sign_correct_rate += torch.sum(dot > 0).item()
                    total_norm_error_abs += torch.sum(
                        torch.abs(dot / pred_norm - pred_norm)
                    ).item()
                    
                    total_end_point_error += torch.sum(
                        torch.norm(pred_flows - flows, dim=-1)
                    ).item()

                    count += len(flows)

        print(f'\tsign correct rate: {total_sign_correct_rate/count:.5f}. norm error abs: {total_norm_error_abs/count:.5f}. end point error: {total_end_point_error/count:.5f}.')

        err = (total_norm_error_abs - total_sign_correct_rate) / count
        # willing to use 0.01 sign correct rate to reduce 0.1 norm error 
        if err < min_eval_total_loss:
            min_eval_total_loss = err
            print(f'\tSaving currently best model.')
            torch.save(model.state_dict(), f'checkpoints/{cur_time_str}/{args.model_name}.pth')

            with open(f'checkpoints/{cur_time_str}/{args.model_name}.txt', 'w') as f:
                f.write(f'epoch {epoch}. '
                        f'total_loss: {total_loss/count:.5f}. '
                        f'norm_pred: {torch.norm(pred_flows, dim=-1).mean():.5f}; '
                        f'norm_opti: {torch.norm(flows, dim=-1).mean():.5f}.\n'
                        f'\tsign correct rate: {total_sign_correct_rate/count:.5f}. norm error abs: {total_norm_error_abs/count:.5f}. end point error: {total_end_point_error/count:.5f}.\n')