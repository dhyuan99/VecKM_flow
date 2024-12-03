import bisect
import numpy as np
from s0_utils import Pose, interpolate_pose
import cv2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_start_with(list_str, start_str):
    return sorted([s for s in list_str if s.startswith(start_str)])
    
def filter_pred(pred, a_vars, a_threshold):
    invalid_indices = a_vars > a_threshold
    pred[invalid_indices] = np.nan
    # print(f'valid pct. after filertering: {np.mean(~np.isnan(pred)):.4f}')
    return pred

class OpticalFlowDatasetEvents:
    def __init__(self, path, dt=0.016, diff_imu_t_data_t=0.0):
        self.path = path
        self.dt = dt
        self.diff_imu_t_data_t = diff_imu_t_data_t
        
        self.events_t = np.load(f'{path}/dataset_events_t.npy')
        self.events_xy = np.load(f'{path}/dataset_events_xy.npy')
        self.flow = np.load(f'{path}/dataset_pred_flow.npy')
        self.a_vars = np.load(f'{path}/dataset_pred_uncertainty.npy')
        
        self.info = np.load(f'{path}/dataset_info.npz', allow_pickle=True)
        self.K = self.info['K']
        self.D = self.info['D']        
        self.info = self.info['meta'].astype(str).item()
        self.info = eval(self.info)
        
        poses = self.info['full_trajectory']
        self.all_poses = [Pose(
            pose['ts'], 
            np.array([
                pose['cam']['pos']['t']['x'],
                pose['cam']['pos']['t']['y'],
                pose['cam']['pos']['t']['z'],
            ]),
            np.array([
                pose['cam']['pos']['q']['x'],
                pose['cam']['pos']['q']['y'],
                pose['cam']['pos']['q']['z'],
                pose['cam']['pos']['q']['w']
            ])
        ) for pose in poses]
        
        # Match the IMU measurements to the GT timestamps with interpolation
        imus = self.info['imu']['/prophesee/left/imu']
        self.imu_angular_velocities_t = np.array([imu['ts'] for imu in imus])
        self.imu_angular_velocities = np.array([(imu['angular_velocity']['y'],
                                                 -imu['angular_velocity']['x'],
                                                 imu['angular_velocity']['z']) for imu in imus])

        # plt.figure()
        # plt.plot(imu_angular_velocities_t, imu_angular_velocities[:, 0])
        # plt.plot(imu_angular_velocities_t, imu_angular_velocities[:, 1])
        # plt.plot(imu_angular_velocities_t, imu_angular_velocities[:, 2])
        # plt.show()

        # pose_t = np.array([pose.t for pose in self.all_poses])

        # print(pose_t.shape, imu_angular_velocities_t.shape, imu_angular_velocities.shape)
        # imu_angular_velocities_x = np.interp(pose_t, imu_angular_velocities_t, imu_angular_velocities[:, 0])
        # imu_angular_velocities_y = np.interp(pose_t, imu_angular_velocities_t, imu_angular_velocities[:, 1])
        # imu_angular_velocities_z = np.interp(pose_t, imu_angular_velocities_t, imu_angular_velocities[:, 2])
        # self.imu_angular_velocities = np.stack((imu_angular_velocities_x, imu_angular_velocities_y, imu_angular_velocities_z), axis=1)

        # This is done by finding the IMU measurement that is closest to the GT timestamp.
        # imu_timestamps = np.array([imu['ts'] for imu in imus])
        # time_diff = np.abs(self.timestamps[:, None] - imu_timestamps)
        # imu_indices = np.argmin(time_diff, axis=1)
        # # print(imu_timestamps[imu_indices])
        # # print(self.timestamps)
        # self.imu_angular_velocities = np.stack([
        #     get_omega(imu['angular_velocity']) for imu in imus
        # ], axis=0)[imu_indices]

        t_start = np.min(self.all_poses[0].t)
        t_end = np.max(self.all_poses[-1].t)
        self.t = np.arange(t_start, t_end, dt)
        self.t_end = self.t + dt

    def __getitem__(self, idx):
        left_time, right_time = self.t[idx], self.t_end[idx]
        left_pose = interpolate_pose(left_time, self.all_poses)
        right_pose = interpolate_pose(right_time, self.all_poses)

        # bisect.bisect_left is much faster than np.searchsorted because
        # np.searchsorted likes to make local copies
        # left_idx = np.searchsorted(self.events_t, left_time)
        # right_idx = np.searchsorted(self.events_t, right_time)
        left_idx  = bisect.bisect_left(self.events_t, left_time)
        right_idx = bisect.bisect_left(self.events_t, right_time)

        indices = np.arange(left_idx, right_idx)
        all_events_xy = self.events_xy[indices]
        flow = self.flow[indices]
        a_vars = self.a_vars[indices]
        
        flow = filter_pred(flow, a_vars, 0.3)
        good_idx = np.linalg.norm(flow, axis=1) > 0
        
        events_xy = all_events_xy[good_idx]
        flow = flow[good_idx]
        
        if len(events_xy) <= 5:
            return None
        
        events_xy_end = events_xy + flow
        events_xy_end_normalized = cv2.undistortPoints(
            events_xy_end.astype(np.float32),
            self.K, self.D
        ).squeeze()
        events_xy_normalized = cv2.undistortPoints(
            events_xy.astype(np.float32),
            self.K, self.D
        ).squeeze()
        flow = events_xy_end_normalized - events_xy_normalized

        imu_angular_velocity_x = np.interp(left_time + self.diff_imu_t_data_t, self.imu_angular_velocities_t, self.imu_angular_velocities[:, 0])
        imu_angular_velocity_y = np.interp(left_time + self.diff_imu_t_data_t, self.imu_angular_velocities_t, self.imu_angular_velocities[:, 1])
        imu_angular_velocity_z = np.interp(left_time + self.diff_imu_t_data_t, self.imu_angular_velocities_t, self.imu_angular_velocities[:, 2])
        imu_angular_velocity = np.array((imu_angular_velocity_x, imu_angular_velocity_y, imu_angular_velocity_z))

        return {
            'left_time': left_time,
            'right_time': right_time,
            'T_wc1': left_pose,
            'T_wc2': right_pose,
            'flow': flow,
            'xy': events_xy,
            'all_xy': all_events_xy,
            'imu_angular_velocity': imu_angular_velocity,
        }

    def __len__(self):
        return len(self.t)