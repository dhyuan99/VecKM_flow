import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List

class Pose:
    """ Class to represent a pose in SE(3)
    Pose.t: float, timestamp
    Pose.position: np.array, (3,), translation
    Pose.quaternion: np.array, (4,), quaternion
    """
    def __init__(self, t, position, quaternion):
        self.t = t
        self.position = position
        self.quaternion = quaternion
    def __repr__(self):
        return f"Pose(t={self.t}, position={self.position}, quaternion={self.quaternion})"

def apply_transform(T_cb: Pose, T_ba: Pose) -> Pose:
    """ Apply an SE(3) transform to an element of SE(3)
    e.g. T_cb to T_ba to get T_ca
    Input:
        T_cb: np.array, (8,)
            T_cb[0]: float, timestamp
            T_cb[1:4]: float, translation
            T_cb[4:8]: float, quaternion
        T_ba: np.array, (8,), same format as T_cb
    Output:
        T_ca: np.array, (8,), same format as T_cb
    """
    
    R_ba = R.from_quat(T_ba.quaternion)
    t_ba = T_ba.position

    R_cb = R.from_quat(T_cb.quaternion)
    t_cb = T_cb.position

    R_ca = R_cb * R_ba
    t_ca = R_cb.as_matrix() @ t_ba + t_cb
    return Pose(T_ba.t, t_ca, R_ca.as_quat())

def inv_transform(T_ba):
    """ Compute the inverse transform of an SE(3) transform
    Input:
        T_ba: np.array, (8,)
            T_ba[0]: float, timestamp
            T_ba[1:4]: float, translation
            T_ba[4:8]: float, quaternion
    Output:
        T_ab: np.array, (8,), same format as T_ba
    """
    R_ba = R.from_quat(T_ba.quaternion)
    t_ba = T_ba.position

    R_ab = R_ba.inv()
    t_ab = -R_ba.inv().as_matrix() @ t_ba
    return Pose(T_ba.t, t_ab, R_ab.as_quat())

def interpolate_pose(t: float, poses: List[Pose]) -> Pose:
    """
    Interpolate a pose at time t from a list of poses
    Input:
        t: float, timestamp
        poses: List[Pose], list of poses
    Output:
        pose: Pose, interpolated pose
    """
    
    timestamps = [pose.t for pose in poses]
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), "Timestamps are not sorted"
    
    right_i = np.searchsorted(timestamps, t)
    if right_i == len(poses):
        return None
    if right_i == 0:
        return None

    left_t  = poses[right_i-1].t
    right_t = poses[right_i].t

    alpha = (t - left_t) / (right_t - left_t)
    if alpha > 1:
        return None
    elif alpha < 0:
        return None

    left_position  = poses[right_i-1].position
    right_position = poses[right_i].position

    position_interp = alpha * (right_position - left_position) + left_position

    left_right_rot_stack = R.from_quat((
        poses[right_i-1].quaternion,
        poses[right_i].quaternion
    ))

    slerp = Slerp((0, 1), left_right_rot_stack)
    R_interp = slerp(alpha)

    return Pose(t, position_interp, R_interp.as_quat())