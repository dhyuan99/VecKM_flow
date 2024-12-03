import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Attempt to disable OpenBLAS multithreading used by NumPy, it slows things down

import numpy as np
import cv2
from s0_dataset import OpticalFlowDatasetEvents
from s0_utils import apply_transform, inv_transform
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy.optimize import minimize
import os
from tqdm import tqdm
import time

def egomotion_from_pose(T_wc1, T_wc2):
    # Get egomotion
    v_c1_w  = (T_wc2.position - T_wc1.position) / (right_time - left_time)
    R_wc1 = R.from_quat(T_wc1.quaternion).as_matrix()
    v_c1 = R_wc1.T @ v_c1_w

    T_c1c2 = apply_transform(inv_transform(T_wc1), T_wc2)
    R_c1c2 = R.from_quat(T_c1c2.quaternion).as_matrix()

    norm_v_c1 = np.linalg.norm(v_c1)
    v_c1 = v_c1 / norm_v_c1 # This does not actually need to be normalized, it doesn't affect the sign
    w_c2 = R.from_matrix(R_c1c2).as_rotvec() / (right_time - left_time)

    return v_c1, w_c2, norm_v_c1

def form_positivity_matrices_with_w(precomputed_A_x, precomputed_B_x, xy, d_normalized, w):
    A_x = precomputed_A_x[xy[:,1].astype(int), xy[:,0].astype(int)]
    B_x = precomputed_B_x[xy[:,1].astype(int), xy[:,0].astype(int)]

    dx_normalized = d_normalized[:, 0]
    dy_normalized = d_normalized[:, 1]

    g_x = np.stack((dx_normalized, dy_normalized), axis=-1)
    g_x = g_x / np.linalg.norm(g_x, axis=-1, keepdims=True)
    n_x = g_x[:, 0] * (dx_normalized / (right_time - left_time)) + g_x[:, 1] * (dy_normalized / (right_time - left_time))

    g_x_A_x = (g_x[:,None,:] @ A_x)[:,0,:]
    n_x_g_x_B_x_w = n_x.squeeze() - np.einsum('ij,ij->i', g_x, B_x @ w)

    return g_x_A_x, n_x_g_x_B_x_w

# v0 is unused because the problem is convex
def svm_positivity(g_x_A_x, n_x_g_x_B_x_w, v0=None):
    #############################################################
    ##### Egomotion estimation from g_x, n_x, A_x, B_x ##########
    ############################################################
    sample_weights = np.abs(n_x_g_x_B_x_w)
    sign_w = np.sign(n_x_g_x_B_x_w)
    # sample_weights = np.abs(sign_w)

    X_balanced = np.concatenate([g_x_A_x, -g_x_A_x], axis=0)
    Y_balanced = np.concatenate([sign_w, -sign_w], axis=0)
    S_balanced = np.concatenate([sample_weights, sample_weights], axis=0)
    res = LinearSVC(fit_intercept=False, C=1).fit(
        X_balanced,
        Y_balanced, 
        sample_weight=S_balanced)
    sign_v = res.decision_function(g_x_A_x)
    v_c1_pred = res.coef_.squeeze()
    v_c1_pred = v_c1_pred / np.linalg.norm(v_c1_pred)
    # print(f"pct: {np.mean(np.sign(sign_v) == np.sign(sign_w)):.3f}", v_c1_pred, v_c1)
    return v_c1_pred

def relu(x):
    return x * ((np.sign(x) + 1) / 2)

def f(v, loss, loss_scale, barrier_scale):
    cheirality = (g_x_A_x @ v) * n_x_g_x_B_x_w
    return loss_scale * (loss(cheirality) + barrier_scale * (np.linalg.norm(v) - 1)**2)

# General cheirality optimization
default_loss_scale = 1e-1
default_barrier_scale = 1e0
def optimize_cheirality(g_x_A_x, n_x_g_x_B_x_w, loss=lambda x: np.mean(x),
                        v0=None, jac=None, loss_scale=default_loss_scale, barrier_scale=default_barrier_scale):
    if v0 is None:
        v0 = np.array((0.0, 0.0, 1.0))

    res = minimize(lambda x: f(x, loss, loss_scale, barrier_scale), x0=v0, method='BFGS', jac=jac)
    v_star = res.x / np.linalg.norm(res.x)
    return v_star

def minimize_negative_cheraility_jac(g_x_A_x, n_x_g_x_B_x_w, v, loss_scale=default_loss_scale, barrier_scale=default_barrier_scale):
    dcheirality_dx = -(n_x_g_x_B_x_w.reshape((-1, 1)) * g_x_A_x)
    cheirality = dcheirality_dx @ v
    drelu_dx = (np.sign(cheirality) + 1) / 2

    dresidual_dx = drelu_dx.reshape((-1, 1)) * dcheirality_dx
    grad_J = np.mean(dresidual_dx, axis=0)

    norm_v = np.linalg.norm(v)
    grad_barrier = 2 * (norm_v - 1) * v / norm_v

    return loss_scale * (grad_J + barrier_scale * grad_barrier)

def minimize_negative_cheraility(g_x_A_x, n_x_g_x_B_x_w, v0):
    loss = lambda x: np.mean(relu(-x))
    jac = lambda v: minimize_negative_cheraility_jac(g_x_A_x, n_x_g_x_B_x_w, v)

    # Check gradient
    # v0 = np.array((0.0, 0.0, 2.0))
    # eps = 1e-6
    # l0 = f(v0, loss)
    # ls = []
    # for i in range(v0.shape[0]):
    #     v_perturbed = np.copy(v0)
    #     v_perturbed[i] += eps
    #     ls.append(f(v_perturbed, loss))
    # ls = np.array(ls)
    # grad_l = (ls - l0) / eps
    # print('grad_l', grad_l)
    # jac(v0)
    # exit(0)

    v_star = optimize_cheirality(g_x_A_x, n_x_g_x_B_x_w, loss=loss, v0=v0, jac=jac)
    return v_star

data_root = './data'
scene_names = os.listdir(data_root)
# scene_names = ['scene_03_02_000002']

for scene in scene_names:
    print('processing:', scene)

    scene_path = os.path.join(data_root, scene)
    dataset = OpticalFlowDatasetEvents(scene_path, dt=1/60.0, diff_imu_t_data_t=-0.02)

    os.system(f'mkdir -p plot/{scene}/xyz')
    os.system(f'mkdir -p plot/{scene}/flow')

    map1, map2 = cv2.initInverseRectificationMap(
        dataset.K, # Intrinsics
        dataset.D, # Distortion
        np.eye(3), # Rectification
        np.eye(3), # New intrinsics
        (640, 480), # Size of the image
        cv2.CV_32FC1)

    precomputed_A_x = np.zeros((*map1.shape, 2, 3))
    precomputed_A_x[:, :, 0, 0] = -1
    precomputed_A_x[:, :, 1, 1] = -1
    precomputed_A_x[:, :, 0, 2] = map1
    precomputed_A_x[:, :, 1, 2] = map2

    precomputed_B_x = np.zeros((*map1.shape, 2, 3))
    precomputed_B_x[:, :, 0, 0] = map1 * map2
    precomputed_B_x[:, :, 0, 1] = -(np.square(map1) + 1)
    precomputed_B_x[:, :, 0, 2] = map2
    precomputed_B_x[:, :, 1, 0] = (np.square(map2) + 1)
    precomputed_B_x[:, :, 1, 1] = -map1 * map2
    precomputed_B_x[:, :, 1, 2] = -map1

    indices, all_gts, norm_v_c1_gt = [], [], []

    methods = [
        {'name': 'min neg cher', 'f': minimize_negative_cheraility, 'preds': []},
        {'name': 'svm', 'f': svm_positivity, 'preds': []},
    ]

    ws_imu = []
    ws_gt = []

    for i in tqdm(range(1, len(dataset))):
        times = []
        times.append(time.time())

        cur_item = dataset[i]

        if cur_item is None:
            continue
        left_time = cur_item['left_time']
        right_time = cur_item['right_time']
        T_wc1 = cur_item['T_wc1']
        T_wc2 = cur_item['T_wc2']
        d_normalized = cur_item['flow']
        xy = cur_item['xy']
        all_xy = cur_item['all_xy']
        indices.append(left_time)
        times.append(time.time())

        v_c1, w_c2, norm_v_c1 = egomotion_from_pose(T_wc1, T_wc2)
        w_imu = cur_item['imu_angular_velocity']
        all_gts.append(v_c1)
        ws_gt.append(w_c2)
        norm_v_c1_gt.append(norm_v_c1)
        ws_imu.append(w_imu)

        times.append(time.time())

        # w_positivity = w_c2
        w_positivity = w_imu
        g_x_A_x, n_x_g_x_B_x_w = form_positivity_matrices_with_w(precomputed_A_x, precomputed_B_x, xy, d_normalized, w_positivity)
        times.append(time.time())

        for j, method in enumerate(methods):
            if len(method['preds']) > 0:
                if not np.any(np.isnan(method['preds'][-1])):
                    v0 = method['preds'][-1]
                else:
                    v0 = None
                # print(method, v0)
            else:
                v0 = None
            # v0 = None
            v_c1_pred = method['f'](g_x_A_x, n_x_g_x_B_x_w, v0)
            method['preds'].append(v_c1_pred)
        times.append(time.time())

    norm_v_c1_gt = np.array(norm_v_c1_gt)
    ws_gt = np.array(ws_gt)
    ws_imu = np.array(ws_imu)

    # Save data for quantitative evalution
    save_dict = {}
    save_dict['t'] = indices
    save_dict['ws_gt'] = ws_gt
    save_dict['ws_imu'] = ws_imu
    save_dict['v_c1_gt'] = np.array(all_gts)
    save_dict['norm_v_c1_gt'] = norm_v_c1_gt
    for method in methods:
        save_dict['v_c1_' + method['name']] = np.array(method['preds'])
    np.savez(f'plot/{scene}/egomotion_data.npz', **save_dict)


    plt.figure(figsize=(20, 10))
    plt.subplot(3,1,1)
    plt.title(f'Velocity Direction (XYZ) Prediction vs Ground Truth', fontsize=20)

    for method in methods:
        plt.plot(np.array(indices), np.array(method['preds'])[:, 0], label=method['name'], linewidth=2)
    plt.plot(np.array(indices), np.array(all_gts)[:, 0], label='Ground Truth', color='black', linewidth=2)
    plt.xlim(dataset.t[0], dataset.t_end[i])
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,2)
    for method in methods:
        plt.plot(np.array(indices), np.array(method['preds'])[:, 1], label=method['name'], linewidth=2)
    plt.plot(np.array(indices), np.array(all_gts)[:, 1], label='gt', color='black', linewidth=2)
    plt.xlim(dataset.t[0], dataset.t_end[i])
    plt.ylim(-1, 1)
    plt.grid(True)
    
    plt.subplot(3,1,3)
    for method in methods:
        plt.plot(np.array(indices), np.array(method['preds'])[:, 2], label=method['name'], linewidth=2)
    plt.plot(np.array(indices), np.array(all_gts)[:, 2], label='gt', color='black', linewidth=2)
    plt.xlim(dataset.t[0], dataset.t_end[i])
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'plot/{scene}/xyz/{str(i).zfill(6)}.jpg')
    plt.savefig(f'plot/{scene}/xyz.jpg')

    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(np.array(indices), ws_gt[:, 0], label='w gt x')
    # plt.plot(np.array(indices), ws_imu[:, 0], label='w imu x')
    # plt.legend()
    # plt.subplot(3,1,2)
    # plt.plot(np.array(indices), ws_gt[:, 1], label='w gt y')
    # plt.plot(np.array(indices), ws_imu[:, 1], label='w imu y')
    # plt.subplot(3,1,3)
    # plt.plot(np.array(indices), ws_gt[:, 2], label='w gt y')
    # plt.plot(np.array(indices), ws_imu[:, 2], label='w imu y')

    plt.show()
    # plt.close()
