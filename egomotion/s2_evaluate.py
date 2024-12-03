import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

data_root = './plot'
scene_names = os.listdir(data_root)

for scene in scene_names:
    print(scene)
    scene_path = os.path.join(data_root, scene)
    data_path = os.path.join(scene_path, 'egomotion_data.npz')
    if not os.path.exists(data_path):
        continue
    data = np.load(data_path)

    methods = []
    for key in data:
        if 'v_c1_' in key and 'v_c1_gt' not in key:
            methods.append(key[len('v_c1_'):])

    max_methods_len = np.max([len(method) for method in methods])

    t = data['t']
    t = t - t[0] # start time from 0 for plots
    ws_gt = data['ws_gt']
    ws_imu = data['ws_imu']
    v_c1_gt = data['v_c1_gt']
    norm_v_c1_gt = data['norm_v_c1_gt']
    norm_v_c1_gt = norm_v_c1_gt.reshape((-1, 1))

    v_c1_gt_meters = v_c1_gt * norm_v_c1_gt

    for method in methods:
        # RMS velocity m/s
        v_est = data['v_c1_' + method]
        # All V's are normlized so multiply by ground truth magnitude m/s
        rms_v = rms(v_est * norm_v_c1_gt - v_c1_gt_meters)

        # RMS omegs deg/s
        ws_imu_deg = ws_imu * 180.0 / np.pi
        ws_gt_deg  = ws_gt  * 180.0 / np.pi
        rms_w = rms(ws_imu_deg - ws_gt_deg)

        method_print = method.ljust(max_methods_len)
        print('{} {:0.3f} m/s {:0.3f} deg/s'.format(method_print, rms_v, rms_w))

    fig = plt.figure()
    rows = 3
    cols = 1

    ylabels = ['$v_x$ (m/s)', '$v_y$ (m/s)', '$v_z$ (m/s)']
    method_labels = {
        'min neg cher': '$-Z$',
        'svm': 'SVM',
    }
    method_colors = {
        'min neg cher': 'blue',
        'svm': 'red',
    }
    for i in range(3):
        ax = plt.subplot(rows, cols, i+1)
        for method in methods:
            v_est = data['v_c1_' + method]
            v_est_meters = v_est * norm_v_c1_gt
            plt.plot(t, v_est_meters[:, i], label=method_labels[method], color=method_colors[method])
        # plt.xlim(xlim)
        plt.plot(t, v_c1_gt_meters[:, i], label='GT', color='black')
        plt.ylim([-1.75, 1.75])
        plt.ylabel(ylabels[i])
        plt.grid()


        if i < 2:
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

    ax = plt.subplot(rows, cols, 3)
    plt.xlabel('Time (seconds)')

    # Put a legend below last axis
    ax = plt.subplot(rows, cols, 1)
    leg = ax.legend(loc='upper right')

    fig.set_size_inches(6, 6)
    fig.tight_layout()
    
    # Move legend up slightly
    # https://stackoverflow.com/a/23254538
    plt.draw()
    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    # Change to location of the legend. 
    # xOffset = 1.5
    # bb.x0 += xOffset
    # bb.x1 += xOffset
    yOffset = 0.05
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

    fig.tight_layout()
    fig.savefig(os.path.join(data_root, f'{scene}.png'), dpi=300)
    plt.close()
    # plt.show()
    # exit(0)
