import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

def images_to_video(image_folder, output_video, fps):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in tqdm(images, desc=f"generating video, saved to {output_video}"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
def gen_flow_video(
    events_t, events_xy, events_flow, 
    frames_folder, video_path, 
    fps=30, max_events=100000, frame_reso=1):
    xlim = (np.min(events_xy[:, 0]), np.max(events_xy[:, 0]))
    ylim = (np.min(events_xy[:, 1]), np.max(events_xy[:, 1]))
    t_reso, t_min, t_max = 1./fps, np.min(events_t), np.max(events_t)
    
    os.system(f"rm -rf {frames_folder}")
    os.system(f"mkdir -p {frames_folder}")

    for idx, t in enumerate(tqdm(
        np.arange(t_min, t_max, t_reso), 
        desc=f"generating frames, saved to {frames_folder}"
        )):
        start_idx = np.searchsorted(events_t, t)
        end_idx = np.searchsorted(events_t, t + t_reso)
        slice_idx = np.arange(start_idx, end_idx)

        xy = events_xy[slice_idx]
        flow = events_flow[slice_idx]
        if len(xy) > max_events:
            r = np.random.permutation(len(xy))[:max_events]
            flow = flow[r]
            xy = xy[r]
        flow_norm = np.linalg.norm(flow, axis=1)
        flow_angle = np.arctan2(flow[:, 1], flow[:, 0])
        valid_indices = np.where(flow_norm > 0)[0]
        num_valid = len(valid_indices)

        plt.figure(figsize=(18*frame_reso, 6*frame_reso))
        plt.subplot(1, 3, 1)
        plt.scatter(xy[:, 0], xy[:, 1], color='grey', s=1, alpha=0.1)
        plt.text(0.01, 0.95, f"Flow Norm", transform=plt.gca().transAxes, fontsize=15, color='black')
        if num_valid > 0:
            plt.scatter(
                xy[valid_indices, 0], xy[valid_indices, 1], c=flow_norm[valid_indices], 
                s=1, vmin=0, vmax=1, cmap='plasma')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks([]); plt.yticks([])
        plt.gca().invert_yaxis()
        plt.subplot(1, 3, 2)
        plt.scatter(xy[:, 0], xy[:, 1], color='grey', s=1, alpha=0.1)
        plt.text(0.01, 0.95, f"Flow Angle", transform=plt.gca().transAxes, fontsize=15, color='black')
        if num_valid > 0:
            plt.scatter(
                xy[valid_indices, 0], xy[valid_indices, 1], c=flow_angle[valid_indices], 
                s=1, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks([]); plt.yticks([])
        plt.gca().invert_yaxis()
        plt.subplot(1, 3, 3)
        plt.scatter(xy[:,0], xy[:,1], s=1, color='grey', alpha=0.1)
        plt.text(0.01, 0.95, f"Flow", transform=plt.gca().transAxes, fontsize=15, color='black')
        if num_valid > 0:
            valid_indices = np.random.choice(valid_indices, min(5000, num_valid), replace=False)
            plt.quiver(
                xy[valid_indices,0], xy[valid_indices,1], 
                flow[valid_indices,0], -flow[valid_indices,1], 
                color='r', scale=10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks([]); plt.yticks([])
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{frames_folder}/{idx:06d}.jpg")
        plt.close()
        
    images_to_video(frames_folder, video_path, fps)