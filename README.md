<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Learning Normal Flow Directly from Event Neighborhoods </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://www.cs.umd.edu/~dhyuan" target="_blank" style="text-decoration: none;">Dehao Yuan</a>&nbsp;,&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Levi Burner</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Jiayi Wu</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Minghui Liu</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Jingxi Chen</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Yiannis Aloimonos</a>&nbsp;&nbsp;
    <a href="http://users.umiacs.umd.edu/~fer/" target="_blank" style="text-decoration: none;">Cornelia Fermüller</a>
</p>

<p align='center';>
<b>
<em>CVPR2025</em> &nbsp&nbsp&nbsp&nbsp <a href="http://arxiv.org/abs/2404.01568" target="_blank" style="text-decoration: none;">[Paper]</a>
</b>
</p>

## API Usage
It is a generic normal flow estimator with event camera inputs. The API is easily used by following codes, after [installing the package](#installation). See [demo](./demo/) for the codes and data for running the demo.
``` python
from VecKM_flow.inference import VecKMNormalFlowEstimator
from VecKM_flow.visualize import gen_flow_video

estimator = VecKMNormalFlowEstimator()
flow_predictions, flow_uncertainty = estimator.inference(events_t, undistorted_events_xy)
flow_predictions[flow_uncertainty > 0.3] = np.nan

gen_flow_video(
    events_t.numpy(), 
    undistorted_events_xy.numpy(), 
    flow_predictions.numpy(), 
    './frames', './output.mp4', fps=30)
```

The data dimensions are as followed:
| Variables        | Description | Shape  |
|-------------|-----|-------------|
| `events_t`  | Sorted event time in seconds | `(n, )` float64    |
| `undistorted_events_xy` | Undistorted normalized event coordinates (focal length one). The range shall be around (-1, 1). See [Undistorted Normalized Coordinates](#undistorted-normalized-coordinates) for computing them. 1st row is width, 2nd row is height.  | `(n, 2)` float32      |
| `flow_predictions` | Predicted normal flow. Unit: undistorted normalized pixels per second. | `(n, 2)` float32      |
| `flow_uncertainty` | Prediction uncertainty. | `(n, )` float32 >= 0 |

The prediction is visualized as a video like this:
<div align="center">
<img src="assets/demo.gif" alt="Watch the video">
</div>

## Installation
```
git clone https://github.com/dhyuan99/VecKM_flow.git
cd VecKM_flow
conda create -n VecKM_flow python=3.13
conda activate VecKM_flow
pip install --upgrade pip setuptools wheel
python setup.py sdist bdist_wheel
pip install .
```

## Undistorted Normalized Coordinates
To obtain the undistorted normalized coordinates, one needs to utilize `cv2.undistortPoints` and obtain the intrinsic camera matrix `K` and distortion coefficient `D` from the dataset.
``` python
def get_undistorted_events_xy(raw_events_xy, K, D):
    # raw_events_xy has shape (n, 2). The range is e.g. (0, 640) int X (0, 480) int.
    raw_events_xy = raw_events_xy.astype(np.float32)
    undistorted_normalized_xy = cv2.undistortPoints(raw_events_xy.reshape(-1, 1, 2), K, D)
    undistorted_normalized_xy = undistorted_normalized_xy.reshape(-1, 2)
    return undistorted_normalized_xy
```

## Evaluated Datasets
**[Recommended to Watch]** We evaluated the estimator on [MVSEC](https://daniilidis-group.github.io/mvsec/), [DSEC](https://dsec.ifi.uzh.ch), [EVIMO](https://better-flow.github.io/evimo/download_evimo_2.html). The flow prediction videos of every evaluated scene can be found here: 

<div align="center">
    <a href="https://drive.google.com/drive/folders/1gkmUyZX5VRf8DxiBKL9CSdWdifjqZVq3?usp=sharing" target="_blank">
    <img src="assets/video_icon.png" alt="Watch the video" width="300">
    </a>
</div>

**[Reproduce the inference]** We precompute the undistorted normalized coordinates. They can be downloaded in [this drive](). The data format is exactly the same as the [demo data](demo/demo_data). Therefore, it is straight-forward to use the API to inference the datasets.