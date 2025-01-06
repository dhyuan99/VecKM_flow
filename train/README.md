## Training Codes for VecKM_flow

### Training Data
Due to the storage limit, I only upload the processed training data from EVIMO. They can be downloaded [here](https://drive.google.com/drive/folders/1FN7hptJrLd_UvWd-uSgYGBFW0dTCLcXP?usp=sharing) [10GB]. Please put them in `./` so that the file structure looks like 
```
.
├── checkpoints
│   └── 20250104_114549
│       ├── EVIMO.pth
│       └── EVIMO.txt
├── model_checkpoint
│   ├── EVIMO.pth
│   └── EVIMO.txt
├── EVIMO [PUT DATA HERE]
│   ├── train
│   ├── eval
├── README.md
├── s0_dataset.py
├── s0_model.py
├── s0_params.py
├── s1_train.py
├── s2_inference.py
└── s3_visualize.py
```
#### To make your own training data,
The training data contains the following files. If you make your data into the following format, it shall work as well.
| Variables        | Description | Data Dimension  |
|-------------|-----|-------------|
| `dataset_events_t.npy`  | Sorted event time in seconds | `(n, )` float64    |
| `undistorted_events_xy.npy` | Undistorted normalized event coordinates (focal length one). The range shall be around (-1, 1). See [Undistorted Normalized Coordinates](#undistorted-normalized-coordinates) for computing them. 1st row is width, 2nd row is height.  | `(n, 2)` float32      |
| `undistorted_optical_flow.npy` | GT optical flow. Unit: undistorted normalized pixels per second. | `(n, 2)` float32      |

### Training
```
python s1_train.py --dataset EVIMO
```
This will load the `train` and `eval` scenes into memory (it would require a large CPU memory). The model will be trained on the training scenes and validated on the eval scenes.

The model checkpoint will be saved at `checkpoints/{timestamp}`. After the model converges, please copy the corresponding `EVIMO.pth` file into `model_checkpoint/EVIMO.pth`. Then we can run the inference code:
```
python s2_inference.py --dataset EVIMO --model_name EVIMO
```

This will save the predictions into `./EVIMO/{scene_name}`, which consists of `dataset_pred_flow_EVIMO.npy` and `dataset_angle_vars_flow_EVIMO.npy`. They are the flow predictions and uncertainty scores. Then we can generate the flow prediction videos by 
```
python s3_visualize.py --dataset EVIMO --model_name EVIMO
```