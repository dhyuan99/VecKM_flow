# Egomotion Estimation on EVIMO2

Please download the data from [this drive](https://drive.google.com/drive/folders/1vOQE6Jm42_kphxF7t-TG-W6iuzqRaUg1?usp=sharing) and place them under `data/*`. For each scene `data/{scene_name}`, it contains the following file:
* `dataset_events_t.npy`: sorted event time in seconds.
* `dataset_events_xy.npy`: event pixels in raw image coordinates.
* `dataset_pred_flow.npy`: predicted per-event normal flow in raw image coordinates.
* `dataset_pred_uncertainty.npy`: prediction uncertainty.
* `dataset_info.npz`: contains camera matrix `K`, distortion coefficients `D`, and IMU measurement.

After running
```
python s1_main.py
python s2_evaluate.py
```
it will generate plots of egomotion GT v.s. predictions at [`plot/`](plot).