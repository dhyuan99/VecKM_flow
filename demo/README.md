# API Usage Demo

## Demo Data
| File        | Description | Shape  |
|-------------|-----|-------------|
| `dataset_events_p.npy`    | event polarity  | `(n, )` short   |
| `dataset_events_t.npy`  | event time in seconds | `(n, )` float64    |
| `undistorted_events_xy.npy` | event coordinates. 1st row is width, 2nd row is height.  | `(n, 2)` float32      |

The data are consistent as the description in [README](../README.md/#api-usage). Polarity is not used in our model, while we keep it in case users want to compare against other methods.

Please ensure the pixel coordinates are transformed into undistorted normalized coordinates as instructed at [README](../README.md/#undistorted-normalized-coordinates).

## Other Data
The precomputed inputs for [MVSEC](https://daniilidis-group.github.io/mvsec/), [DSEC](https://dsec.ifi.uzh.ch), [EVIMO](https://better-flow.github.io/evimo/download_evimo_2.html) can be downloaded from [this drive](). The data format is exactly the same as the demo data.

## Inference
```
python main.py
```
It will generate flow predictions and uncertainties. It also generates a flow prediction video at `output.mp4`. It is helpful to read the comments at [main.py](./main.py) to understand the usage.