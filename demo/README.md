# API Usage Demo

## Demo Data
| File        | Description | Shape  |
|-------------|-----|-------------|
| `dataset_events_p.npy`    | event polarity  | `(n, )` int   |
| `dataset_events_t.npy`  | event time in seconds | `(n, )` float64    |
| `undistorted_events_xy.npy` | event coordinates. 1st row is width, 2nd row is height.  | `(n, 2)` float32      |
The data are consistent as the description in [README](../README.md/#api-usage).