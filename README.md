# fpsample
[![pypi package version badge](https://img.shields.io/pypi/v/fpsample)](https://pypi.org/project/fpsample/)
![python version badge](https://img.shields.io/badge/python-%3E%3D3.7-blue)
[![license badge](https://img.shields.io/github/license/leonardodalinky/fpsample)](https://github.com/leonardodalinky/fpsample/blob/main/LICENSE)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/fpsample?style=social)](https://github.com/leonardodalinky/fpsample)

Python efficient farthest point sampling (FPS) library, 100x faster than pure Python implementation.

`fpsample` is coupled with `numpy` and built upon Rust pyo3 bindings.

*Library for GPU is under construction*. Any issues are welcome.

## Installation

### Install from PyPI

`numpy>=1.16.0` is required. Install `fpsample` using pip:

```shell
pip install -U fpsample
```

### Build from source

The library is built using [maturin](https://github.com/PyO3/maturin). Ensure you have maturin, rust and cargo installed.

Build the library and install using:
```shell
maturin develop --release
```

## Usage

```python
import fpsample
import numpy as np

# Generate random point cloud
pc = np.random.rand(4096, 3)
# sample 1024 points
fps_samples_idx = fpsample.fps_sampling(pc, 1024)

fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024)
# or specify the windows size
fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024, k=64)

fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024)
# or specify the windows size
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024, k=64)
```

**NOTE**: NPDU method only gives sub-optimal answers. And it assumes that the points are approximately sorted or have **dimensional locality**. Otherwise, the result may be **worse**. Check the paper for details.

NPDU+KDTree method is more robust than NPDU method. It does not require the dimensional locality. But it is slightly slower than vanilla NPDU method. **It is recommended to use NPDU+KDTree method in general cases**.

## Performance
Setup:
  - CPU: Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
  - RAM: 128 GiB

|     Method      | #samples | #points |        Time        |
| :-------------: | :------: | :-----: | :----------------: |
|       FPS       |   1024   |  4096   | 18.6 ms ± 0.17 ms  |
|    FPS+NPDU     |   1024   |  4096   | 3.68 ms ± 0.10 ms  |
| FPS+NPDU+KDTree |   1024   |  4096   | 13.10 ms ± 0.16 ms |
|       FPS       |   4000   | 50,000  |  832 ms ± 9.01 ms  |
|    FPS+NPDU     |   4000   | 50,000  |  143 ms ± 1.98 ms  |
| FPS+NPDU+KDTree |   4000   | 50,000  |  294 ms ± 9.06 ms  |
|       FPS       |  50,000  | 100,000 |  22.1 s ± 207 ms   |
|    FPS+NPDU     |  50,000  | 100,000 |  3.35 s ± 66.8 ms  |
| FPS+NPDU+KDTree |  50,000  | 100,000 |  4.08 s ± 60.4 ms  |

## Reference
The nearest-point-distance-updating (NPDU) heuristic strategy is proposed in the following paper:
```
@INPROCEEDINGS{9919246,
  author={Li, Jingtao and Zhou, Jian and Xiong, Yan and Chen, Xing and Chakrabarti, Chaitali},
  booktitle={2022 IEEE Workshop on Signal Processing Systems (SiPS)},
  title={An Adjustable Farthest Point Sampling Method for Approximately-sorted Point Cloud Data},
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/SiPS55645.2022.9919246}
}
```
