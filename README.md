# fpsample
[![pypi package version badge](https://img.shields.io/pypi/v/fpsample)](https://pypi.org/project/fpsample/)
![python version badge](https://img.shields.io/badge/python-%3E%3D3.7-blue)
[![license badge](https://img.shields.io/github/license/leonardodalinky/fpsample)](https://github.com/leonardodalinky/fpsample/blob/main/LICENSE)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/fpsample?style=social)](https://github.com/leonardodalinky/fpsample)

Python farthest point sampling library built on Rust pyo3 bindings. Compatible with numpy.

## Installation

### Install from PyPI

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
fps_samples = fpsample.fps_sampling(pc, 1024)

fps_npdu_samples = fpsample.fps_npdu_sampling(pc, 1024)
# or specify the windows size
fps_npdu_samples = fpsample.fps_npdu_sampling(pc, 1024, k=64)
```

> NOTE: NPDU method only gives sub-optimal answers. And it assumes that the points are approximately sorted or have **dimensional locality**. Otherwise, the result may be **worse**. Check the paper for details.

## Performance
Setup:
  - CPU: Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
  - RAM: 128 GiB

|  Method  | #samples | #points |        Time       |
|:--------:|:--------:|:-------:|:-----------------:|
|    FPS   |   1024   |   4096  |  18.6 ms ± 1.3 ms |
| FPS+NPDU |   1024   |   4096  | 5.97 ms ± 0.61 ms |
|    FPS   |  50,000  | 100,000 |  27.4 s ± 551 ms  |
| FPS+NPDU |  50,000  | 100,000 |  5.36 s ± 152 ms  |

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
