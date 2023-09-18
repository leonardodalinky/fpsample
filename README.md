# fpsample
[![pypi package version badge](https://img.shields.io/pypi/v/fpsample)](https://pypi.org/project/fpsample/)
![python version badge](https://img.shields.io/badge/python-%3E%3D3.7-blue)
[![license badge](https://img.shields.io/github/license/leonardodalinky/fpsample)](https://github.com/leonardodalinky/fpsample/blob/main/LICENSE)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/fpsample?style=social)](https://github.com/leonardodalinky/fpsample)

Python efficient farthest point sampling (FPS) library, 100x faster than `numpy` implementation.

`fpsample` is coupled with `numpy` and built upon Rust pyo3 bindings. This library aims at achieving the best performance for FPS in single-threaded CPU environment.

*Library for GPU is under construction*. Any issues are welcome.

## Installation

### Install from PyPI

`numpy>=1.16.0` is required. Install `fpsample` using pip:

```shell
pip install -U fpsample
```

*NOTE: Only 64 bit package provided.*

If you encounter any installation errors, please make an issue and try to compile from source.

### Build from source

The library is built using [maturin](https://github.com/PyO3/maturin). Therefore, `rust` and `cargo` are required for compiling.

```shell
pip install -r requirements.txt
```

C++ compiler must support C++14. For example, `gcc>=8` or `clang>=5`.

Build the library and install using:
```shell
maturin develop --release
```

#### Compile options

For macos users, if the compilation fails to link libstdc++, try to pass `FORCE_CXXSTDLIB=c++` as an environment variable.

For users that want larger maximum dimension support (currently set to 8), check `build_info.rs` for details.

## Usage

```python
import fpsample
import numpy as np

# Generate random point cloud
pc = np.random.rand(4096, 3)
## sample 1024 points

# Vanilla FPS
fps_samples_idx = fpsample.fps_sampling(pc, 1024)

# FPS + NPDU
fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024)
## or specify the windows size
fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024, k=64)

# FPS + NPDU + KDTree
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024)
## or specify the windows size
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024, k=64)

# KDTree-based FPS
kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 1024)

# Bucket-based FPS or QuickFPS
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=7)
```

* `FPS`: Vanilla farthest point sampling. Implemented in Rust. Achieve the same performance as `numpy`.
* `FPS + NPDU`: Farthest point sampling with nearest-point-distance-updating (NPDU) heuristic strategy. 5x~10x faster than vanilla FPS. **Require dimensional locality and give sub-optimal answers**.
* `FPS + NPDU + KDTree`: Farthest point sampling with NPDU heuristic strategy and KDTree. 3x~8x faster than vanilla FPS. Slightly slower than `FPS + NPDU`. But **DOES NOT** require dimensional locality.
* `KDTree-based FPS`: A farthest point sampling algorithm based on KDTree. About 50x faster than vanilla FPS.
* `Bucket-based FPS` or `QuickFPS`: A bucket-based farthest point sampling algorithm. About 100x faster than vanilla FPS. Require an additional hyperparameter for the height of the KDTree. In practice, `h=3` is recommended for small data, `h=7` is recommended for medium data, and `h=9` for extremely large data.

**NOTE**: In most cases, `Bucket-based FPS` is the best choice, with proper hyperparameter setting.

## Performance
Setup:
  - CPU: Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
  - RAM: 128 GiB
  - SYSTEM: Ubuntu 22.04.3 LTS

Run benchmark:
```shell
pytest bench/ --benchmark-columns=min,mean,stddev
```

Results:
```
-------------------------- benchmark '1024 of 4096': 5 tests --------------------------
Name (time in ms)                    Min               Mean            StdDev
---------------------------------------------------------------------------------------
test_bucket_fps_kdline_4k_h3      2.0043 (1.0)       2.3166 (1.0)      0.3812 (5.51)
test_fps_npdu_4k                  3.5585 (1.78)      3.7348 (1.61)     0.0691 (1.0)
test_bucket_fps_kdtree_4k         6.4947 (3.24)      7.0000 (3.02)     0.4249 (6.15)
test_fps_npdu_kdtree_4k          13.2702 (6.62)     13.9802 (6.03)     0.3151 (4.56)
test_vanilla_fps_4k              14.3000 (7.13)     15.0144 (6.48)     0.3563 (5.15)
---------------------------------------------------------------------------------------

---------------------------- benchmark '4096 of 50000': 5 tests ---------------------------
Name (time in ms)                      Min                Mean             StdDev
-------------------------------------------------------------------------------------------
test_bucket_fps_kdline_50k_h7      23.8785 (1.0)       25.8189 (1.0)       1.0348 (1.0)
test_bucket_fps_kdtree_50k         90.6234 (3.80)      99.7299 (3.86)      4.8106 (4.65)
test_fps_npdu_50k                 140.5237 (5.88)     146.9772 (5.69)      5.3462 (5.17)
test_fps_npdu_kdtree_50k          315.5046 (13.21)    324.7891 (12.58)     8.2886 (8.01)
test_vanilla_fps_50k              900.7202 (37.72)    916.6968 (35.50)    11.2134 (10.84)
-------------------------------------------------------------------------------------------

------------------------------ benchmark '50000 of 100000': 5 tests ------------------------------
Name (time in ms)                          Min                   Mean             StdDev
--------------------------------------------------------------------------------------------------
test_bucket_fps_kdline_100k_h7        262.8185 (1.0)         270.1024 (1.0)       7.9086 (1.14)
test_bucket_fps_kdtree_100k           436.0765 (1.66)        441.7569 (1.64)      6.9266 (1.0)
test_fps_npdu_100k                  3,325.4431 (12.65)     3,341.8474 (12.37)    16.0163 (2.31)
test_fps_npdu_kdtree_100k           4,698.6172 (17.88)     4,712.4720 (17.45)    12.0861 (1.74)
test_vanilla_fps_100k              23,520.6776 (89.49)    23,542.1331 (87.16)    22.5513 (3.26)
--------------------------------------------------------------------------------------------------
```

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

Bucket-based farthest point sampling (QuickFPS) is proposed in the following paper. The implementation is based on the author's [Repo](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU). To port the implementation to other C++ program, check [this](https://github.com/leonardodalinky/fpsample/tree/main/src/bucket_fps/_ext) for details.
```
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great work.
