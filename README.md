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

For users that want larger maximum dimension support (currently set to 8), modify `build_info.rs` and compile.

#### Direct porting of `QuickFPS`

See `src/bucket_fps/c_warpper.cpp` and `src/bucket_fps/_ext/` for details.

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
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3)
```

* `FPS`: Vanilla farthest point sampling. Implemented in Rust. Achieve the same performance as `numpy`.
* `FPS + NPDU`: Farthest point sampling with nearest-point-distance-updating (NPDU) heuristic strategy. 5x~10x faster than vanilla FPS. **Require dimensional locality and give sub-optimal answers**.
* `FPS + NPDU + KDTree`: Farthest point sampling with NPDU heuristic strategy and KDTree. 3x~8x faster than vanilla FPS. Slightly slower than `FPS + NPDU`. But **DOES NOT** require dimensional locality.
* `KDTree-based FPS`: A farthest point sampling algorithm based on KDTree. About 40~50x faster than vanilla FPS.
* `Bucket-based FPS` or `QuickFPS`: A bucket-based farthest point sampling algorithm. About 80~100x faster than vanilla FPS. Require an additional hyperparameter for the height of the KDTree. In practice, `h=3` or `h=5` is recommended for small data, `h=7` is recommended for medium data, and `h=9` for extremely large data.

> **NOTE**: In most cases, `Bucket-based FPS` is the best choice, with proper hyperparameter setting.

### Determinism

For deterministic results, fix the first sampled point index by passing the `start_idx` parameter.
```python
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3, start_idx=0)
```

**OR** set the random seed before calling the function.
```python
np.random.seed(42)
```

## Performance
Setup:
  - CPU: Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
  - RAM: 128 GiB
  - SYSTEM: Ubuntu 22.04.3 LTS

Run benchmark:
```shell
pytest bench/ --benchmark-columns=mean,stddev --benchmark-sort=mean
```

Results:
```
---------------- benchmark '1024 of 4096': 7 tests -----------------
Name (time in ms)                   Mean            StdDev
--------------------------------------------------------------------
test_bucket_fps_kdline_4k_h5      1.9469 (1.0)      0.0354 (1.54)
test_bucket_fps_kdline_4k_h3      2.0028 (1.03)     0.0750 (3.27)
test_fps_npdu_4k                  3.3361 (1.71)     0.0229 (1.0)
test_bucket_fps_kdline_4k_h7      3.6899 (1.90)     0.0548 (2.39)
test_bucket_fps_kdtree_4k         6.5072 (3.34)     0.4018 (17.52)
test_fps_npdu_kdtree_4k          12.3689 (6.35)     0.0380 (1.66)
test_vanilla_fps_4k              14.1073 (7.25)     0.4171 (18.20)
--------------------------------------------------------------------

----------------- benchmark '4096 of 50000': 7 tests -----------------
Name (time in ms)                     Mean            StdDev
----------------------------------------------------------------------
test_bucket_fps_kdline_50k_h7      25.7244 (1.0)      0.5605 (1.0)
test_bucket_fps_kdline_50k_h5      30.0820 (1.17)     0.5973 (1.07)
test_bucket_fps_kdline_50k_h3      59.9939 (2.33)     1.0208 (1.82)
test_bucket_fps_kdtree_50k         98.2151 (3.82)     5.1610 (9.21)
test_fps_npdu_50k                 129.3240 (5.03)     0.5638 (1.01)
test_fps_npdu_kdtree_50k          287.4457 (11.17)    8.5040 (15.17)
test_vanilla_fps_50k              794.4958 (30.88)    5.2105 (9.30)
----------------------------------------------------------------------

------------------- benchmark '50000 of 100000': 7 tests -------------------
Name (time in ms)                         Mean              StdDev
----------------------------------------------------------------------------
test_bucket_fps_kdline_100k_h7        247.6833 (1.0)        4.8640 (6.85)
test_bucket_fps_kdline_100k_h5        393.8612 (1.59)       3.8099 (5.37)
test_bucket_fps_kdtree_100k           419.4466 (1.69)       8.5836 (12.09)
test_bucket_fps_kdline_100k_h9        437.0670 (1.76)       2.8537 (4.02)
test_fps_npdu_100k                  2,990.6574 (12.07)      0.7101 (1.0)
test_fps_npdu_kdtree_100k           4,236.8786 (17.11)      3.3208 (4.68)
test_vanilla_fps_100k              20,131.7747 (81.28)    155.4407 (218.91)
----------------------------------------------------------------------------
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
