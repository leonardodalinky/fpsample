# fpsample
[![pypi package version badge](https://img.shields.io/pypi/v/fpsample)](https://pypi.org/project/fpsample/)
![python version badge](https://img.shields.io/badge/python-%3E%3D3.7-blue)
[![license badge](https://img.shields.io/github/license/leonardodalinky/fpsample)](https://github.com/leonardodalinky/fpsample/blob/main/LICENSE)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/fpsample?style=social)](https://github.com/leonardodalinky/fpsample)

Python efficient farthest point sampling (FPS) library, 100x faster than `numpy` implementation.

`fpsample` is coupled with `numpy` and built upon Rust pyo3 bindings. This library aims at achieving the best performance for FPS in single-threaded CPU environment.

🎉 **PyTorch version with native multithreading, batch ops, Autograd and CUDA supports is in [pytorch_fpsample](https://github.com/leonardodalinky/pytorch_fpsample).**

## Installation

### Install from PyPI

`numpy>=1.16.0` is required. Install `fpsample` using pip:

```shell
pip install -U fpsample
```

If you encounter any installation errors, please make an issue and try to compile from source.

### Build from source

Since Version 1.0.0, fpsample is built using [Scikit-build-core + pybind11](https://github.com/pybind/scikit_build_example). Therefore, you can build and install this library from source just with pip.

C++ compiler must support C++17.

CMake>=3.15 is required.

```shell
git clone https://github.com/leonardodalinky/fpsample.git
cd fpsample
pip install .
```

#### Direct porting of `QuickFPS`

See `src/warpper.hpp` and `src/_ext/` for details.

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
fps_npdu_samples_idx = fpsample.fps_npdu_sampling(pc, 1024, w=64)

# FPS + NPDU + KDTree
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024)
## or specify the windows size
fps_npdu_kdtree_samples_idx = fpsample.fps_npdu_kdtree_sampling(pc, 1024, w=64)

# KDTree-based FPS
kdtree_fps_samples_idx = fpsample.bucket_fps_kdtree_sampling(pc, 1024)

# NOTE: Probably the best
# Bucket-based FPS or QuickFPS
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3)
```

* `FPS`: Vanilla farthest point sampling. Implemented in Rust. Achieve the same performance as `numpy`.
* `FPS + NPDU`: Farthest point sampling with nearest-point-distance-updating (NPDU) heuristic strategy. 5x~10x faster than vanilla FPS. **Require dimensional locality and give sub-optimal answers**.
* `FPS + NPDU + KDTree`: Farthest point sampling with NPDU heuristic strategy and KDTree. 3x~8x faster than vanilla FPS. Slightly slower than `FPS + NPDU`. But **DOES NOT** require dimensional locality.
* `KDTree-based FPS`: A farthest point sampling algorithm based on KDTree. About 40~50x faster than vanilla FPS.
* `Bucket-based FPS` or `QuickFPS`: A bucket-based farthest point sampling algorithm. About 80~100x faster than vanilla FPS. Require an additional hyperparameter for the height of the KDTree. In practice, `h=3` or `h=5` is recommended for small data, `h=7` is recommended for medium data, and `h=9` for extremely large data.

> **NOTE**: 🔥 In most cases, `Bucket-based FPS` is the best choice, with proper hyperparameter setting.

### Determinism

For deterministic results, fix the first sampled point index by passing the `start_idx` parameter.
```python
kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, 1024, h=3, start_idx=0)
```

**OR** set the random seed before calling the function.
```python
np.random.seed(42)
```

## Development

Install dependencies:
```shell
uv sync
```

## Performance
Setup:
  - CPU: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
  - RAM: 512 GiB
  - SYSTEM: Ubuntu 20.04.6 LTS

Run benchmark:
```shell
py.test bench/ --benchmark-columns=mean,stddev --benchmark-sort=mean
```

Results:
```
------------------------------------------------------------------------------- benchmark '1024 of 4096': 8 tests -------------------------------------------------------------------------------
Name (time in ms)                    Min                Max               Mean            StdDev             Median               IQR            Outliers       OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_bucket_fps_kdline_4k_h5      1.7892 (1.0)       2.2010 (1.01)      1.8301 (1.0)      0.0273 (1.20)      1.8255 (1.0)      0.0182 (1.0)         53;31  546.4295 (1.0)         491           1
test_bucket_fps_kdline_4k_h3      2.0452 (1.14)      2.1714 (1.0)       2.0862 (1.14)     0.0227 (1.0)       2.0817 (1.14)     0.0220 (1.21)        99;34  479.3375 (0.88)        436           1
test_bucket_fps_kdline_4k_h7      3.0348 (1.70)      3.2599 (1.50)      3.1082 (1.70)     0.0391 (1.72)      3.1005 (1.70)     0.0568 (3.12)        107;4  321.7322 (0.59)        306           1
test_fps_npdu_4k                  3.9024 (2.18)      4.3511 (2.00)      3.9514 (2.16)     0.0395 (1.74)      3.9457 (2.16)     0.0283 (1.55)        29;14  253.0768 (0.46)        245           1
test_bucket_fps_kdtree_4k         6.5697 (3.67)      7.3863 (3.40)      6.7525 (3.69)     0.1150 (5.07)      6.7240 (3.68)     0.0926 (5.08)        26;11  148.0923 (0.27)        126           1
test_fps_npdu_kdtree_4k           9.4114 (5.26)      9.7609 (4.50)      9.4979 (5.19)     0.0613 (2.70)      9.4807 (5.19)     0.0362 (1.99)        14;13  105.2866 (0.19)         97           1
test_vanilla_fps_4k              12.3702 (6.91)     12.8378 (5.91)     12.5875 (6.88)     0.1840 (8.11)     12.5024 (6.85)     0.3712 (20.38)        40;0   79.4438 (0.15)         77           1
test_vanilla_fps_4k_multiple     12.4058 (6.93)     13.2153 (6.09)     12.4706 (6.81)     0.1639 (7.22)     12.4139 (6.80)     0.0268 (1.47)         7;12   80.1886 (0.15)         80           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------- benchmark '4096 of 50000': 7 tests --------------------------------------------------------------------------------
Name (time in ms)                      Min                 Max                Mean            StdDev              Median               IQR            Outliers      OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_bucket_fps_kdline_50k_h7      24.7285 (1.0)       25.2791 (1.0)       25.0179 (1.0)      0.1237 (1.0)       25.0139 (1.0)      0.1184 (1.0)          10;3  39.9714 (1.0)          40           1
test_bucket_fps_kdline_50k_h5      31.7191 (1.28)      32.9791 (1.30)      32.1800 (1.29)     0.3308 (2.68)      32.0584 (1.28)     0.3701 (3.13)          6;3  31.0752 (0.78)         31           1
test_bucket_fps_kdline_50k_h3      66.7806 (2.70)      67.5433 (2.67)      67.1429 (2.68)     0.2265 (1.83)      67.0898 (2.68)     0.2654 (2.24)          5;0  14.8936 (0.37)         15           1
test_bucket_fps_kdtree_50k         82.7767 (3.35)      85.9608 (3.40)      84.5684 (3.38)     0.9575 (7.74)      84.5108 (3.38)     1.2572 (10.62)         3;0  11.8248 (0.30)         11           1
test_fps_npdu_50k                 179.8323 (7.27)     181.9054 (7.20)     180.5911 (7.22)     0.7326 (5.92)     180.4329 (7.21)     0.7742 (6.54)          2;0   5.5374 (0.14)          6           1
test_fps_npdu_kdtree_50k          253.4978 (10.25)    255.2537 (10.10)    254.2810 (10.16)    0.7408 (5.99)     253.9220 (10.15)    1.1832 (9.99)          2;0   3.9327 (0.10)          5           1
test_vanilla_fps_50k              592.6405 (23.97)    594.1419 (23.50)    593.2049 (23.71)    0.5970 (4.83)     593.2359 (23.72)    0.7815 (6.60)          1;0   1.6858 (0.04)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------- benchmark '50000 of 100000': 7 tests ---------------------------------------------------------------------------------------
Name (time in ms)                          Min                    Max                   Mean              StdDev                 Median                 IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_bucket_fps_kdline_100k_h7        232.5230 (1.0)         234.3285 (1.0)         233.3726 (1.0)        0.7632 (1.49)        233.5817 (1.0)        1.2734 (1.66)          2;0  4.2850 (1.0)           5           1
test_bucket_fps_kdtree_100k           381.3098 (1.64)        388.0433 (1.66)        384.0891 (1.65)       3.5171 (6.84)        382.9141 (1.64)       5.0501 (6.56)          1;0  2.6036 (0.61)          3           1
test_bucket_fps_kdline_100k_h9        409.0352 (1.76)        420.7832 (1.80)        414.3774 (1.78)       5.9458 (11.57)       413.3137 (1.77)       8.8110 (11.45)         1;0  2.4133 (0.56)          3           1
test_bucket_fps_kdline_100k_h5        410.1506 (1.76)        411.1764 (1.75)        410.6451 (1.76)       0.5139 (1.0)         410.6085 (1.76)       0.7693 (1.0)           1;0  2.4352 (0.57)          3           1
test_fps_npdu_kdtree_100k           3,134.7921 (13.48)     3,137.0544 (13.39)     3,135.7064 (13.44)      1.1919 (2.32)      3,135.2726 (13.42)      1.6967 (2.21)          1;0  0.3189 (0.07)          3           1
test_fps_npdu_100k                  4,325.4402 (18.60)     4,341.7247 (18.53)     4,333.3156 (18.57)      8.1554 (15.87)     4,332.7821 (18.55)     12.2134 (15.87)         1;0  0.2308 (0.05)          3           1
test_vanilla_fps_100k              14,460.2331 (62.19)    14,664.4396 (62.58)    14,561.1574 (62.39)    102.1237 (198.73)   14,558.7995 (62.33)    153.1549 (199.07)        1;0  0.0687 (0.02)          3           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

```

## Reference
The nearest-point-distance-updating (NPDU) heuristic strategy is proposed in the following paper:
```bibtex
@INPROCEEDINGS{Li2022adjust,
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
```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

The KDTree-based NN search algorithm implemented by [nanoflann](https://github.com/jlblancoc/nanoflann)

Thanks to the authors for their great work.
