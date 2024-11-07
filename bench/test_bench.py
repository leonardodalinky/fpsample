import numpy as np
import pytest

import fpsample

TEST_SEED = 42
TEST_BENCHMARK_SETTINGS = {
    "4k": {"group": "1024 of 4096", "warmup": False},
    "50k": {"group": "4096 of 50000", "warmup": False},
    "100k": {"group": "50000 of 100000", "warmup": False, "min_rounds": 3},
}
TEST_CASE_SETTINGS = {
    "4k": (4096, 1024, 3),
    "50k": (50000, 4096, 3),
    "100k": (100_000, 50_000, 3),
}


def create_sample_data(n_points: int, n_dim: int = 3, seed: int = TEST_SEED):
    np.random.seed(seed)
    return np.random.rand(n_points, n_dim)


####################
#                  #
#    4k setting    #
#                  #
####################
@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_vanilla_fps_4k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_sampling, pc, n_samples)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_vanilla_fps_4k_multiple(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_sampling, pc, n_samples, start_idx=[0, 23, 64, 128])


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_fps_npdu_4k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_fps_npdu_kdtree_4k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_bucket_fps_kdtree_4k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_bucket_fps_kdline_4k_h3(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 3)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_bucket_fps_kdline_4k_h5(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 5)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["4k"])
def test_bucket_fps_kdline_4k_h7(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["4k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 7)


#####################
#                   #
#    50k setting    #
#                   #
#####################
@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_vanilla_fps_50k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_fps_npdu_50k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_fps_npdu_kdtree_50k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_bucket_fps_kdtree_50k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_bucket_fps_kdline_50k_h3(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 3)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_bucket_fps_kdline_50k_h5(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 5)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["50k"])
def test_bucket_fps_kdline_50k_h7(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["50k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 7)


######################
#                    #
#    100k setting    #
#                    #
######################
@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_vanilla_fps_100k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_fps_npdu_100k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_fps_npdu_kdtree_100k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.fps_npdu_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_bucket_fps_kdtree_100k(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdtree_sampling, pc, n_samples)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_bucket_fps_kdline_100k_h5(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 5)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_bucket_fps_kdline_100k_h7(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 7)


@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["100k"])
def test_bucket_fps_kdline_100k_h9(benchmark):
    n_points, n_samples, n_dim = TEST_CASE_SETTINGS["100k"]
    pc = create_sample_data(n_points, n_dim)
    benchmark(fpsample.bucket_fps_kdline_sampling, pc, n_samples, 9)
