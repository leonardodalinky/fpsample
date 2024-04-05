import warnings
from typing import Optional

import numpy as np

from .fpsample import (
    _bucket_fps_kdline_sampling,
    _bucket_fps_kdtree_sampling,
    _fps_npdu_kdtree_sampling,
    _fps_npdu_sampling,
    _fps_sampling,
)


def fps_sampling(pc: np.ndarray, n_samples: int, start_idx: Optional[int] = None) -> np.ndarray:
    """
    Vanilla FPS sampling.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        start_idx (int, default=None): The starting index of sampling. If set to None, it will be randomly picked.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    assert (
        start_idx is None or 0 <= start_idx < n_pts
    ), "start_idx should be None or 0 <= start_idx < n_pts"
    pc = pc.astype(np.float32)
    # best performance with fortran array
    pc = np.asfortranarray(pc)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts) if start_idx is None else start_idx
    return _fps_sampling(pc, n_samples, start_idx)


def fps_npdu_sampling(
    pc: np.ndarray, n_samples: int, w: Optional[int] = None, start_idx: Optional[int] = None
) -> np.ndarray:
    """
    FPS sampling with nearest-point-distance-updating (NPDU) heuristic strategy.
    **Requires dimensional locality for best samples**.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        w (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
        start_idx (int, default=None): The starting index of sampling. If set to None, it will be randomly picked.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    assert (
        start_idx is None or 0 <= start_idx < n_pts
    ), "start_idx should be None or 0 <= start_idx < n_pts"
    pc = pc.astype(np.float32)
    w = w or int(n_pts / n_samples * 16)
    if w >= n_pts - 1:
        warnings.warn(f"k is too large, set to {n_pts - 1}")
        w = n_pts - 1
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts) if start_idx is None else start_idx
    return _fps_npdu_sampling(pc, n_samples, w, start_idx)


def fps_npdu_kdtree_sampling(
    pc: np.ndarray, n_samples: int, w: Optional[int] = None, start_idx: Optional[int] = None
) -> np.ndarray:
    """
    FPS sampling with nearest-point-distance-updating (NPDU) heuristic strategy.
    Using KDTree to eliminate the need of dimensional locality.
    Slower than `fps_npdu_sampling` but more robust.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        w (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
        start_idx (int, default=None): The starting index of sampling. If set to None, it will be randomly picked.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    assert (
        start_idx is None or 0 <= start_idx < n_pts
    ), "start_idx should be None or 0 <= start_idx < n_pts"
    pc = pc.astype(np.float32)
    w = w or int(n_pts / n_samples * 16)
    if w >= n_pts:
        warnings.warn(f"k is too large, set to {n_pts}")
        w = n_pts
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts) if start_idx is None else start_idx
    return _fps_npdu_kdtree_sampling(pc, n_samples, w, start_idx)


def bucket_fps_kdtree_sampling(
    pc: np.ndarray, n_samples: int, start_idx: Optional[int] = None
) -> np.ndarray:
    """
    Bucket-based FPS sampling using KDTree. Also called "QuickFPS" in the paper.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        k (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
        start_idx (int, default=None): The starting index of sampling. If set to None, it will be randomly picked.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    assert (
        start_idx is None or 0 <= start_idx < n_pts
    ), "start_idx should be None or 0 <= start_idx < n_pts"
    pc = pc.astype(np.float32)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts) if start_idx is None else start_idx
    return _bucket_fps_kdtree_sampling(pc, n_samples, start_idx)


def bucket_fps_kdline_sampling(
    pc: np.ndarray, n_samples: int, h: int, start_idx: Optional[int] = None
) -> np.ndarray:
    """
    Bucket-based FPS sampling using KDTree, with multiple points in each bucket. Also called "QuickFPS" in the paper.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        h (int, default=None): Height of KDTree. The bucket size is `2**h`.
            According to the paper, for small workload, h=3 is enough;
            for medium workload, h=7 is enough; for large workload, h=9 is enough.
        start_idx (int, default=None): The starting index of sampling. If set to None, it will be randomly picked.

    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    assert h >= 1, "h should be >= 1"
    assert 2**h <= n_pts, "2**h should be <= n_pts"
    assert (
        start_idx is None or 0 <= start_idx < n_pts
    ), "start_idx should be None or 0 <= start_idx < n_pts"
    pc = pc.astype(np.float32)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts) if start_idx is None else start_idx
    return _bucket_fps_kdline_sampling(pc, n_samples, h, start_idx)
