from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np

from ._fpsample import (
    __doc__,
    __version__,
    _bucket_fps_kdline_sampling,
    _bucket_fps_kdtree_sampling,
    _fps_npdu_kdtree_sampling,
    _fps_npdu_sampling,
    _fps_sampling,
)


def get_start_idx(n_pts: int, start_idx: Optional[Union[int, List[int]]]) -> Union[int, np.ndarray]:
    # Random pick a start or use the given start indices
    if start_idx is None:
        start_idx = np.random.randint(low=0, high=n_pts)
    elif isinstance(start_idx, int):
        # just use the int start_idx
        start_idx = start_idx
    elif isinstance(start_idx, list):
        # convert to numpy array for rust
        # TODO: maybe better performance with fortran array?
        start_idx = np.array(start_idx, dtype=np.uint64)
    else:
        raise ValueError("start_idx should be None, int or list")
    return start_idx


def fps_sampling(
    pc: np.ndarray, n_samples: int, start_idx: Optional[Union[int, List[int]]] = None
) -> np.ndarray:
    """
    Vanilla FPS sampling.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        start_idx (int or list[int], default=None): The starting index of sampling. If set to None, it will be randomly picked.
            If multiple start indices are given, they are used as a start set for the FPS and will all be present in the final samples.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    if isinstance(start_idx, int):
        assert (
            start_idx is None or 0 <= start_idx < n_pts
        ), "start_idx should be None or 0 <= start_idx < n_pts"
    if isinstance(start_idx, list):
        assert len(start_idx) <= n_samples, "len(start_idx) should be <= n_samples"
        for idx in start_idx:
            assert 0 <= idx < n_pts, "start_idx should be None or 0 <= start_idx < n_pts"
    # best performance with fortran array
    pc = np.asfortranarray(pc, dtype=np.float32)
    # Random pick a start if not given
    start_idx = get_start_idx(n_pts, start_idx)
    return _fps_sampling(pc, n_samples, start_idx)


def fps_npdu_sampling(
    pc: np.ndarray,
    n_samples: int,
    w: Optional[int] = None,
    start_idx: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """
    FPS sampling with nearest-point-distance-updating (NPDU) heuristic strategy.
    **Requires dimensional locality for best samples**.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        w (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
        start_idx (int or list[int], default=None): The starting index of sampling. If set to None, it will be randomly picked.
            If multiple start indices are given, they are used as a start set for the FPS and will all be present in the final samples.
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
    if isinstance(start_idx, list):
        assert len(start_idx) <= n_samples, "len(start_idx) should be <= n_samples"
    pc = np.ascontiguousarray(pc, dtype=np.float32)
    w = w or int(n_pts / n_samples * 16)
    if w >= n_pts - 1:
        warnings.warn(f"k is too large, set to {n_pts - 1}")
        w = n_pts - 1
    # Random pick a start if not given
    start_idx = get_start_idx(n_pts, start_idx)
    return _fps_npdu_sampling(pc, n_samples, w, start_idx)


def fps_npdu_kdtree_sampling(
    pc: np.ndarray,
    n_samples: int,
    w: Optional[int] = None,
    start_idx: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """
    FPS sampling with nearest-point-distance-updating (NPDU) heuristic strategy.
    Using KDTree to eliminate the need of dimensional locality.
    Slower than `fps_npdu_sampling` but more robust.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        w (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
        start_idx (int or list[int], default=None): The starting index of sampling. If set to None, it will be randomly picked.
            If multiple start indices are given, they are used as a start set for the FPS and will all be present in the final samples.
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
    if isinstance(start_idx, list):
        assert len(start_idx) <= n_samples, "len(start_idx) should be <= n_samples"
    pc = np.ascontiguousarray(pc, dtype=np.float32)
    w = w or int(n_pts / n_samples * 16)
    if w >= n_pts:
        warnings.warn(f"k is too large, set to {n_pts}")
        w = n_pts
    # Random pick a start if not given
    start_idx = get_start_idx(n_pts, start_idx)
    return _fps_npdu_kdtree_sampling(pc, n_samples, w, start_idx)


def bucket_fps_kdtree_sampling(
    pc: np.ndarray, n_samples: int, start_idx: Optional[Union[int, List[int]]] = None
) -> np.ndarray:
    """
    Bucket-based FPS sampling using KDTree. Also called "QuickFPS" in the paper.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        start_idx (int or list[int], default=None): The starting index of sampling. If set to None, it will be randomly picked.
            If multiple start indices are given, they are used as a start set for the FPS and will all be present in the final samples.
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
    if isinstance(start_idx, list):
        assert len(start_idx) <= n_samples, "len(start_idx) should be <= n_samples"
    pc = np.ascontiguousarray(pc, dtype=np.float32)
    # Random pick a start if not given
    start_idx = get_start_idx(n_pts, start_idx)
    return _bucket_fps_kdtree_sampling(pc, n_samples, start_idx)


def bucket_fps_kdline_sampling(
    pc: np.ndarray, n_samples: int, h: int, start_idx: Optional[Union[int, List[int]]] = None
) -> np.ndarray:
    """
    Bucket-based FPS sampling using KDTree, with multiple points in each bucket. Also called "QuickFPS" in the paper.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        h (int, default=None): Height of KDTree. The bucket size is `2**h`.
            According to the paper, for small workload, h=3 is enough;
            for medium workload, h=5 or 7 is enough; for large workload, h=9 is enough.
        start_idx (int or list[int], default=None): The starting index of sampling. If set to None, it will be randomly picked.
            If multiple start indices are given, they are used as a start set for the FPS and will all be present in the final samples.

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
    if isinstance(start_idx, list):
        assert len(start_idx) <= n_samples, "len(start_idx) should be <= n_samples"
    pc = np.ascontiguousarray(pc, dtype=np.float32)
    # Random pick a start if not given
    start_idx = get_start_idx(n_pts, start_idx)
    return _bucket_fps_kdline_sampling(pc, n_samples, h, start_idx)


__all__ = [
    "__doc__",
    "__version__",
    "fps_sampling",
    "fps_npdu_sampling",
    "fps_npdu_kdtree_sampling",
    "bucket_fps_kdtree_sampling",
    "bucket_fps_kdline_sampling",
]
