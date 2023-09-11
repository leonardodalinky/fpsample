from typing import Optional

import numpy as np

from .fpsample import _fps_npdu_sampling, _fps_sampling


def fps_sampling(pc: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    pc = pc.astype(np.float32)
    # best performance with fortran array
    pc = np.asfortranarray(pc)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts)
    return _fps_sampling(pc, n_samples, start_idx)


def fps_npdu_sampling(pc: np.ndarray, n_samples: int, k: Optional[int] = None) -> np.ndarray:
    """
    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        k (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2
    n_pts, _ = pc.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    pc = pc.astype(np.float32)
    k = k or int(n_pts / n_samples * 16)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts)
    return _fps_npdu_sampling(pc, n_samples, k, start_idx)
