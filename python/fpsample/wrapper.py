from typing import Optional

import numpy as np

from .fpsample import _fps_npdu_sampling, _fps_sampling


def fps_sampling(pcd_xyz: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Args:
        pcd_xyz (np.ndarray): The input point cloud of shape (n_pts, 3).
        n_samples (int): Number of samples.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pcd_xyz.ndim == 2
    n_pts, _ = pcd_xyz.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    pcd_xyz = pcd_xyz.astype(np.float32)
    # best performance with fortran array
    pcd_xyz = np.asfortranarray(pcd_xyz)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts)
    return _fps_sampling(pcd_xyz, n_samples, start_idx)


def fps_npdu_sampling(pcd_xyz: np.ndarray, n_samples: int, k: Optional[int] = None) -> np.ndarray:
    """
    Args:
        pcd_xyz (np.ndarray): The input point cloud of shape (n_pts, 3).
        n_samples (int): Number of samples.
        k (int, default=None): Windows size of local heuristic search. If set to None, it will be set to `n_pts / n_samples * 16`.
    Returns:
        np.ndarray: The selected indices of shape (n_samples,).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pcd_xyz.ndim == 2
    n_pts, _ = pcd_xyz.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    pcd_xyz = pcd_xyz.astype(np.float32)
    k = k or int(n_pts / n_samples * 16)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts)
    return _fps_npdu_sampling(pcd_xyz, n_samples, k, start_idx)
