mod ffi;
use numpy::ndarray::{Array1, ArrayView2};

pub fn bucket_fps_kdtree_sampling(
    points: ArrayView2<f32>,
    n_samples: usize,
    start_idx: usize,
) -> Array1<usize> {
    let[p, c] = points.shape() else {panic !("points must be a 2D array")};
    let raw_data = points.as_standard_layout().as_ptr();
    let mut sampled_point_indices = vec![0; n_samples];
    let ret_code;
    unsafe {
        ret_code = ffi::bucket_fps_kdtree(
            raw_data,
            *p,
            *c,
            n_samples,
            start_idx,
            sampled_point_indices.as_mut_ptr(),
        )
    }
    if ret_code != 0 {
        panic!("bucket_fps_kdtree failed with error code {}", ret_code);
    }
    sampled_point_indices.into()
}

pub fn bucket_fps_kdline_sampling(
    points: ArrayView2<f32>,
    n_samples: usize,
    height: usize,
    start_idx: usize,
) -> Array1<usize> {
    let[p, c] = points.shape() else {panic !("points must be a 2D array")};
    let raw_data = points.as_standard_layout().as_ptr();
    let mut sampled_point_indices = vec![0; n_samples];
    let ret_code;
    unsafe {
        ret_code = ffi::bucket_fps_kdline(
            raw_data,
            *p,
            *c,
            n_samples,
            start_idx,
            height,
            sampled_point_indices.as_mut_ptr(),
        )
    }
    if ret_code != 0 {
        panic!("bucket_fps_kdline failed with error code {}", ret_code);
    }
    sampled_point_indices.into()
}
