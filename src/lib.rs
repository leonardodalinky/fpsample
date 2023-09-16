use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use numpy::{
    ndarray::s,
    ndarray::{Array1, ArrayView2, Axis, Zip},
    PyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

mod bucket_fps;
pub mod build_info {
    include!("../build_info.rs");
}

fn check_py_input(
    points: &PyReadonlyArray2<f32>,
    n_samples: usize,
    start_idx: usize,
    max_dim: Option<usize>,
) -> PyResult<()> {
    let [p, c] = points.shape() else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "points must be a 2D array, but got shape {:?}",
            points.shape()
        )));
    };
    if *c == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "points must have at least one column",
        ));
    }
    if let Some(max_dim) = max_dim {
        if *c > max_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "points must have at most {} columns, but got {}",
                max_dim, c
            )));
        }
    }
    if n_samples > *p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "n_samples must be less than the number of points: n_samples={}, P={}",
            n_samples, p
        )));
    }
    if start_idx >= *p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "start_idx must be less than the number of points: start_idx={}, P={}",
            start_idx, p
        )));
    }
    Ok(())
}

fn fps_sampling(points: ArrayView2<f32>, n_samples: usize, start_idx: usize) -> Array1<usize> {
    let [p, _c] = points.shape() else {
        panic!("points must be a 2D array")
    };
    // previous round selected point index
    let mut res_selected_idx: Option<usize> = None;
    // distance from each point to the selected point set
    let mut dist_pts_to_selected_min = Array1::<f32>::from_elem((*p,), f32::INFINITY);
    // selected points index
    let mut selected_pts_idx = Vec::<usize>::with_capacity(n_samples);

    while selected_pts_idx.len() < n_samples {
        if let Some(prev_idx) = res_selected_idx {
            // update distance
            let dist = &points - &points.slice(s![prev_idx, ..]);
            let dist = dist.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            // update min distance
            Zip::from(&mut dist_pts_to_selected_min)
                .and(&dist)
                .for_each(|x, &y| {
                    if *x > y {
                        *x = y;
                    }
                });
            // select the point with max distance
            let max_idx = dist_pts_to_selected_min
                .indexed_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            selected_pts_idx.push(max_idx);
            res_selected_idx = Some(max_idx);
        } else {
            // first point
            selected_pts_idx.push(start_idx);
            res_selected_idx = Some(start_idx);
        }
    }
    selected_pts_idx.into()
}

#[pyfunction]
#[pyo3(name = "_fps_sampling")]
fn fps_sampling_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    start_idx: usize,
) -> PyResult<&'py PyArray1<usize>> {
    check_py_input(&points, n_samples, start_idx, None)?;
    let points = points.as_array();
    let idxs = py.allow_threads(|| fps_sampling(points, n_samples, start_idx));
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

fn fps_npdu_sampling(
    points: ArrayView2<f32>,
    n_samples: usize,
    k: usize,
    start_idx: usize,
) -> Array1<usize> {
    let [p, _c] = points.shape() else {
        panic!("points must be a 2D array")
    };
    // previous round selected point index
    let mut res_selected_idx: Option<usize> = None;
    // distance from each point to the selected point set
    let mut dist_pts_to_selected_min = Array1::<f32>::from_elem((*p,), f32::INFINITY);
    // selected points index
    let mut selected_pts_idx = Vec::<usize>::with_capacity(n_samples);

    while selected_pts_idx.len() < n_samples {
        if let Some(prev_idx) = res_selected_idx {
            // find window
            let window_range = {
                let p = *p as isize;
                let hw = (k / 2) as isize;
                let mut start = prev_idx as isize - hw;
                let mut end = prev_idx as isize + hw;
                if start < 0 {
                    end -= start;
                    start = 0;
                }
                if end >= p {
                    start = std::cmp::max(start - (end - p + 1), 0);
                    end = p - 1;
                }
                start..=end
            };

            // update distance
            let dist =
                &points.slice(s![window_range.clone(), ..]) - &points.slice(s![prev_idx, ..]);
            let dist = dist.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            // update min distance
            Zip::from(dist_pts_to_selected_min.slice_mut(s![window_range]))
                .and(&dist)
                .for_each(|x, &y| {
                    *x = f32::min(*x, y);
                });
            // select the point with max distance
            let max_idx = dist_pts_to_selected_min
                .indexed_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            selected_pts_idx.push(max_idx);
            res_selected_idx = Some(max_idx);
        } else {
            // update distance at the first round
            let dist = &points - &points.slice(s![start_idx, ..]);
            let dist = dist.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            // update min distance
            Zip::from(&mut dist_pts_to_selected_min)
                .and(&dist)
                .for_each(|x, &y| {
                    *x = f32::min(*x, y);
                });
            // first point
            selected_pts_idx.push(start_idx);
            res_selected_idx = Some(start_idx);
        }
    }
    selected_pts_idx.into()
}

#[pyfunction]
#[pyo3(name = "_fps_npdu_sampling")]
fn fps_npdu_sampling_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    k: usize,
    start_idx: usize,
) -> PyResult<&'py PyArray1<usize>> {
    check_py_input(&points, n_samples, start_idx, None)?;
    let points = points.as_array();
    let idxs = py.allow_threads(|| fps_npdu_sampling(points, n_samples, k, start_idx));
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

fn fps_npdu_kdtree_sampling(
    points: ArrayView2<f32>,
    n_samples: usize,
    k: usize,
    start_idx: usize,
) -> Array1<usize> {
    let [p, c] = points.shape() else {
        panic!("points must be a 2D array")
    };
    // contruct kd-tree
    let std_points = points.as_standard_layout();
    let mut kdtree: KdTree<f32, usize, _> = KdTree::new(*c);
    let std_points_vec = std_points.outer_iter().collect::<Vec<_>>();
    let std_points_slice_vec = std_points_vec
        .iter()
        .map(|x| x.as_slice().unwrap())
        .collect::<Vec<_>>();
    for (i, point) in std_points_slice_vec.iter().enumerate() {
        kdtree.add(*point, i).unwrap();
    }

    // previous round selected point index
    let mut res_selected_idx: Option<usize> = None;
    // distance from each point to the selected point set
    let mut dist_pts_to_selected_min = Array1::<f32>::from_elem((*p,), f32::INFINITY);
    // selected points index
    let mut selected_pts_idx = Vec::<usize>::with_capacity(n_samples);

    while selected_pts_idx.len() < n_samples {
        if let Some(prev_idx) = res_selected_idx {
            // find nearest point
            let nearest_idx = kdtree
                .nearest(std_points_slice_vec[prev_idx], k, &squared_euclidean)
                .unwrap();
            for (dist, idx) in nearest_idx {
                // dist_pts_to_selected_min[idx] = f32::min(dist_pts_to_selected_min[idx], dist);
                let d = dist_pts_to_selected_min.get_mut(*idx).unwrap();
                *d = f32::min(*d, dist);
            }

            let max_idx = dist_pts_to_selected_min
                .indexed_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            selected_pts_idx.push(max_idx);
            res_selected_idx = Some(max_idx);
        } else {
            // update distance at the first round
            let dist = &points - &points.slice(s![start_idx, ..]);
            let dist = dist.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            // update min distance
            Zip::from(&mut dist_pts_to_selected_min)
                .and(&dist)
                .for_each(|x, &y| {
                    *x = f32::min(*x, y);
                });
            // first point
            selected_pts_idx.push(start_idx);
            res_selected_idx = Some(start_idx);
        }
    }
    selected_pts_idx.into()
}

#[pyfunction]
#[pyo3(name = "_fps_npdu_kdtree_sampling")]
fn fps_npdu_kdtree_sampling_py<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    k: usize,
    start_idx: usize,
) -> PyResult<&'py PyArray1<usize>> {
    check_py_input(&points, n_samples, start_idx, None)?;
    let points = points.as_array();
    let idxs = py.allow_threads(|| fps_npdu_kdtree_sampling(points, n_samples, k, start_idx));
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

//////////////////////
//                  //
//    bucket fps    //
//                  //
//////////////////////
#[pyfunction]
#[pyo3(name = "_bucket_fps_kdtree_sampling")]
fn bucket_fps_kdtree_sampling<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    start_idx: usize,
) -> PyResult<&'py PyArray1<usize>> {
    check_py_input(
        &points,
        n_samples,
        start_idx,
        Some(build_info::BUCKET_FPS_MAX_DIM),
    )?;
    let points = points.as_array();
    let idxs =
        py.allow_threads(|| bucket_fps::bucket_fps_kdtree_sampling(points, n_samples, start_idx));
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

#[pyfunction]
#[pyo3(name = "_bucket_fps_kdline_sampling")]
fn bucket_fps_kdline_sampling<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f32>,
    n_samples: usize,
    height: usize,
    start_idx: usize,
) -> PyResult<&'py PyArray1<usize>> {
    check_py_input(
        &points,
        n_samples,
        start_idx,
        Some(build_info::BUCKET_FPS_MAX_DIM),
    )?;
    let points = points.as_array();
    let idxs = py.allow_threads(|| {
        bucket_fps::bucket_fps_kdline_sampling(points, n_samples, height, start_idx)
    });
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

////////////////////////
//                    //
//    Python Entry    //
//                    //
////////////////////////
#[pymodule]
fn fpsample(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fps_sampling_py, m)?)?;
    m.add_function(wrap_pyfunction!(fps_npdu_sampling_py, m)?)?;
    m.add_function(wrap_pyfunction!(fps_npdu_kdtree_sampling_py, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_fps_kdtree_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_fps_kdline_sampling, m)?)?;

    Ok(())
}
