use numpy::{
    ndarray::s,
    ndarray::{Array1, ArrayView2, Axis, Zip},
    PyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

fn check_py_input(
    points: &PyReadonlyArray2<f32>,
    n_samples: usize,
    start_idx: usize,
) -> PyResult<()> {
    let [p, _c] = points.shape() else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "points must be a 2D array, but got shape {:?}",
            points.shape()
        )));
    };
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
    check_py_input(&points, n_samples, start_idx)?;
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
    check_py_input(&points, n_samples, start_idx)?;
    let points = points.as_array();
    let idxs = py.allow_threads(|| fps_npdu_sampling(points, n_samples, k, start_idx));
    let ret = idxs.to_pyarray(py);
    Ok(ret)
}

#[pymodule]
fn fpsample(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fps_sampling_py, m)?)?;
    m.add_function(wrap_pyfunction!(fps_npdu_sampling_py, m)?)?;

    Ok(())
}
