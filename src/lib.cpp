#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nanoflann.hpp"
#include "wrapper.hpp"

#if defined(_MSC_VER)
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class StartIndex {
public:
    enum Type { SINGLE, ARRAY } type;
    size_t single_idx;
    py::array_t<size_t> array_idx;

    StartIndex(size_t idx) : type(SINGLE), single_idx(idx) {}
    StartIndex(py::array_t<size_t> arr) : type(ARRAY), array_idx(arr) {}
};

struct PointCloud {
    size_t N, dim;
    const float* data;  // row-major: N x dim

    inline size_t kdtree_get_point_count() const { return N; }

    inline float kdtree_distance(const float* a, const size_t b_idx, size_t /*dim_*/) const {
        const float* b = data + b_idx * dim;
        float dist = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float d = a[i] - b[i];
            dist += d * d;
        }
        return dist;
    }

    inline float kdtree_get_pt(const size_t idx, int dim_) const {
        return data[idx * dim + dim_];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};

void check_py_input(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    const StartIndex& start_idx,
    std::optional<size_t> max_dim = std::nullopt
) {
    if (points.ndim() != 2) {
        throw py::value_error(
            "points must be a 2D array, but got shape " + std::to_string(points.ndim())
        );
    }

    ssize_t P = points.shape(0);
    ssize_t C = points.shape(1);

    if (C == 0) {
        throw py::value_error("points must have at least one column");
    }

    if (max_dim.has_value() && C > max_dim.value()) {
        throw py::value_error(
            "points must have at most " + std::to_string(max_dim.value()) +
            " columns, but got " + std::to_string(C)
        );
    }

    if (n_samples > static_cast<size_t>(P)) {
        throw py::value_error(
            "n_samples must be less than the number of points: n_samples=" +
            std::to_string(n_samples) + ", P=" + std::to_string(P)
        );
    }

    if (start_idx.type == StartIndex::SINGLE) {
        if (start_idx.single_idx >= static_cast<size_t>(P)) {
            throw py::value_error(
                "start_idx must be less than the number of points: start_idx=" +
                std::to_string(start_idx.single_idx) + ", P=" + std::to_string(P)
            );
        }
    } else {
        auto arr = start_idx.array_idx.unchecked<1>();
        if (arr.shape(0) > n_samples) {
            throw py::value_error(
                "The number of start indices must be less than or equal to n_samples: " +
                std::to_string(arr.shape(0)) + ", n_samples=" + std::to_string(n_samples)
            );
        }
        for (ssize_t i = 0; i < arr.shape(0); ++i) {
            if (arr(i) >= static_cast<size_t>(P)) {
                throw py::value_error(
                    "All indices in start_idx must be less than the number of points: " +
                    std::to_string(arr(i)) + ", P=" + std::to_string(P)
                );
            }
        }
    }
}

py::array_t<size_t>
fps_sampling_multi_start_index(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    py::array_t<size_t, py::array::c_style | py::array::forcecast> start_idx)
{
    auto pts = points.unchecked<2>();   // 2D view
    auto starts = start_idx.unchecked<1>();

    ssize_t P = pts.shape(0);
    ssize_t C = pts.shape(1);

    if (P <= 0 || C <= 0) {
        throw std::runtime_error("points must be a 2D array");
    }

    size_t res_selected_idx = 0;
    bool has_prev = false;

    std::vector<float> dist_min(P, std::numeric_limits<float>::infinity());
    std::vector<size_t> selected;
    selected.reserve(n_samples);

    size_t start_counter = 0;

    while (selected.size() < n_samples) {
        if (has_prev) {
            size_t i0 = res_selected_idx;

            for (ssize_t i = 0; i < P; i++) {
                float dist = 0.0f;
                for (ssize_t j = 0; j < C; j++) {
                    float d = pts(i, j) - pts(i0, j);
                    dist += d * d;
                }
                if (dist < dist_min[i]) {
                    dist_min[i] = dist;
                }
            }

            if (start_counter < (size_t) starts.shape(0)) {
                size_t idx = starts(start_counter);
                selected.push_back(idx);
                res_selected_idx = idx;
                start_counter++;
            } else {
                size_t max_idx = 0;
                float max_val = -1.0f;

                for (ssize_t i = 0; i < P; i++) {
                    if (dist_min[i] >= max_val) {
                        max_val = dist_min[i];
                        max_idx = i;
                    }
                }

                selected.push_back(max_idx);
                res_selected_idx = max_idx;
            }

        } else {
            size_t idx = starts(start_counter);
            selected.push_back(idx);
            res_selected_idx = idx;
            start_counter++;
            has_prev = true;
        }
    }

    py::array_t<size_t> out(selected.size());
    auto out_buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < selected.size(); i++) {
        out_buf(i) = selected[i];
    }
    return out;
}

py::array_t<size_t> fps_sampling(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    size_t start_idx
) {
    auto pts = points.unchecked<2>();
    ssize_t P = pts.shape(0);
    ssize_t C = pts.shape(1);

    if (P <= 0 || C <= 0) {
        throw py::value_error("points must be a 2D array with at least one column");
    }
    if (start_idx >= static_cast<size_t>(P)) {
        throw py::value_error("start_idx out of range");
    }

    size_t res_selected_idx = start_idx;
    bool has_prev = false;
    std::vector<float> dist_min(P, std::numeric_limits<float>::infinity());

    std::vector<size_t> selected;
    selected.reserve(n_samples);

    while (selected.size() < n_samples) {
        if (has_prev) {
            size_t i0 = res_selected_idx;
            for (ssize_t i = 0; i < P; ++i) {
                float dist = 0.0f;
                for (ssize_t j = 0; j < C; ++j) {
                    float d = pts(i, j) - pts(i0, j);
                    dist += d * d;
                }
                if (dist < dist_min[i]) dist_min[i] = dist;
            }

            size_t max_idx = 0;
            float max_val = -1.0f;
            for (ssize_t i = 0; i < P; ++i) {
                if (dist_min[i] >= max_val) {
                    max_val = dist_min[i];
                    max_idx = i;
                }
            }
            selected.push_back(max_idx);
            res_selected_idx = max_idx;
        } else {
            selected.push_back(start_idx);
            res_selected_idx = start_idx;
            has_prev = true;
        }
    }

    py::array_t<size_t> out(selected.size());
    auto out_buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < selected.size(); ++i) {
        out_buf(i) = selected[i];
    }
    return out;
}

// EXPORT TO _fps_sample
py::array_t<size_t> _fps_sampling(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    py::object start_idx_obj
) {
    StartIndex start_idx = [&]() -> StartIndex {
        if (py::isinstance<py::int_>(start_idx_obj)) {
            return StartIndex(start_idx_obj.cast<size_t>());
        } else if (py::isinstance<py::array_t<size_t>>(start_idx_obj)) {
            return StartIndex(start_idx_obj.cast<py::array_t<size_t>>());
        } else {
            throw py::type_error("start_idx must be int or 1D numpy array of size_t");
        }
    }();

    check_py_input(points, n_samples, start_idx, std::nullopt);

    if (start_idx.type == StartIndex::SINGLE)
        return fps_sampling(points, n_samples, start_idx.single_idx);
    else
        return fps_sampling_multi_start_index(points, n_samples, start_idx.array_idx);
}

py::array_t<size_t> fps_npdu_sampling(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    size_t k,
    size_t start_idx
) {
    auto pts = points.unchecked<2>();
    ssize_t P = pts.shape(0);
    ssize_t C = pts.shape(1);

    if (P <= 0 || C <= 0)
        throw py::value_error("points must be a 2D array with at least one column");
    if (start_idx >= static_cast<size_t>(P))
        throw py::value_error("start_idx out of range");

    std::vector<float> dist_min(P, std::numeric_limits<float>::infinity());
    std::vector<size_t> selected;
    selected.reserve(n_samples);

    size_t res_selected_idx = start_idx;
    bool has_prev = false;

    while (selected.size() < n_samples) {
        if (has_prev) {
            ssize_t hw = static_cast<ssize_t>(k / 2);
            ssize_t start = static_cast<ssize_t>(res_selected_idx) - hw;
            ssize_t end   = static_cast<ssize_t>(res_selected_idx) + hw;
            if (start < 0) { end -= start; start = 0; }
            if (end >= P) { start = std::max(start - (end - P + 1), ssize_t(0)); end = P - 1; }

            for (ssize_t i = start; i <= end; ++i) {
                float dist = 0.0f;
                for (ssize_t j = 0; j < C; ++j) {
                    float d = pts(i, j) - pts(res_selected_idx, j);
                    dist += d*d;
                }
                if (dist < dist_min[i]) dist_min[i] = dist;
            }

            size_t max_idx = 0;
            float max_val = -1.0f;
            for (ssize_t i = 0; i < P; ++i) {
                if (dist_min[i] > max_val) { max_val = dist_min[i]; max_idx = i; }
            }
            selected.push_back(max_idx);
            res_selected_idx = max_idx;

        } else {
            for (ssize_t i = 0; i < P; ++i) {
                float dist = 0.0f;
                for (ssize_t j = 0; j < C; ++j) {
                    float d = pts(i,j) - pts(start_idx,j);
                    dist += d*d;
                }
                if (dist < dist_min[i]) dist_min[i] = dist;
            }
            selected.push_back(start_idx);
            res_selected_idx = start_idx;
            has_prev = true;
        }
    }

    py::array_t<size_t> out(selected.size());
    auto out_buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < selected.size(); ++i)
        out_buf(i) = selected[i];

    return out;
}

// EXPORT TO _fps_npdu_sample
py::array_t<size_t> fps_npdu_sampling_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    size_t k,
    py::object start_idx_obj
) {
    StartIndex start_idx = [&]() -> StartIndex {
        if (py::isinstance<py::int_>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<size_t>());
        else if (py::isinstance<py::array_t<size_t>>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<py::array_t<size_t>>());
        else
            throw py::type_error("start_idx must be int or 1D numpy array of size_t");
    }();

    check_py_input(points, n_samples, start_idx);

    if (start_idx.type == StartIndex::SINGLE)
        return fps_npdu_sampling(points, n_samples, k, start_idx.single_idx);
    else {
        PyErr_SetString(PyExc_NotImplementedError, "Array of start indices not implemented yet");
        throw py::error_already_set();
    }
}

// EXPORT TO _fps_npdu_kdtree_sample
py::array_t<size_t> fps_npdu_kdtree_sampling_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    size_t k,
    py::object start_idx_obj
) {
    StartIndex start_idx = [&]() -> StartIndex {
        if (py::isinstance<py::int_>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<size_t>());
        else if (py::isinstance<py::array_t<size_t>>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<py::array_t<size_t>>());
        else
            throw py::type_error("start_idx must be int or 1D numpy array of size_t");
    }();

    check_py_input(points, n_samples, start_idx);

    if (start_idx.type == StartIndex::ARRAY) {
        PyErr_SetString(PyExc_NotImplementedError, "Array of start indices not implemented yet");
        throw py::error_already_set();
    }

    auto pts = points.unchecked<2>();
    ssize_t P = pts.shape(0);
    ssize_t C = pts.shape(1);

    if (P <= 0 || C <= 0)
        throw py::value_error("points must be a 2D array with at least one column");

    PointCloud cloud;
    cloud.N = static_cast<size_t>(P);
    cloud.dim = static_cast<size_t>(C);
    cloud.data = points.data();

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, -1>;
    KDTree index(static_cast<int>(C), cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    std::vector<float> dist_min(P, std::numeric_limits<float>::infinity());
    std::vector<size_t> selected;
    selected.reserve(n_samples);

    size_t res_selected_idx = start_idx.single_idx;
    bool has_prev = false;

    size_t k_use = std::min<size_t>(k, static_cast<size_t>(P));
    std::vector<size_t> ret_indexes(k_use);
    std::vector<float> out_dists(k_use);

    while (selected.size() < n_samples) {
        if (has_prev) {
            std::vector<float> query(static_cast<size_t>(C));
            for (ssize_t d = 0; d < C; ++d) query[d] = pts(res_selected_idx, d);

            nanoflann::KNNResultSet<float> resultSet(static_cast<int>(k_use));
            resultSet.init(ret_indexes.data(), out_dists.data());
            nanoflann::SearchParameters params;
            index.findNeighbors(resultSet, query.data(), params);

            for (size_t idx_i = 0; idx_i < k_use; ++idx_i) {
                size_t nb = ret_indexes[idx_i];
                float dist = 0.0f;
                for (ssize_t d = 0; d < C; ++d) {
                    float diff = pts(nb, d) - pts(res_selected_idx, d);
                    dist += diff * diff;
                }
                if (dist < dist_min[nb]) dist_min[nb] = dist;
            }

            size_t max_idx = 0;
            float max_val = -1.0f;
            for (ssize_t i = 0; i < P; ++i) {
                if (dist_min[i] > max_val) { max_val = dist_min[i]; max_idx = i; }
            }

            selected.push_back(max_idx);
            res_selected_idx = max_idx;
        } else {
            for (ssize_t i = 0; i < P; ++i) {
                float dist = 0.0f;
                for (ssize_t j = 0; j < C; ++j) {
                    float d = pts(i,j) - pts(res_selected_idx,j);
                    dist += d*d;
                }
                if (dist < dist_min[i]) dist_min[i] = dist;
            }
            selected.push_back(res_selected_idx);
            has_prev = true;
        }
    }

    py::array_t<size_t> out(selected.size());
    auto out_buf = out.mutable_unchecked<1>();
    for (size_t i = 0; i < selected.size(); ++i) out_buf(i) = selected[i];

    return out;
}

py::array_t<size_t> bucket_fps_kdtree_sampling_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    py::object start_idx_obj
) {
    StartIndex start_idx = [&]() -> StartIndex {
        if (py::isinstance<py::int_>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<size_t>());
        else if (py::isinstance<py::array_t<size_t>>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<py::array_t<size_t>>());
        else
            throw py::type_error("start_idx must be int or 1D numpy array of size_t");
    }();

    if (start_idx.type == StartIndex::ARRAY) {
        PyErr_SetString(PyExc_NotImplementedError, "Array of start indices not implemented yet");
        throw py::error_already_set();
    }

    check_py_input(points, n_samples, start_idx);

    if (points.ndim() != 2) {
        throw py::value_error("points must be a 2D float32 array");
    }
    ssize_t P = points.shape(0);
    ssize_t C = points.shape(1);

    if (start_idx.single_idx >= static_cast<size_t>(P)) {
        throw py::value_error("start_idx out of range");
    }
    if (n_samples == 0 || n_samples > static_cast<size_t>(P)) {
        throw py::value_error("n_samples must be in [1, num_points]");
    }

    auto buf = points.unchecked<2>();

    py::array_t<size_t> out(n_samples);
    size_t* out_ptr = static_cast<size_t*>(out.mutable_data());

    int ret = bucket_fps_kdtree(
        buf.data(0,0),                       // raw_data
        static_cast<size_t>(P),              // n_points
        static_cast<size_t>(C),              // dim
        n_samples,                           // n_samples
        start_idx.single_idx,                // start_idx
        out_ptr                              // output buffer
    );

    if (ret != 0) {
        throw std::runtime_error("bucket_fps_kdtree failed with error code " + std::to_string(ret));
    }

    return out;
}

py::array_t<size_t> bucket_fps_kdline_sampling_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    size_t n_samples,
    size_t height,
    py::object start_idx_obj
) {
    StartIndex start_idx = [&]() -> StartIndex {
        if (py::isinstance<py::int_>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<size_t>());
        else if (py::isinstance<py::array_t<size_t>>(start_idx_obj))
            return StartIndex(start_idx_obj.cast<py::array_t<size_t>>());
        else
            throw py::type_error("start_idx must be int or 1D numpy array of size_t");
    }();

    if (start_idx.type == StartIndex::ARRAY) {
        PyErr_SetString(PyExc_NotImplementedError, "Array of start indices not implemented yet");
        throw py::error_already_set();
    }

    if (points.ndim() != 2) {
        throw py::value_error("points must be a 2D float32 array");
    }

    ssize_t P = points.shape(0);
    ssize_t C = points.shape(1);

    if (start_idx.single_idx >= static_cast<size_t>(P)) {
        throw py::value_error("start_idx out of range");
    }
    if (n_samples == 0 || n_samples > static_cast<size_t>(P)) {
        throw py::value_error("n_samples must be in [1, num_points]");
    }
    if (height == 0) {
        throw py::value_error("height must be >= 1");
    }

    auto buf = points.unchecked<2>();

    py::array_t<size_t> out(n_samples);
    size_t* out_ptr = static_cast<size_t*>(out.mutable_data());

    int ret = bucket_fps_kdline(
        buf.data(0,0),                        // raw_data
        static_cast<size_t>(P),               // n_points
        static_cast<size_t>(C),               // dim
        n_samples,                            // n_samples
        start_idx.single_idx,                 // start_idx
        height,                               // window height
        out_ptr                               // output buffer
    );

    if (ret != 0) {
        throw std::runtime_error("bucket_fps_kdline failed with error code " + std::to_string(ret));
    }

    return out;
}

PYBIND11_MODULE(_fpsample, m, py::mod_gil_not_used(), py::multiple_interpreters::per_interpreter_gil()) {
    m.doc() = R"pbdoc(
        Python efficient farthest point sampling (FPS) library
        -----------------------

        .. currentmodule:: fpsample

        .. autosummary::
           :toctree: _generate

           _fps_sampling
           _fps_npdu_sampling
           _fps_npdu_kdtree_sampling
           _bucket_fps_kdtree_sampling
           _bucket_fps_kdline_sampling
    )pbdoc";

    m.def("_fps_sampling", &_fps_sampling, R"pbdoc(
            Farthest Point Sampling (FPS)
            Args:
                points (np.ndarray[float32, 2D]): N x C point array.
                n_samples (int): number of samples to pick.
                start_idx (int or np.ndarray[int32, 1D]): initial index or indices to start FPS.
            Returns:
                np.ndarray[int32]: sampled point indices.
    )pbdoc");

    m.def("_fps_npdu_sampling", &fps_npdu_sampling_py, R"pbdoc(
            FPS with Nearest Point Distance Update
            Args:
                points (np.ndarray[float32, 2D]): N x C point array.
                n_samples (int): number of samples to pick.
                k (int): number of neighbors for local update.
                start_idx (int): initial index to start FPS.
            Returns:
                np.ndarray[int32]: sampled point indices.
    )pbdoc");

    m.def("_fps_npdu_kdtree_sampling", &fps_npdu_kdtree_sampling_py, R"pbdoc(
            FPS with Nearest Point Distance Update using KD-tree acceleration
            Args:
                points (np.ndarray[float32, 2D]): N x C point array.
                n_samples (int): number of samples to pick.
                k (int): number of neighbors for local update.
                start_idx (int): initial index to start FPS.
            Returns:
                np.ndarray[int32]: sampled point indices.
    )pbdoc");

    m.def("_bucket_fps_kdtree_sampling",
      &bucket_fps_kdtree_sampling_py,
      R"pbdoc(
          Bucket FPS sampling using KD-tree acceleration.
          Args:
              points (np.ndarray[float32, 2D]): N x C point array.
              n_samples (int): number of samples to pick.
              start_idx (int): initial index to start FPS.
          Returns:
              np.ndarray[int32]: sampled point indices.
      )pbdoc");

m.def("_bucket_fps_kdline_sampling",
      &bucket_fps_kdline_sampling_py,
      R"pbdoc(
          Bucket FPS sampling using KD-line local window update.
          Args:
              points (np.ndarray[float32, 2D]): N x C point array.
              n_samples (int): number of samples to pick.
              height (int): window size around selected point.
              start_idx (int): first index.
          Returns:
              np.ndarray[int32]: sampled point indices.
      )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
