#include "_ext/KDLineTree.h"
#include "_ext/KDTree.h"
#include <array>
#include <memory>
#include <utility>
#include <vector>

#ifndef BUCKET_FPS_MAX_DIM
#define BUCKET_FPS_MAX_DIM 8
#endif
constexpr size_t max_dim = BUCKET_FPS_MAX_DIM;

using quickfps::KDLineTree;
using quickfps::KDTree;
using quickfps::Point;

template <typename T, size_t DIM, typename S>
std::vector<Point<T, DIM, S>> raw_data_to_points(const float *raw_data,
                                                 size_t n_points, size_t dim) {
    std::vector<Point<T, DIM, S>> points;
    points.reserve(n_points);
    for (size_t i = 0; i < n_points; i++) {
        const float *ptr = raw_data + i * dim;
        points.push_back(Point<T, DIM, S>(ptr, i));
    }
    return points;
}

template <typename T, size_t DIM, typename S = T>
void kdtree_sample(const float *raw_data, size_t n_points, size_t dim,
                   size_t n_samples, size_t start_idx,
                   size_t *sampled_point_indices) {
    auto points = raw_data_to_points<T, DIM, S>(raw_data, n_points, dim);
    std::unique_ptr<Point<T, DIM, S>[]> sampled_points(
        new Point<T, DIM, S>[n_samples]);
    KDTree<T, DIM, S> tree(points.data(), n_points, sampled_points.get());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

template <typename T, size_t DIM, typename S = T>
void kdline_sample(const float *raw_data, size_t n_points, size_t dim,
                   size_t n_samples, size_t start_idx, size_t height,
                   size_t *sampled_point_indices) {
    auto points = raw_data_to_points<T, DIM, S>(raw_data, n_points, dim);
    std::unique_ptr<Point<T, DIM, S>[]> sampled_points(
        new Point<T, DIM, S>[n_samples]);
    KDLineTree<T, DIM, S> tree(points.data(), n_points, height,
                               sampled_points.get());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

////////////////////////////////////////
//                                    //
//    Compile Time Function Helper    //
//                                    //
////////////////////////////////////////
using KDTreeFuncType = void (*)(const float *, size_t, size_t, size_t, size_t,
                                size_t *);
using KDLineFuncType = void (*)(const float *, size_t, size_t, size_t, size_t,
                                size_t, size_t *);

template <typename T, size_t Count, typename M, size_t... I>
constexpr std::array<T, Count> mapIndices(M &&m, std::index_sequence<I...>) {
    std::array<T, Count> result{m.template operator()<I + 1>()...};
    return result;
}

template <typename T, size_t Count, typename M>
constexpr std::array<T, Count> map(M m) {
    return mapIndices<T, Count>(m, std::make_index_sequence<Count>());
}

template <typename T, typename S = T> struct kdtree_func_helper {
    template <size_t DIM> KDTreeFuncType operator()() {
        return &kdtree_sample<T, DIM, S>;
    }
};

template <typename T, typename S = T> struct kdline_func_helper {
    template <size_t DIM> KDLineFuncType operator()() {
        return &kdline_sample<T, DIM, S>;
    }
};

/////////////////
//             //
//    C API    //
//             //
/////////////////

extern "C" {
int bucket_fps_kdtree(const float *raw_data, size_t n_points, size_t dim,
                      size_t n_samples, size_t start_idx,
                      size_t *sampled_point_indices) {
    if (dim == 0 || dim > max_dim) {
        // only support 1 to MAX_DIM dimensions
        return 1;
    } else if (start_idx >= n_points) {
        // start_idx should be smaller than n_samples
        return 2;
    }
    auto func_arr = map<KDTreeFuncType, max_dim>(kdtree_func_helper<float>{});
    func_arr[dim - 1](raw_data, n_points, dim, n_samples, start_idx,
                      sampled_point_indices);
    return 0;
}

int bucket_fps_kdline(const float *raw_data, size_t n_points, size_t dim,
                      size_t n_samples, size_t start_idx, size_t height,
                      size_t *sampled_point_indices) {
    if (dim == 0 || dim > max_dim) {
        // only support 1 to MAX_DIM dimensions
        return 1;
    } else if (start_idx >= n_points) {
        // start_idx should be smaller than n_samples
        return 2;
    }
    auto func_arr = map<KDLineFuncType, max_dim>(kdline_func_helper<float>{});
    func_arr[dim - 1](raw_data, n_points, dim, n_samples, start_idx, height,
                      sampled_point_indices);
    return 0;
}
}
