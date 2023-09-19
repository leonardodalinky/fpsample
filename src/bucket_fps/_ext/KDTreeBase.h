//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2023/9/16.
//

#ifndef KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDTREE_H
#define KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDTREE_H

#include "KDNode.h"
#include "Point.h"
#include <algorithm>
#include <array>
#include <numeric>

namespace quickfps {

template <typename T, size_t DIM, typename S> class KDTreeBase {
  public:
    using _Point = Point<T, DIM, S>;
    using _Points = _Point *;
    using NodePtr = KDNode<T, DIM, S> *;
    using _Interval = Interval<T>;

    size_t pointSize;
    _Points sample_points;
    NodePtr root_;
    _Points points_;

  public:
    KDTreeBase(_Points data, size_t pointSize, _Points samplePoints);

    ~KDTreeBase();

    void buildKDtree();

    NodePtr get_root() const { return this->root_; };

    void init(const _Point &ref);

    virtual _Point max_point() = 0;

    virtual void sample(size_t sample_num) = 0;

  protected:
    void deleteNode(NodePtr node_p);
    virtual void addNode(NodePtr p) = 0;
    virtual bool leftNode(size_t high, size_t count) const = 0;
    virtual void update_distance(const _Point &ref_point) = 0;

    NodePtr divideTree(ssize_t left, ssize_t right,
                       const std::array<_Interval, DIM> &bboxs,
                       size_t curr_high);

    size_t planeSplit(ssize_t left, ssize_t right, size_t split_dim,
                      T split_val);

    T qSelectMedian(size_t dim, size_t left, size_t right);
    static size_t findSplitDim(const std::array<_Interval, DIM> &bboxs);
    inline std::array<_Interval, DIM> computeBoundingBox(size_t left,
                                                         size_t right);
};

template <typename T, size_t DIM, typename S>
KDTreeBase<T, DIM, S>::KDTreeBase(_Points data, size_t pointSize,
                                  _Points samplePoints)
    : pointSize(pointSize), sample_points(samplePoints), root_(nullptr),
      points_(data) {}

template <typename T, size_t DIM, typename S>
KDTreeBase<T, DIM, S>::~KDTreeBase() {
    if (root_ != nullptr)
        deleteNode(root_);
}

template <typename T, size_t DIM, typename S>
void KDTreeBase<T, DIM, S>::deleteNode(NodePtr node_p) {
    if (node_p->left)
        deleteNode(node_p->left);
    if (node_p->right)
        deleteNode(node_p->right);
    delete node_p;
}

template <typename T, size_t DIM, typename S>
void KDTreeBase<T, DIM, S>::buildKDtree() {
    size_t left = 0;
    size_t right = pointSize;
    std::array<_Interval, DIM> bboxs = this->computeBoundingBox(left, right);
    this->root_ = divideTree(left, right, bboxs, 0);
}

template <typename T, size_t DIM, typename S>
typename KDTreeBase<T, DIM, S>::NodePtr
KDTreeBase<T, DIM, S>::divideTree(ssize_t left, ssize_t right,
                                  const std::array<_Interval, DIM> &bboxs,
                                  size_t curr_high) {
    NodePtr node = new KDNode<T, DIM, S>(bboxs);

    ssize_t count = right - left;
    if (this->leftNode(curr_high, count)) {
        node->pointLeft = left;
        node->pointRight = right;
        node->points = this->points_;
        this->addNode(node);
        return node;
    } else {
        size_t split_dim = this->findSplitDim(bboxs);
        T split_val = this->qSelectMedian(split_dim, left, right);

        size_t split_delta = planeSplit(left, right, split_dim, split_val);

        std::array<_Interval, DIM> bbox_cur =
            this->computeBoundingBox(left, left + split_delta);
        node->left =
            this->divideTree(left, left + split_delta, bbox_cur, curr_high + 1);
        bbox_cur = this->computeBoundingBox(left + split_delta, right);
        node->right = this->divideTree(left + split_delta, right, bbox_cur,
                                       curr_high + 1);
        return node;
    }
}

template <typename T, size_t DIM, typename S>
size_t KDTreeBase<T, DIM, S>::planeSplit(ssize_t left, ssize_t right,
                                         size_t split_dim, T split_val) {
    ssize_t start = left;
    ssize_t end = right - 1;

    for (;;) {
        while (start <= end && points_[start].pos[split_dim] < split_val)
            ++start;
        while (start <= end && points_[end].pos[split_dim] >= split_val)
            --end;

        if (start > end)
            break;
        std::swap(points_[start], points_[end]);
        ++start;
        --end;
    }

    ssize_t lim1 = start - left;
    if (start == left)
        lim1 = 1;
    if (start == right)
        lim1 = (right - left - 1);

    return static_cast<ssize_t>(lim1);
}

template <typename T, size_t DIM, typename S>
T KDTreeBase<T, DIM, S>::qSelectMedian(size_t dim, size_t left, size_t right) {
    T sum = std::accumulate(this->points_ + left, this->points_ + right, 0.0,
                            [dim](const T &acc, const _Point &point) {
                                return acc + point.pos[dim];
                            });
    return sum / (right - left);
}

template <typename T, size_t DIM, typename S>
size_t
KDTreeBase<T, DIM, S>::findSplitDim(const std::array<_Interval, DIM> &bboxs) {
    T min_, max_;
    T span = 0;
    size_t best_dim = 0;

    for (size_t cur_dim = 0; cur_dim < DIM; cur_dim++) {
        min_ = bboxs[cur_dim].low;
        max_ = bboxs[cur_dim].high;
        T cur_span = (max_ - min_);

        if (cur_span > span) {
            best_dim = cur_dim;
            span = cur_span;
        }
    }

    return best_dim;
}

template <typename T, size_t DIM, typename S>
inline std::array<Interval<T>, DIM>
KDTreeBase<T, DIM, S>::computeBoundingBox(size_t left, size_t right) {
    T min_vals[DIM];
    T max_vals[DIM];
    std::fill(min_vals, min_vals + DIM, std::numeric_limits<T>::max());
    std::fill(max_vals, max_vals + DIM, std::numeric_limits<T>::lowest());

    for (size_t i = left; i < right; ++i) {
        const _Point &pos = points_[i];

        for (size_t cur_dim = 0; cur_dim < DIM; cur_dim++) {
            T val = pos[cur_dim];
            min_vals[cur_dim] = std::min(min_vals[cur_dim], val);
            max_vals[cur_dim] = std::max(max_vals[cur_dim], val);
        }
    }

    std::array<_Interval, DIM> bboxs;

    for (size_t cur_dim = 0; cur_dim < DIM; cur_dim++) {
        bboxs[cur_dim].low = min_vals[cur_dim];
        bboxs[cur_dim].high = max_vals[cur_dim];
    }

    return bboxs;
}

template <typename T, size_t DIM, typename S>
void KDTreeBase<T, DIM, S>::init(const _Point &ref) {
    this->sample_points[0] = ref;
    this->root_->init(ref);
}

} // namespace quickfps

#endif // KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDTREE_H
