//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2023/9/16.
//

#ifndef KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDNODE_H
#define KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDNODE_H
#include <limits>
#include <queue>
#include <vector>

#include "Interval.h"
#include "Point.h"

namespace quickfps {

template <typename T, size_t DIM, typename S> class KDNode {
  public:
    using _Point = Point<T, DIM, S>;
    using _Points = _Point *;
    _Points points;
    size_t pointLeft, pointRight;
    size_t idx;

    Interval<T> bbox[DIM];
    std::vector<_Point> waitpoints;
    std::vector<_Point> delaypoints;
    _Point max_point;
    KDNode *left;
    KDNode *right;

    KDNode();

    KDNode(const KDNode &a);

    KDNode(const Interval<T> (&bbox_ptr)[DIM]);

    void init(const _Point &ref);

    void updateMaxPoint(const _Point &lpoint, const _Point &rpoint);

    S bound_distance(const _Point &ref_point);

    void send_delay_point(const _Point &point);

    void update_distance();

    void reset();

    size_t size();
};

template <typename T, size_t DIM, typename S>
KDNode<T, DIM, S>::KDNode()
    : points(nullptr), pointLeft(0), pointRight(0), left(nullptr),
      right(nullptr) {}

template <typename T, size_t DIM, typename S>
KDNode<T, DIM, S>::KDNode(const Interval<T> (&bbox_ptr)[DIM])
    : points(nullptr), pointLeft(0), pointRight(0), left(nullptr),
      right(nullptr) {
    std::copy(bbox_ptr, bbox_ptr + DIM, bbox);
}

template <typename T, size_t DIM, typename S>
KDNode<T, DIM, S>::KDNode(const KDNode &a)
    : points(a.points), pointLeft(a.pointLeft), pointRight(a.pointRight),
      left(a.left), right(a.right), idx(a.idx) {
    std::copy(a.bbox, a.bbox + DIM, bbox);
    std::copy(a.waitpoints.begin(), a.waitpoints.end(), waitpoints.begin());
    std::copy(a.delaypoints.begin(), a.delaypoints.end(), delaypoints.begin());
}

template <typename T, size_t DIM, typename S>
void KDNode<T, DIM, S>::init(const _Point &ref) {
    waitpoints.clear();
    delaypoints.clear();
    if (this->left && this->right) {
        this->left->init(ref);
        this->right->init(ref);
        updateMaxPoint(this->left->max_point, this->right->max_point);
    } else {
        S dis;
        S maxdis = std::numeric_limits<S>::lowest();
        for (size_t i = pointLeft; i < pointRight; i++) {
            dis = points[i].updatedistance(ref);
            if (dis > maxdis) {
                maxdis = dis;
                max_point = points[i];
            }
        }
    }
}

template <typename T, size_t DIM, typename S>
void KDNode<T, DIM, S>::updateMaxPoint(const _Point &lpoint,
                                       const _Point &rpoint) {
    if (lpoint.dis > rpoint.dis)
        this->max_point = lpoint;
    else
        this->max_point = rpoint;
}

template <typename T, size_t DIM, typename S>
S KDNode<T, DIM, S>::bound_distance(const _Point &ref_point) {
    S bound_dis(0);
    S dim_distance;
    for (size_t cur_dim = 0; cur_dim < DIM; cur_dim++) {
        dim_distance = 0;
        if (ref_point[cur_dim] > bbox[cur_dim].high)
            dim_distance = ref_point[cur_dim] - bbox[cur_dim].high;
        if (ref_point[cur_dim] < bbox[cur_dim].low)
            dim_distance = bbox[cur_dim].low - ref_point[cur_dim];
        bound_dis += powi(dim_distance, 2);
    }
    return bound_dis;
}

template <typename T, size_t DIM, typename S>
void KDNode<T, DIM, S>::send_delay_point(const _Point &point) {
    this->waitpoints.push_back(point);
}

template <typename T, size_t DIM, typename S>
void KDNode<T, DIM, S>::update_distance() {
    for (const auto &ref_point : this->waitpoints) {
        S lastmax_distance = this->max_point.dis;
        S cur_distance = this->max_point.distance(ref_point);
        // cur_distance >
        // lastmax_distance意味着当前Node的max_point不会进行更新
        if (cur_distance > lastmax_distance) {
            S boundary_distance = bound_distance(ref_point);
            if (boundary_distance < lastmax_distance)
                this->delaypoints.push_back(ref_point);
        } else {
            if (this->right && this->left) {
                if (!delaypoints.empty()) {
                    for (const auto &delay_point : delaypoints) {
                        this->left->send_delay_point(delay_point);
                        this->right->send_delay_point(delay_point);
                    }
                    delaypoints.clear();
                }
                this->left->send_delay_point(ref_point);
                this->left->update_distance();

                this->right->send_delay_point(ref_point);
                this->right->update_distance();

                updateMaxPoint(this->left->max_point, this->right->max_point);
            } else {
                S dis;
                S maxdis;
                this->delaypoints.push_back(ref_point);
                for (const auto &delay_point : delaypoints) {
                    maxdis = std::numeric_limits<S>::lowest();
                    for (size_t i = pointLeft; i < pointRight; i++) {
                        dis = points[i].updatedistance(delay_point);
                        if (dis > maxdis) {
                            maxdis = dis;
                            max_point = points[i];
                        }
                    }
                }
                this->delaypoints.clear();
            }
        }
    }
    this->waitpoints.clear();
}

template <typename T, size_t DIM, typename S> void KDNode<T, DIM, S>::reset() {
    for (size_t i = pointLeft; i < pointRight; i++) {
        points[i].reset();
    }
    this->waitpoints.clear();
    this->delaypoints.clear();
    this->max_point.reset();
    if (this->left && this->right) {
        this->left->reset();
        this->right->reset();
    }
}

template <typename T, size_t DIM, typename S> size_t KDNode<T, DIM, S>::size() {
    if (this->left && this->right)
        return this->left->size() + this->right->size();
    return (pointRight - pointLeft);
}

} // namespace quickfps

#endif // KD_TREE_BASED_FARTHEST_POINT_SAMPLING_KDNODE_H
