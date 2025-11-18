//
// Created by hanm on 22-6-15.
// Refactored by AyajiLin on 2023/9/16.
//

#ifndef FPS_CPU_KDLINETREE_H
#define FPS_CPU_KDLINETREE_H

#include <limits>
#include <vector>

#include "KDTreeBase.h"

namespace quickfps {

template <typename T, size_t DIM, typename S = T>
class KDLineTree : public KDTreeBase<T, DIM, S> {
  public:
    using typename KDTreeBase<T, DIM, S>::_Point;
    using typename KDTreeBase<T, DIM, S>::_Points;
    using typename KDTreeBase<T, DIM, S>::NodePtr;

    KDLineTree(_Points data, size_t pointSize, size_t treeHigh,
               _Points samplePoints);
    ~KDLineTree();

    std::vector<NodePtr> KDNode_list;

    size_t high_;

    _Point max_point() override;

    void update_distance(const _Point &ref_point) override;

    void sample(size_t sample_num) override;

    bool leftNode(size_t high, size_t count) const override {
        return high == this->high_ || count == 1;
    };

    void addNode(NodePtr p) override;
};

template <typename T, size_t DIM, typename S>
KDLineTree<T, DIM, S>::KDLineTree(_Points data, size_t pointSize,
                                  size_t treeHigh, _Points samplePoints)
    : KDTreeBase<T, DIM, S>(data, pointSize, samplePoints), high_(treeHigh) {
    KDNode_list.clear();
}

template <typename T, size_t DIM, typename S>
KDLineTree<T, DIM, S>::~KDLineTree() {
    KDNode_list.clear();
}

template <typename T, size_t DIM, typename S>
typename KDLineTree<T, DIM, S>::_Point KDLineTree<T, DIM, S>::max_point() {
    _Point tmpPoint;
    S max_distance = std::numeric_limits<S>::lowest();
    for (const auto &bucket : KDNode_list) {
        if (bucket->max_point.dis > max_distance) {
            max_distance = bucket->max_point.dis;
            tmpPoint = bucket->max_point;
        }
    }
    return tmpPoint;
}

template <typename T, size_t DIM, typename S>
void KDLineTree<T, DIM, S>::update_distance(const _Point &ref_point) {
    for (const auto &bucket : KDNode_list) {
        bucket->send_delay_point(ref_point);
        bucket->update_distance();
    }
}

template <typename T, size_t DIM, typename S>
void KDLineTree<T, DIM, S>::sample(size_t sample_num) {
    _Point ref_point;
    for (size_t i = 1; i < sample_num; i++) {
        ref_point = this->max_point();
        this->sample_points[i] = ref_point;
        this->update_distance(ref_point);
    }
}

template <typename T, size_t DIM, typename S>
void KDLineTree<T, DIM, S>::addNode(NodePtr p) {
    size_t nodeIdx = KDNode_list.size();
    p->idx = nodeIdx;
    KDNode_list.push_back(p);
}

} // namespace quickfps

#endif // FPS_CPU_KDLINETREE_H
