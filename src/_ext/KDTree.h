//
// Created by hanm on 22-6-15.
// Refactored by AyajiLin on 2023/9/16.
//

#ifndef FPS_CPU_KDTREE_H
#define FPS_CPU_KDTREE_H

#include "KDTreeBase.h"

namespace quickfps {

template <typename T, size_t DIM, typename S = T>
class KDTree : public KDTreeBase<T, DIM, S> {
  public:
    using typename KDTreeBase<T, DIM, S>::_Point;
    using typename KDTreeBase<T, DIM, S>::_Points;
    using typename KDTreeBase<T, DIM, S>::NodePtr;
    explicit KDTree(_Points data, size_t pointSize, _Points samplePoints);

    _Point max_point() override { return this->root_->max_point; };

    void update_distance(const _Point &ref_point) override;

    void sample(size_t sample_num) override;

    bool leftNode(size_t, size_t count) const override { return count == 1; };

    void addNode(NodePtr) override{};
};

template <typename T, size_t DIM, typename S>
KDTree<T, DIM, S>::KDTree(_Points data, size_t pointSize, _Points samplePoints)
    : KDTreeBase<T, DIM, S>(data, pointSize, samplePoints) {}

template <typename T, size_t DIM, typename S>
void KDTree<T, DIM, S>::update_distance(const _Point &ref_point) {
    this->root_->send_delay_point(ref_point);
    this->root_->update_distance();
}

template <typename T, size_t DIM, typename S>
void KDTree<T, DIM, S>::sample(size_t sample_num) {
    _Point ref_point;
    for (size_t i = 1; i < sample_num; i++) {
        ref_point = this->max_point();
        this->sample_points[i] = ref_point;
        this->update_distance(ref_point);
    }
}

} // namespace quickfps

#endif // FPS_CPU_KDTREE_H
