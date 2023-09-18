//
// Created by 韩萌 on 2022/6/14.
// Refactored by AyajiLin on 2023/9/16.
//

#ifndef KD_TREE_BASED_FARTHEST_POINT_SAMPLING_POINT_H
#define KD_TREE_BASED_FARTHEST_POINT_SAMPLING_POINT_H

#include "cmath"
#include "utils.h"
#include <algorithm>
#include <limits>

namespace quickfps {

template <typename T, size_t DIM, typename S = T> class Point {
  public:
    T pos[DIM]; // x, y, z
    S dis;
    size_t id;

    Point();
    Point(const T pos[DIM], size_t id);
    Point(const T pos[DIM], size_t id, S dis);
    Point(const Point &obj);
    ~Point(){};

    bool operator<(const Point &aii) const;

    constexpr T operator[](size_t i) const { return pos[i]; }

    Point &operator=(const Point &obj) {
        std::copy(obj.pos, obj.pos + DIM, this->pos);
        this->dis = obj.dis;
        this->id = obj.id;
        return *this;
    }

    constexpr S distance(const Point &b) { return _distance(b, DIM); }

    void reset();

    S updatedistance(const Point &ref);

    S updateDistanceAndCount(const Point &ref, size_t &count);

  private:
    constexpr S _distance(const Point &b, size_t dim_left) {
        return (dim_left == 0)
                   ? 0
                   : powi((this->pos[dim_left - 1] - b.pos[dim_left - 1]), 2) +
                         _distance(b, dim_left - 1);
    }
};

template <typename T, size_t DIM, typename S>
Point<T, DIM, S>::Point() : dis(std::numeric_limits<S>::max()), id(0) {
    std::fill(pos, pos + DIM, 0);
}

template <typename T, size_t DIM, typename S>
Point<T, DIM, S>::Point(const T pos[DIM], size_t id)
    : dis(std::numeric_limits<S>::max()), id(id) {
    std::copy(pos, pos + DIM, this->pos);
}

template <typename T, size_t DIM, typename S>
Point<T, DIM, S>::Point(const T pos[DIM], size_t id, S dis) : dis(dis), id(id) {
    std::copy(pos, pos + DIM, this->pos);
}

template <typename T, size_t DIM, typename S>
Point<T, DIM, S>::Point(const Point &obj) : dis(obj.dis), id(obj.id) {
    std::copy(obj.pos, obj.pos + DIM, this->pos);
}

template <typename T, size_t DIM, typename S>
bool Point<T, DIM, S>::operator<(const Point &aii) const {
    return dis < aii.dis;
}

template <typename T, size_t DIM, typename S>
S Point<T, DIM, S>::updatedistance(const Point &ref) {
    this->dis = std::min(this->dis, this->distance(ref));
    return this->dis;
}

template <typename T, size_t DIM, typename S>
S Point<T, DIM, S>::updateDistanceAndCount(const Point &ref, size_t &count) {
    S tempDistance = this->distance(ref);
    if (tempDistance < this->dis) {
        this->dis = tempDistance;
        count++;
    }
    return this->dis;
}

template <typename T, size_t DIM, typename S> void Point<T, DIM, S>::reset() {
    this->dis = std::numeric_limits<S>::max();
}

} // namespace quickfps

#endif // KD_TREE_BASED_FARTHEST_POINT_SAMPLING_POINT_H
