// Refactored by AyajiLin on 2023/9/16.

/* functional  */
#ifndef KD_TREE_UTILS_HPP
#define KD_TREE_UTILS_HPP
#include <cstddef>

template <typename T>
inline constexpr T powi(const T base, const size_t exponent) {
    // (parentheses not required in next line)
    return (exponent == 0) ? 1 : (base * pow(base, exponent - 1));
}

#endif // KD_TREE_UTILS_HPP
