// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {

// Rational polynomial approximation of erfinv.
// Based on: Mike Giles, "Approximating the erfinv function,"
// GPU Computing Gems Jade Edition, pp 109-116, 2011.
// https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
//
// Handles special values:
//   erfinv(0)      =  0
//   erfinv(1)      =  +inf
//   erfinv(-1)     =  -inf
//   erfinv(|x|>1)  =  NaN
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T erfinv(const T val) {
    using F = double;
    const F x = static_cast<F>(val);

    if (x == F{0})
        return T{0};
    if (x >= F{1})
        return x > F{1} ? T{std::numeric_limits<float>::quiet_NaN()} : T{std::numeric_limits<float>::infinity()};
    if (x <= F{-1})
        return x < F{-1} ? T{std::numeric_limits<float>::quiet_NaN()} : T{-std::numeric_limits<float>::infinity()};

    // w = -log((1-x)*(1+x))
    const F w = -std::log((F{1} - x) * (F{1} + x));
    F r;
    if (w < 5.0) {
        const F s = w - F{2.5};
        r =  2.81022636e-08;
        r =  3.43273939e-07 + r * s;
        r = -3.5233877e-06  + r * s;
        r = -4.39150654e-06 + r * s;
        r =  0.00021858087  + r * s;
        r = -0.00125372503  + r * s;
        r = -0.00417768164  + r * s;
        r =  0.246640727    + r * s;
        r =  1.50140941     + r * s;
    } else {
        const F s = std::sqrt(w) - F{3.0};
        r = -0.000200214257;
        r =  0.000100950558 + r * s;
        r =  0.00134934322  + r * s;
        r = -0.00367342844  + r * s;
        r =  0.00573950773  + r * s;
        r = -0.0076224613   + r * s;
        r = -0.00943887047  + r * s;
        r =  1.00167406     + r * s;
        r =  2.83297682     + r * s;
    }
    return static_cast<T>(r * x);
}

}  // namespace func

/**
 * @brief Reference implementation of ErfInv operator.
 *
 * Computes the inverse error function element-wise:
 *   y = erfinv(x)  such that  erf(y) = x,  for x in (-1, 1).
 * Special values:
 *   erfinv(0)      =  0
 *   erfinv(1)      =  +inf
 *   erfinv(-1)     =  -inf
 *   erfinv(|x|>1)  =  NaN
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void erfinv(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, func::erfinv<T>);
}
}  // namespace reference
}  // namespace ov
