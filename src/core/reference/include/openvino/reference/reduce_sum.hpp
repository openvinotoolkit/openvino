// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace details {

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
bool isfinite(T x) {
    return std::isfinite(x);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, bfloat16>::value || std::is_same<T, float16>::value, bool>::type = true>
bool isfinite(T x) {
    return std::isfinite(static_cast<float>(x));
}

/**
 * @brief Performs one element summation based on Kahan algorithm to significantly reduce (integral types).
 *
 * @param in        Value to add with previous value of summation.
 * @param prev_sum  Previous value of summation (accumulator).
 * @return Compensate sum.
 */
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
constexpr T kahan_summation(const T in, const T prev_sum, T&) {
    return in + prev_sum;
}

/**
 * @brief Performs one element summation based on Kahan algorithm to significantly reduce (floating point types).
 *
 * @param in            Value to add with previous value of summation.
 * @param prev_sum      Previous value of summation (accumulator).
 * @param compensation  Accumulates the summation error.
 * @return Compensate sum.
 */
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T kahan_summation(const T in, const T prev_sum, T& compensation) {
    if (isfinite(in) && isfinite(prev_sum)) {
        T temp = prev_sum + (in - compensation);
        compensation = (temp - prev_sum) - (in - compensation);
        return temp;
    } else {
        return in + prev_sum;
    }
}
}  // namespace details

/**
 * @brief Reference implementation of ReduceSum operator.
 *
 * @param in             Input pointer to data.
 * @param out            Output pointer to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
template <typename T>
void reduce_sum(const T* in, T* out, const Shape& in_shape, const AxisSet& reduction_axes) {
    const auto out_shape = util::reduce(in_shape, reduction_axes);

    const auto out_size = shape_size(out_shape);
    std::vector<T> cs(out_size, T{0});
    std::fill(out, std::next(out, out_size), T{0});

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const auto& in_coord : input_transform) {
        const auto out_coord = util::reduce(in_coord, reduction_axes);
        const auto in_idx = coordinate_offset(in_coord, in_strides);
        const auto out_idx = coordinate_offset(out_coord, out_strides);

        out[out_idx] = details::kahan_summation(in[in_idx], out[out_idx], cs[out_idx]);
    }
}
}  // namespace reference
}  // namespace ov
