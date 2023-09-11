// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/reference/utils/coordinate_transform.hpp"
#include "openvino/reference/utils/type_util.hpp"
#include "shape_util.hpp"

namespace ov {
namespace reference {
namespace details {
///
/// \brief      Performs one element summation based on Kahan algorithm to
/// significantly reduce
///             the numerical error.
///
/// \param[in]  elem            Element to add into the accumulator.
/// \param      compensation    Variable that accumulates the error.
/// \param      sum             Result of compensated summation.
///
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
constexpr T kahan_summation(const T in, const T prev_sum, T&) {
    return in + prev_sum;
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T kahan_summation(const T in, const T prev_sum, T& compensation) {
    if (std::isfinite(in) && std::isfinite(prev_sum)) {
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
        constexpr uint64_t init_value = 0;
        const auto out_coord = util::reduce(in_coord, reduction_axes);

        const auto in_idx = std::inner_product(in_coord.begin(), in_coord.end(), in_strides.begin(), init_value);
        const auto out_idx = std::inner_product(out_coord.begin(), out_coord.end(), out_strides.begin(), init_value);

        out[out_idx] = details::kahan_summation(in[in_idx], out[out_idx], cs[out_idx]);
    }
}
}  // namespace reference
}  // namespace ov
