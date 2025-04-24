// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/abs.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of ReduceL1 operator.
 *
 * @param in             Input iterator to data.
 * @param out            Output iterator to results.
 * @param in_shape       Input shape.
 * @param reduction_axes Axes on which reduction is applied.
 */
template <class InputIt, class OutputIt>
void reduce_l1(InputIt in, OutputIt out, const Shape& in_shape, const AxisSet& reduction_axes) {
    using T = typename std::iterator_traits<OutputIt>::value_type;
    static_assert(std::is_same<typename std::iterator_traits<InputIt>::value_type, T>::value,
                  "Assume in and out same type.");

    const auto out_shape = ov::util::reduce(in_shape, reduction_axes);
    std::fill(out, std::next(out, shape_size(out_shape)), T(0));

    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);

    CoordinateTransformBasic input_transform(in_shape);
    for (const Coordinate& in_coord : input_transform) {
        constexpr uint64_t idx_init = 0;
        const auto out_coord = ov::util::reduce(in_coord, reduction_axes);

        const size_t in_idx = std::inner_product(in_coord.begin(), in_coord.end(), in_strides.begin(), idx_init);
        const size_t out_idx = std::inner_product(out_coord.begin(), out_coord.end(), out_strides.begin(), idx_init);

        out[out_idx] += func::abs(in[in_idx]);
    }
}
}  // namespace reference
}  // namespace ov
