// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/reshape.hpp"

#include <cstring>
#include <numeric>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_range.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
std::vector<size_t> reorder(const std::vector<size_t>& origin, const AxisVector& order) {
    std::vector<size_t> reordered = origin;
    auto out = begin(reordered);
    NGRAPH_CHECK(origin.size() <= order.size());
    for (size_t i = 0; i < origin.size(); ++i) {
        *out = origin.at(order[i]);
        ++out;
    }
    return reordered;
}
}  // namespace

void reshape(const char* arg,
             char* out,
             const Shape& in_shape,
             const AxisVector& in_axis_order,
             const Shape& out_shape,
             size_t elem_size) {
    if (shape_size(in_shape) == 1) {
        std::memcpy(out, arg, elem_size);
        return;
    }

    char* output = out;
    const char* const output_end = out + shape_size(out_shape) * elem_size;
    const auto axis_strides = reorder(row_major_strides(in_shape), in_axis_order);
    for (const auto& coordinate : CoordinateTransformBasic(reorder(in_shape, in_axis_order))) {
        if (output >= output_end) {
            break;
        }
        const auto elem_offset = std::inner_product(begin(coordinate), end(coordinate), begin(axis_strides), 0ll);
        const auto input = arg + elem_offset * elem_size;
        std::memcpy(output, input, elem_size);
        output += elem_size;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
