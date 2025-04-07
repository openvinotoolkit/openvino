// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/slice.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/reference/utils/coordinate_range.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace reference {

void slice(const char* data,
           const Shape& data_shape,
           char* out,
           const Shape& out_shape,
           size_t elem_size,
           const std::vector<int64_t>& starts,
           const std::vector<int64_t>& steps,
           const std::vector<int64_t>& axes) {
    const auto ind_size = starts.size();
    OPENVINO_ASSERT(steps.size() == ind_size && axes.size() == ind_size,
                    "Slice starts, steps, axes args need to have the same size.");
    OPENVINO_ASSERT(data_shape.size() == out_shape.size(),
                    "Slice output data rank need to be equal to input data rank.");

    // Align inputs rank with data shape and normalize
    const auto data_rank = data_shape.size();
    std::vector<int64_t> aligned_starts(data_rank, 0);
    std::vector<int64_t> aligned_steps(data_rank, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
        const int64_t axis = axes[i] >= 0 ? axes[i] : axes[i] + static_cast<int64_t>(data_rank);
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < data_rank, "Slice `axes` arg has out of range value.");
        const auto& dim = data_shape[axis];
        aligned_starts[axis] = starts[i] >= 0 ? std::min<int64_t>(starts[i], steps[i] < 0 ? dim - 1 : dim)
                                              : std::min<int64_t>(std::max<int64_t>(0, starts[i] + dim), dim - 1);
        aligned_steps[axis] = steps[i];
    }

    // Slice elements
    const auto in_data_strides = row_major_strides(data_shape);
    const auto out_data_strides = row_major_strides(out_shape);
    std::vector<int64_t> in_data_coord(aligned_starts);
    for (size_t out_idx = 0; out_idx < shape_size(out_shape); ++out_idx) {
        for (size_t i = 0; i < in_data_coord.size(); ++i) {
            in_data_coord[i] = aligned_starts[i] + (out_idx / out_data_strides[i] % out_shape[i]) * aligned_steps[i];
        }
        const auto in_idx =
            std::inner_product(in_data_coord.begin(), in_data_coord.end(), in_data_strides.begin(), uint64_t(0));
        const auto in_mem = data + in_idx * elem_size;
        std::memcpy(out, in_mem, elem_size);
        out += elem_size;
    }
}

void slice(const char* arg,
           char* out,
           const Shape& arg_shape,
           const Coordinate& lower_bounds,
           const Coordinate& upper_bounds,
           const Strides& strides,
           const Shape& out_shape,
           size_t elem_size) {
    const auto rank = arg_shape.size();
    OPENVINO_ASSERT(
        lower_bounds.size() == rank && upper_bounds.size() == rank && strides.size() == rank &&
            out_shape.size() == rank,
        "arg_shape, lower_bounds, upper_bounds, strides and out_shape are expected to have the same rank equal ",
        rank);

    auto expected_out_shape = Shape(arg_shape);
    for (size_t i = 0; i < rank; ++i)
        expected_out_shape[i] = util::ceil_div(upper_bounds[i] - lower_bounds[i], strides[i]);
    OPENVINO_ASSERT(out_shape == expected_out_shape,
                    "Expected output shape is ",
                    expected_out_shape,
                    ". Got ",
                    out_shape);

    auto dst_mem = out;

    for (const auto& range : coordinates::slice(arg_shape, lower_bounds, upper_bounds, strides)) {
        auto src_index = range.begin_index;
        for (size_t i = 0; i < range.element_number; src_index += range.step, ++i) {
            const auto src_mem = arg + src_index * elem_size;
            std::memcpy(dst_mem, src_mem, elem_size);
            std::advance(dst_mem, elem_size);
        }
    }
}
}  // namespace reference
}  // namespace ov
