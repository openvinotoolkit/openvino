// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {
template <class T>
void slice_scatter(const T* data,
                   const Shape& data_shape,
                   const T* updates,
                   const Shape& updates_shape,
                   T* out,
                   const Shape& out_shape,
                   size_t elem_size,
                   const std::vector<int64_t>& starts,
                   const std::vector<int64_t>& steps,
                   const std::vector<int64_t>& axes) {
    std::memcpy(out, data, elem_size * shape_size(data_shape));
    const auto ind_size = starts.size();

    // Align inputs rank with data shape and normalize
    const auto data_rank = data_shape.size();
    std::vector<int64_t> aligned_starts(data_rank, 0);
    std::vector<int64_t> aligned_steps(data_rank, 1);
    for (size_t i = 0; i < ind_size; ++i) {
        int64_t axis = i;
        if (!axes.empty()) {
            axis = axes[i] >= 0 ? axes[i] : axes[i] + static_cast<int64_t>(data_rank);
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < data_rank,
                            "Slice `axes` arg has out of range value.");
        }
        const auto& dim = data_shape[axis];
        aligned_starts[axis] = starts[i] >= 0 ? std::min<int64_t>(starts[i], steps[i] < 0 ? dim - 1 : dim)
                                              : std::min<int64_t>(std::max<int64_t>(0, starts[i] + dim), dim - 1);
        aligned_steps[axis] = steps[i];
    }

    // Slice elements
    const auto in_data_strides = row_major_strides(data_shape);
    const auto out_data_strides = row_major_strides(updates_shape);
    std::vector<int64_t> in_data_coord(aligned_starts);
    for (size_t upd_idx = 0; upd_idx < shape_size(updates_shape); ++upd_idx) {
        for (size_t i = 0; i < in_data_coord.size(); ++i) {
            in_data_coord[i] =
                aligned_starts[i] + (upd_idx / out_data_strides[i] % updates_shape[i]) * aligned_steps[i];
        }
        const auto in_idx =
            std::inner_product(in_data_coord.begin(), in_data_coord.end(), in_data_strides.begin(), uint64_t(0));
        std::memcpy(out + in_idx, updates + upd_idx, elem_size);
    }
}
}  // namespace reference
}  // namespace ov
