// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"

namespace ov {
namespace reference {
/**
 * @brief Reference implementation of SliceScatter operator.
 *
 * @param data            Pointer to input 0 data containing values to be updated from `data`.
 * @param data_shape      Input 0 shape.
 * @param updates         Pointer to input 1 data containing updated values from `updates`.
 * @param updates_shape   Input 1 shape.
 * @param out             Pointer to output data.
 * @param elem_size       Element type size for data and updates input.
 * @param starts          Vector containing start coordinates for given axes.
 * @param steps           Vector containing step values for given axes.
 * @param axes            Vector containing axes indices.
 */
void slice_scatter(const char* data,
                   const Shape& data_shape,
                   const char* updates,
                   const Shape& updates_shape,
                   char* out,
                   size_t elem_size,
                   const std::vector<int64_t>& starts,
                   const std::vector<int64_t>& steps,
                   const AxisVector& axes) {
    std::memcpy(out, data, elem_size * shape_size(data_shape));
    const auto ind_size = starts.size();

    // Align inputs rank with data shape and normalize
    const auto data_rank = data_shape.size();
    ov::Coordinate aligned_starts(data_rank, 0);
    ov::Coordinate aligned_steps(data_rank, 1);
    auto tmp_axes = axes;
    if (tmp_axes.empty()) {
        tmp_axes.resize(ind_size);
        std::iota(tmp_axes.begin(), tmp_axes.end(), 0);
    }
    for (size_t i = 0; i < ind_size; ++i) {
        const auto& dim = data_shape[tmp_axes[i]];
        aligned_starts[tmp_axes[i]] = starts[i] >= 0
                                          ? std::min<int64_t>(starts[i], steps[i] < 0 ? dim - 1 : dim)
                                          : std::min<int64_t>(std::max<int64_t>(0, starts[i] + dim), dim - 1);
        aligned_steps[tmp_axes[i]] = steps[i];
    }

    // Slice elements
    const auto in_data_strides = row_major_strides(data_shape);
    const auto out_data_strides = row_major_strides(updates_shape);
    ov::Coordinate in_data_coord(aligned_starts);
    for (size_t upd_idx = 0; upd_idx < shape_size(updates_shape); ++upd_idx) {
        for (size_t i = 0; i < in_data_coord.size(); ++i) {
            in_data_coord[i] =
                aligned_starts[i] + (upd_idx / out_data_strides[i] % updates_shape[i]) * aligned_steps[i];
        }
        const auto in_idx = ov::coordinate_offset(in_data_coord, in_data_strides) * elem_size;
        std::memcpy(out + in_idx, updates + (upd_idx * elem_size), elem_size);
    }
}
}  // namespace reference
}  // namespace ov
