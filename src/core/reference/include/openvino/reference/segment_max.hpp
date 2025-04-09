// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::reference {

template <typename T,
          typename T_idx,
          typename std::enable_if_t<std::is_same_v<typename std::decay_t<T_idx>, int64_t>>* = nullptr>
void segment_max(const T* data,
                 const Shape& data_shape,
                 const T_idx* segment_ids,
                 T* out,
                 const Shape& output_shape,
                 const T empty_segment_value) {
    const T_idx num_segments = output_shape[0];
    const auto inner_dim_size = shape_size(data_shape.begin() + 1, data_shape.end());

    // Initialize output with empty_segment_value
    std::fill(out, out + num_segments * inner_dim_size, empty_segment_value);

    // Iterate over each element in the first dimension
    for (size_t i = 0; i < data_shape[0]; ++i) {
        const T_idx segment_id = segment_ids[i];
        if (segment_id >= num_segments) {
            continue;
        }
        // Iterate over each element in the inner dimensions
        for (size_t j = 0; j < inner_dim_size; ++j) {
            const size_t index = i * inner_dim_size + j;
            const size_t out_index = segment_id * inner_dim_size + j;
            // Update the maximum value for the current segment and inner dimension
            out[out_index] = std::max(out[out_index], data[index]);
        }
    }
}

template <typename T,
          typename T_idx,
          typename std::enable_if_t<!std::is_same_v<typename std::decay_t<T_idx>, int64_t>>* = nullptr>
void segment_max(const T* data,
                 const Shape& data_shape,
                 const T_idx* segment_ids,
                 T* out,
                 const Shape& output_shape,
                 const T empty_segment_value) {
    std::vector<int64_t> segment_ids_int64(segment_ids, segment_ids + data_shape[0]);
    segment_max(data, data_shape, segment_ids_int64.data(), out, output_shape, empty_segment_value);
}

}  // namespace ov::reference
