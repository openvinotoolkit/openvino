// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

template <typename T, typename T_idx>
void segment_max(const T* data,
                 const Shape& data_shape,
                 const T_idx* segment_ids,
                 T* out,
                 const Shape& output_shape,
                 const T empty_segment_value) {
    const T_idx num_segments = output_shape[0];
    const size_t inner_dim_size =
        std::accumulate(data_shape.begin() + 1, data_shape.end(), 1, std::multiplies<size_t>());
    std::vector<std::vector<T>> max_values(num_segments,
                                           std::vector<T>(inner_dim_size, std::numeric_limits<T>::lowest()));
    std::vector<bool> segment_has_values(num_segments, false);

    // Iterate over each element in the first dimension
    T_idx base_idx;
    for (size_t i = 0; i < data_shape[0]; ++i) {
        const T_idx segment_id = segment_ids[i];
        if (segment_id >= num_segments) {
            break;
        }
        segment_has_values[segment_id] = true;
        // Iterate over each element in the inner dimensions
        base_idx = i * inner_dim_size;
        for (size_t j = 0; j < inner_dim_size; ++j) {
            // Update the maximum value for the current segment and inner dimension
            max_values[segment_id][j] = std::max(max_values[segment_id][j], data[base_idx + j]);
        }
    }

    // Populate the output array with the maximum values for each segment
    for (T_idx segment_id = 0; segment_id < num_segments; ++segment_id) {
        base_idx = segment_id * inner_dim_size;
        for (size_t j = 0; j < inner_dim_size; ++j) {
            out[base_idx + j] = segment_has_values[segment_id] ? max_values[segment_id][j] : empty_segment_value;
        }
    }
}

}  // namespace reference
}  // namespace ov