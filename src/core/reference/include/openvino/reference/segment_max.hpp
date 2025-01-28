// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include <limits>
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

template <typename T, typename T_idx>
void segment_max(const T* data,
                 const Shape& data_shape,
                 const T_idx* segment_ids,
                 T* out,
                 const int64_t empty_segment_value) {
    const T_idx num_segments = *std::max_element(segment_ids, segment_ids + data_shape[0]) + 1;
    const size_t inner_dim_size = std::accumulate(data_shape.begin() + 1, data_shape.end(), 1, std::multiplies<size_t>());

    // Initialize max_values with empty_segment_value for each segment
    std::vector<std::vector<T>> max_values(num_segments, std::vector<T>(inner_dim_size, std::numeric_limits<T>::lowest()));
    std::vector<bool> segment_has_values(num_segments, false);

    // Iterate over each element in the first dimension
    for(size_t i = 0; i < data_shape[0]; ++i) {
        const T_idx segment_id = segment_ids[i];
        segment_has_values[segment_id] = true;
        // Iterate over each element in the inner dimensions
        for (size_t j = 0; j < inner_dim_size; ++j) {
            const size_t index = i * inner_dim_size + j;
            // Update the maximum value for the current segment and inner dimension
            max_values[segment_id][j] = std::max(max_values[segment_id][j], data[index]);
            std::cout << "Max value for segment " << segment_id << " at inner dimension " << j << " is " << max_values[segment_id][j] << std::endl;
        }
    }

    // Populate the output array with the maximum values for each segment
    for(T_idx segment_id = 0; segment_id < num_segments; ++segment_id) {
        for (size_t j = 0; j < inner_dim_size; ++j) {
            if (segment_has_values[segment_id]) {
                out[segment_id * inner_dim_size + j] = max_values[segment_id][j];
            } else {
                out[segment_id * inner_dim_size + j] = empty_segment_value;
            }
            std::cout << "Output value for segment " << segment_id << " at inner dimension " << j << " is " << out[segment_id * inner_dim_size + j] << std::endl;
        }
    }
}

}  // namespace reference
}  // namespace ov