// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape.hpp"
#include "openvino/reference/reduce_mean.hpp"
#include "openvino/reference/reduce_sum.hpp"

namespace ov {
namespace reference {

template <typename T>
void group_normalization(const T* const data,
                         const T* const scale,
                         const T* const bias,
                         T* const out,
                         const Shape& data_shape,
                         const size_t num_groups,
                         const double epsilon) {
    using namespace std;

    const auto num_batches = data_shape[0];
    const auto num_channels = data_shape[1];
    const auto num_channels_in_group = num_channels / num_groups;
    const auto data_size = shape_size(data_shape);
    const auto batch_size = data_size / num_batches;
    const auto channel_size = batch_size / num_channels;
    const auto group_size = num_channels_in_group * channel_size;
    const auto eps = static_cast<T>(epsilon);

    for (size_t n = 0; n < num_batches; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            const auto group_begin = data + n * batch_size + g * group_size;
            const auto group_end = group_begin + group_size;
            std::vector<T> mean_value(1);
            reduce_mean(group_begin, mean_value.data(), Shape{group_size}, {0});
            T mean = mean_value[0];
            T variance = 0, err = 0;
            for_each(group_begin, group_end, [&](const T d) {
                variance = details::kahan_summation(static_cast<T>(pow(d - mean, 2)), variance, err);
            });
            variance /= group_size;
            const T standard_deviation = sqrt(variance + eps);

            for (size_t s = 0; s < num_channels_in_group; ++s) {
                const auto c = g * num_channels_in_group + s;
                const auto channel_offset = n * batch_size + c * channel_size;
                for (size_t i = channel_offset; i < channel_offset + channel_size; ++i)
                    out[i] = ((data[i] - mean) / standard_deviation) * scale[c] + bias[c];
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
