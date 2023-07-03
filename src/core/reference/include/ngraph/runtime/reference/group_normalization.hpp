// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_transform.hpp"

using namespace std;

namespace ngraph {
namespace runtime {
namespace reference {
namespace {

template <typename T>
T calc_norm(T x, T scale, T mean, T variance, T epsilon, T bias) {
    return scale * ((x - mean) / sqrt(variance + epsilon)) + bias;
}
}  // namespace

template <typename T>
void group_normalization(const T* const data,
                         const T* const scale,
                         const T* const bias,
                         T* const out,
                         const ov::Shape& data_shape,
                         const size_t num_groups,
                         const double epsilon) {
    const auto num_batches = data_shape[0];
    const auto num_channels = data_shape[1];
    const auto num_sub_channels = num_channels / num_groups;
    const auto data_size = shape_size(data_shape);
    const auto batch_size = data_size / num_batches;
    const auto channel_size = batch_size / num_channels;
    const auto group_size = num_sub_channels * channel_size;

    vector<T> group_mean(num_batches * num_groups);
    vector<T> group_variance(num_batches * num_groups);  // TODO consider keeping sqrt(variance + eps)
    for (size_t n = 0; n < num_batches; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            const auto group_begin = data + n * batch_size + g * group_size;
            const auto group_end = group_begin + group_size;
            const auto mean = accumulate(group_begin, group_end, static_cast<T>(0)) / group_size;
            const auto variance = accumulate(group_begin,
                                             group_end,
                                             static_cast<T>(0),
                                             [mean](const T acc, const T e) {
                                                 return acc + pow(e - mean, 2);
                                             }) /
                                  group_size;
            const auto i = n * num_groups + g;
            group_mean[i] = mean;
            group_variance[i] = variance;
        }
    }

    // TODO simplify it, `s' is needed only to obtain `c', while `g' might be got by dividing `c'
    const auto eps = static_cast<T>(epsilon);
    for (size_t n = 0; n < num_batches; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            for (size_t s = 0; s < num_sub_channels; ++s) {
                const auto c = n * num_groups + g * num_sub_channels + s;
                const auto channel_offset = c * channel_size;
                for (size_t i = channel_offset; i < channel_offset + channel_size; ++i)
                    out[i] = calc_norm(data[i], scale[c], group_mean[g], group_variance[g], eps, bias[c]);
            }
        }
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
