// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
static inline int
entry_index(int width, int height, int coords, int classes, int outputs, int batch, int location, int entry) {
    int n = location / (width * height);
    int loc = location % (width * height);
    return batch * outputs + n * width * height * (coords + classes + 1) + entry * width * height + loc;
}

template <typename T>
static inline T sigmoid(float x) {
    return static_cast<T>(1.f / (1.f + std::exp(-x)));
}
template <typename T>
static inline void softmax_generic(const T* src_data, T* dst_data, int batches, int channels, int height, int width) {
    const int area = height * width;
    for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
        const int offset = batch_idx * channels * area;
        for (int i = 0; i < height * width; i++) {
            T max = src_data[batch_idx * channels * area + i];
            for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
                T val = src_data[offset + channel_idx * area + i];
                max = std::max(max, val);
            }

            T sum = 0;
            for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
                dst_data[offset + channel_idx * area + i] =
                    static_cast<T>(std::exp(src_data[offset + channel_idx * area + i] - max));
                sum += dst_data[offset + channel_idx * area + i];
            }

            for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
                dst_data[offset + channel_idx * area + i] /= sum;
            }
        }
    }
}

template <typename T>
void region_yolo(const T* input,
                 T* output,
                 const Shape& input_shape,
                 const int coords,
                 const int classes,
                 const int regions,
                 const bool do_softmax,
                 const std::vector<int64_t>& mask) {
    OPENVINO_ASSERT(input_shape.size() == 4);

    const int batches = static_cast<int>(input_shape[0]);
    const int height = static_cast<int>(input_shape[2]);
    const int width = static_cast<int>(input_shape[3]);

    const auto mask_size = mask.size();

    size_t num_regions = 0;
    size_t end_index = 0;
    size_t output_size = 0;

    if (do_softmax) {
        // Region layer (Yolo v2)
        num_regions = regions;
        end_index = width * height;
        output_size = shape_size(input_shape);
    } else {
        // Yolo layer (Yolo v3)
        num_regions = mask_size;
        end_index = width * height * (classes + 1);
        output_size = width * height * num_regions * (classes + coords + 1);
    }

    std::copy(input, input + output_size, output);

    const int inputs_size = width * height * static_cast<int>(num_regions) * (classes + coords + 1);

    for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
        for (int n = 0; n < static_cast<int>(num_regions); n++) {
            int index = entry_index(width, height, coords, classes, inputs_size, batch_idx, n * width * height, 0);
            std::transform(output + index, output + index + 2 * width * height, output + index, [](T elem) {
                return sigmoid<T>(static_cast<float>(elem));
            });

            index = entry_index(width, height, coords, classes, inputs_size, batch_idx, n * width * height, coords);
            std::transform(output + index, output + index + end_index, output + index, [](T elem) {
                return sigmoid<T>(static_cast<float>(elem));
            });
        }
    }

    if (do_softmax) {
        int index = entry_index(width, height, coords, classes, inputs_size, 0, 0, coords + 1);
        int batch_offset = inputs_size / regions;
        for (int batch_idx = 0; batch_idx < batches * regions; batch_idx++) {
            softmax_generic<T>(input + index + batch_idx * batch_offset,
                               output + index + batch_idx * batch_offset,
                               1,
                               classes,
                               height,
                               width);
        }
    }
}

}  // namespace reference
}  // namespace ov
