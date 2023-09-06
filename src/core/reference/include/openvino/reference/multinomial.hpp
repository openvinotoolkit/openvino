// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include <openvino/reference/min.hpp>
#include <openvino/reference/max.hpp>
#include <openvino/reference/random_uniform.hpp>
#include <ngraph/shape.hpp>

namespace ov {
namespace reference {

namespace {
    template <typename T>
    void prefix_sum_with_normalize(const T* input,
            const T* output,
            const Shape& input_output_shape,
            T* min,
            T* max) {
        max[0] -= min[0];
        for(uint64_t i = 1; i<input_output_shape.at(0); ++i) {
                output[i] = input[i] + output[i-1];
                output[i] = (output[i] - min[i]) / max[i];
        }
    }
}

template <typename T>
void multinomial(const T* input,
    const Shape& input_shape,
    const T* num_samples, 
    const Shape& num_samples_shape,
    const T* output,
    const Shape& output_shape,
    const bool replacement, // TODO
    const bool log_probs,   // TODO
    const int64_t global_seed = 0, 
    const int64_t op_seed = 0) {

        float one = 1.0f;
        float zero = 0.0f;
        size_t count = num_samples_shape.at(0);
        std::pair<uint64_t, uint64_t> initial_state(0, 0);

        float* uniform_samples = malloc(sizeof(float) * count);
        random_uniform(*count, *zero, *one, uniform_samples, *one, ov::element::f32, global_seed, op_seed, initial_state);

        ov::AxisSet input_axes();
        for(uint64_t i = 0; input_shape.size(); i++) {
            axes.push_back(i);
        }
        T* min_value = malloc(sizeof(T));
        min<T>(input, min_value, input_shape, ov::AxisSet(axes));
        T* max_value = malloc(sizeof(T));
        max<T>(input, max_value, input_shape, ov::AxisSet(axes));
        T* cdf = calloc(count, sizeof(T));
        prefix_sum_with_normalize<T>(input, cdf, input_shape, min_value, max_value);

        for(size_t i = 0; i < count; ++i) {
            float random_nr = uniform_samples[i];
            for (size_t j = 0; j < count; ++j) {
                if (j + 1 < count) {
                    if (cdf[j] == cdf[j+1]) {
                        continue;
                    }
                    if (std::abs((float)cdf[j] - random_nr) < std::abs((float)cdf[j + 1] - random_nr)) {
                        output[i] = j;
                        break;
                    }
                } else {
                    output[i] = j;
                }
            }
        }

        free(uniform_samples);
        free (min_value);
        free (max_value);
        free (cdf);
    }
}  // namespace reference
}  // namespace ov
