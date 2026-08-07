// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/core/shape.hpp"

namespace ov::reference {

template <typename T>
void selective_ssm(const T* A_data,
                   const T* dt_data,
                   const T* B_data,
                   const T* x_data,
                   const T* C_data,
                   const T* state_data,
                   T* out_data,
                   T* out_state,
                   const Shape& x_shape,
                   const Shape& B_shape) {
    const size_t batch = x_shape[0];
    const size_t seq_len = x_shape[1];
    const size_t num_heads = x_shape[2];
    const size_t head_dim = x_shape[3];
    const size_t num_groups = B_shape[2];
    const size_t state_size = B_shape[3];
    const size_t heads_per_group = num_heads / num_groups;

    const size_t state_batch_stride = num_heads * head_dim * state_size;
    const size_t x_batch_stride = seq_len * num_heads * head_dim;
    const size_t dt_batch_stride = seq_len * num_heads;
    const size_t bc_batch_stride = seq_len * num_groups * state_size;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            const size_t g = h / heads_per_group;
            for (size_t p = 0; p < head_dim; ++p) {
                const size_t state_offset = b * state_batch_stride + h * head_dim * state_size + p * state_size;
                for (size_t n = 0; n < state_size; ++n) {
                    out_state[state_offset + n] = state_data[state_offset + n];
                }
            }

            for (size_t t = 0; t < seq_len; ++t) {
                const size_t dt_offset = b * dt_batch_stride + t * num_heads + h;
                const T dA = static_cast<T>(std::exp(static_cast<float>(A_data[h] * dt_data[dt_offset])));
                const size_t bc_base = b * bc_batch_stride + t * num_groups * state_size + g * state_size;
                const size_t x_base = b * x_batch_stride + t * num_heads * head_dim + h * head_dim;
                const size_t out_base = b * x_batch_stride + t * num_heads * head_dim + h * head_dim;

                for (size_t p = 0; p < head_dim; ++p) {
                    const size_t state_offset = b * state_batch_stride + h * head_dim * state_size + p * state_size;
                    const T x_val = x_data[x_base + p];
                    T acc = static_cast<T>(0);
                    for (size_t n = 0; n < state_size; ++n) {
                        const T new_state = out_state[state_offset + n] * dA + x_val * dt_data[dt_offset] * B_data[bc_base + n];
                        out_state[state_offset + n] = new_state;
                        acc += new_state * C_data[bc_base + n];
                    }
                    out_data[out_base + p] = acc;
                }
            }
        }
    }
}

}  // namespace ov::reference
