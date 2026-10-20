// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/normalize_l2.hpp"

namespace ov::reference {

template <typename T>
void gated_delta_net(const T* q_data,
                     const T* k_data,
                     const T* v_data,
                     const T* state_data,
                     const T* gate_data,
                     const T* beta_data,
                     T* out_data,
                     T* out_state,
                     const Shape& q_shape,
                     const Shape& v_shape,
                     bool fuse_qk_l2norm,
                     T q_l2_norm_eps,
                     T k_l2_norm_eps) {
    const size_t B = q_shape[0];
    const size_t S = q_shape[1];
    const size_t qk_H = q_shape[2];
    const size_t D = q_shape[3];
    const size_t v_H = v_shape[2];
    const size_t Dv = v_shape[3];
    const size_t group_size = v_H / qk_H;

    const T attn_scale = static_cast<T>(1) / std::sqrt(static_cast<T>(D));

    const size_t qk_stride_batch = S * qk_H * D;
    const size_t v_stride_batch = S * v_H * Dv;
    const size_t gate_beta_stride_batch = S * v_H;

    auto dot_product = [](const T* a, const T* b, size_t n) {
        T result = static_cast<T>(0);
        for (size_t i = 0; i < n; i++) {
            result += a[i] * b[i];
        }
        return result;
    };

    const Shape norm_shape{D};
    const AxisSet norm_axes{0};

    for (size_t b = 0; b < B; b++) {
        for (size_t h_v = 0; h_v < v_H; h_v++) {
            const size_t h_qk = h_v / group_size;
            for (size_t d_v = 0; d_v < Dv; d_v++) {
                const size_t state_offset = b * v_H * D * Dv + h_v * D * Dv + d_v;
                T* state_ptr = out_state + state_offset;

                std::vector<T> local_state(D);
                const T* src_state = state_data + state_offset;
                for (size_t d = 0; d < D; d++) {
                    local_state[d] = src_state[d * Dv];
                }

                for (size_t t = 0; t < S; t++) {
                    const T* q_ptr = q_data + b * qk_stride_batch + t * qk_H * D + h_qk * D;
                    const T* k_ptr = k_data + b * qk_stride_batch + t * qk_H * D + h_qk * D;

                    std::vector<T> q_vec(q_ptr, q_ptr + D);
                    std::vector<T> k_vec(k_ptr, k_ptr + D);

                    if (fuse_qk_l2norm) {
                        normalize_l2(q_vec.data(),
                                     q_vec.data(),
                                     norm_shape,
                                     norm_axes,
                                     static_cast<float>(q_l2_norm_eps),
                                     op::EpsMode::ADD);
                        normalize_l2(k_vec.data(),
                                     k_vec.data(),
                                     norm_shape,
                                     norm_axes,
                                     static_cast<float>(k_l2_norm_eps),
                                     op::EpsMode::ADD);
                    }

                    for (size_t i = 0; i < D; i++)
                        q_vec[i] *= attn_scale;

                    T g = std::exp(gate_data[b * gate_beta_stride_batch + t * v_H + h_v]);
                    T bt = beta_data[b * gate_beta_stride_batch + t * v_H + h_v];

                    for (size_t d = 0; d < D; d++) {
                        local_state[d] *= g;
                    }

                    T h_k = dot_product(local_state.data(), k_vec.data(), D);

                    T v_val = v_data[b * v_stride_batch + t * v_H * Dv + h_v * Dv + d_v] - h_k;

                    T update_scale = v_val * bt;
                    for (size_t d = 0; d < D; d++) {
                        local_state[d] += k_vec[d] * update_scale;
                    }

                    out_data[b * v_stride_batch + t * v_H * Dv + h_v * Dv + d_v] =
                        dot_product(local_state.data(), q_vec.data(), D);
                }

                for (size_t d = 0; d < D; d++) {
                    state_ptr[d * Dv] = local_state[d];
                }
            }
        }
    }
}

}  // namespace ov::reference
