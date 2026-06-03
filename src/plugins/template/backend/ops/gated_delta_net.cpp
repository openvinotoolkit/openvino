// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/gated_delta_net.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::GatedDeltaNet>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto& q_shape = inputs[0].get_shape();
    const auto& v_shape = inputs[2].get_shape();
    const auto& state_shape = inputs[3].get_shape();

    const size_t B = q_shape[0];
    const size_t S = q_shape[1];
    const size_t qk_H = q_shape[2];
    const size_t D = q_shape[3];
    const size_t v_H = v_shape[2];
    const size_t Dv = v_shape[3];

    OPENVINO_ASSERT(qk_H > 0 && v_H >= qk_H && v_H % qk_H == 0,
                    "GatedDeltaNet evaluate: v_H (",
                    v_H,
                    ") must be a positive multiple of qk_H (",
                    qk_H,
                    ")");
    const size_t group_size = v_H / qk_H;

    outputs[0].set_shape(v_shape);
    outputs[1].set_shape(state_shape);

    const T* q_data = inputs[0].data<const T>();
    const T* k_data = inputs[1].data<const T>();
    const T* v_data = inputs[2].data<const T>();
    const T* state_data = inputs[3].data<const T>();
    const T* gate_data = inputs[4].data<const T>();
    const T* beta_data = inputs[5].data<const T>();

    T* out_state = outputs[1].data<T>();
    T* out_data = outputs[0].data<T>();
    const T attn_scale = static_cast<T>(1) / std::sqrt(static_cast<T>(D));

    const size_t qk_stride_batch = S * qk_H * D;
    const size_t v_stride_batch = S * v_H * Dv;
    const size_t gate_beta_stride_batch = S * v_H;

    const bool fuse_qk_l2norm = op->get_fuse_qk_l2norm();
    const T q_l2_norm_eps = static_cast<T>(op->get_q_l2_norm_eps());
    const T k_l2_norm_eps = static_cast<T>(op->get_k_l2_norm_eps());

    auto dot_product = [](const T* a, const T* b, size_t n) {
        T result = static_cast<T>(0);
        for (size_t i = 0; i < n; i++) {
            result += a[i] * b[i];
        }
        return result;
    };

    auto l2norm = [](std::vector<T>& vec, T eps) {
        T sum = static_cast<T>(0);
        for (size_t i = 0; i < vec.size(); i++)
            sum += vec[i] * vec[i];
        sum = static_cast<T>(1) / std::sqrt(sum + eps);
        for (size_t i = 0; i < vec.size(); i++)
            vec[i] *= sum;
    };

    for (size_t b = 0; b < B; b++) {
        for (size_t h_v = 0; h_v < v_H; h_v++) {
            const size_t h_qk = h_v / group_size;
            for (size_t d_v = 0; d_v < Dv; d_v++) {
                // state layout: [B, v_H, D, Dv]
                const size_t state_offset = b * v_H * D * Dv + h_v * D * Dv + d_v;
                T* state_ptr = out_state + state_offset;

                // Load initial state from input
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
                        l2norm(q_vec, q_l2_norm_eps);
                        l2norm(k_vec, k_l2_norm_eps);
                    }

                    // Scale q
                    for (size_t i = 0; i < D; i++)
                        q_vec[i] *= attn_scale;

                    // gate[b, t, h_v] — layout [B, S, v_H]
                    T g = std::exp(gate_data[b * gate_beta_stride_batch + t * v_H + h_v]);
                    T bt = beta_data[b * gate_beta_stride_batch + t * v_H + h_v];

                    // Decay state: state *= g
                    for (size_t d = 0; d < D; d++) {
                        local_state[d] *= g;
                    }

                    // h_k = dot(state, k)
                    T h_k = dot_product(local_state.data(), k_vec.data(), D);

                    // delta: v_val = value[b, t, h_v, d_v] - h_k
                    T v_val = v_data[b * v_stride_batch + t * v_H * Dv + h_v * Dv + d_v] - h_k;

                    // Update state: state += k * (v_val * beta)
                    T update_scale = v_val * bt;
                    for (size_t d = 0; d < D; d++) {
                        local_state[d] += k_vec[d] * update_scale;
                    }

                    // Output: out[b, t, h_v, d_v] = dot(state, q)
                    out_data[b * v_stride_batch + t * v_H * Dv + h_v * Dv + d_v] =
                        dot_product(local_state.data(), q_vec.data(), D);
                }

                // Write final state back
                for (size_t d = 0; d < D; d++) {
                    state_ptr[d * Dv] = local_state[d];
                }
            }
        }
    }
    return true;
}

template <>
bool evaluate_node<ov::op::internal::GatedDeltaNet>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    const auto& element_type = node->get_input_element_type(0);

    switch (element_type) {
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node<GatedDeltaNet>()");
    }
}
