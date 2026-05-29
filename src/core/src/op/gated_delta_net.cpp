// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include <cmath>
#include <cstddef>

#include "dimension_util.hpp"
#include "gated_delta_net_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace {

// Validates input rank and type for a node input.
inline void gdn_input_check(const ov::Node* node,
                            size_t idx,
                            const std::string_view input_name,
                            std::initializer_list<ov::Rank>&& allowed_ranks,
                            const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return !rank.is_dynamic() && is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return !type.is_dynamic() && (allowed_types.empty() || it != allowed_types.end());
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}
}  // namespace

namespace ov::op::internal {

GatedDeltaNet::GatedDeltaNet(const Output<Node>& query,
                             const Output<Node>& key,
                             const Output<Node>& value,
                             const Output<Node>& recurrent_state,
                             const Output<Node>& gate,
                             const Output<Node>& beta,
                             bool fuse_qk_l2norm,
                             float q_l2_norm_eps,
                             float k_l2_norm_eps)
    : Op({query, key, value, recurrent_state, gate, beta}),
      m_fuse_qk_l2norm(fuse_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

GatedDeltaNet::GatedDeltaNet(const ov::OutputVector& args,
                             bool fuse_qk_l2norm,
                             float q_l2_norm_eps,
                             float k_l2_norm_eps)
    : ov::op::Op(args),
      m_fuse_qk_l2norm(fuse_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

void GatedDeltaNet::validate_and_infer_types() {
    OV_OP_SCOPE(GatedDeltaNet_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 6, "GatedDeltaNet expects 6 inputs, but it has ", get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    gdn_input_check(this, 0, "query", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    gdn_input_check(this, 1, "key", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    gdn_input_check(this, 2, "value", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    gdn_input_check(this, 3, "recurrent_state", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    gdn_input_check(this, 4, "gate", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    gdn_input_check(this, 5, "beta", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, get_input_element_type(3), output_shapes[1]);
}

bool GatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GatedDeltaNet_visit_attributes);
    visitor.on_attribute("fuse_qk_l2norm", m_fuse_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", m_q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_k_l2_norm_eps);
    return true;
}

std::shared_ptr<ov::Node> GatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    auto cloned = std::make_shared<GatedDeltaNet>(new_args, m_fuse_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps);
    return cloned;
}

bool GatedDeltaNet::has_evaluate() const {
    for (size_t i = 0; i < get_input_size(); i++) {
        if (get_input_element_type(i) != ov::element::f32)
            return false;
    }
    for (size_t i = 0; i < get_output_size(); i++) {
        if (get_output_element_type(i) != ov::element::f32)
            return false;
    }
    return true;
}

bool GatedDeltaNet::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(GatedDeltaNet_evaluate);

    const auto& q_tensor = inputs[0];
    const auto& k_tensor = inputs[1];
    const auto& v_tensor = inputs[2];
    const auto& state_tensor = inputs[3];
    const auto& gate_tensor = inputs[4];
    const auto& beta_tensor = inputs[5];

    const auto& q_shape = q_tensor.get_shape();
    const auto& v_shape = v_tensor.get_shape();
    const auto& state_shape = state_tensor.get_shape();

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

    const float* q_data = static_cast<const float*>(q_tensor.data());
    const float* k_data = static_cast<const float*>(k_tensor.data());
    const float* v_data = static_cast<const float*>(v_tensor.data());
    const float* gate_data = static_cast<const float*>(gate_tensor.data());
    const float* beta_data = static_cast<const float*>(beta_tensor.data());

    float* out_state = static_cast<float*>(outputs[1].data());
    float* out_data = static_cast<float*>(outputs[0].data());
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));

    const size_t qk_stride_batch = S * qk_H * D;
    const size_t v_stride_batch = S * v_H * Dv;
    const size_t gate_beta_stride_batch = S * v_H;

    auto dot_product = [](const float* a, const float* b, size_t n, size_t a_stride = 1) {
        float result = 0.0f;
        for (size_t i = 0; i < n; i++) {
            result += a[i * a_stride] * b[i];
        }
        return result;
    };

    for (size_t b = 0; b < B; b++) {
        for (size_t h_v = 0; h_v < v_H; h_v++) {
            const size_t h_qk = h_v / group_size;
            for (size_t d_v = 0; d_v < Dv; d_v++) {
                // state slice: state[b, h_v, :, d_v] — D elements with stride Dv
                // state layout: [B, v_H, D, Dv]
                const size_t state_offset = b * v_H * D * Dv + h_v * D * Dv + d_v;
                float* state_ptr = out_state + state_offset;

                // Load initial state from input
                std::vector<float> local_state(D);
                const float* src_state = static_cast<const float*>(state_tensor.data()) + state_offset;
                for (size_t d = 0; d < D; d++) {
                    local_state[d] = src_state[d * Dv];
                }

                for (size_t t = 0; t < S; t++) {
                    const float* q_ptr = q_data + b * qk_stride_batch + t * qk_H * D + h_qk * D;
                    const float* k_ptr = k_data + b * qk_stride_batch + t * qk_H * D + h_qk * D;

                    // L2-normalize q and k
                    std::vector<float> q_vec(q_ptr, q_ptr + D);
                    std::vector<float> k_vec(k_ptr, k_ptr + D);

                    if (m_fuse_qk_l2norm) {
                        auto l2norm = [](std::vector<float>& vec, float eps) {
                            float sum = 0.0f;
                            for (const auto v : vec)
                                sum += v * v;
                            sum = 1.0f / std::sqrt(sum + eps);
                            for (auto& v : vec)
                                v *= sum;
                        };
                        l2norm(q_vec, m_q_l2_norm_eps);
                        l2norm(k_vec, m_k_l2_norm_eps);
                    }

                    // Scale q
                    for (auto& v : q_vec)
                        v *= attn_scale;

                    // gate[b, t, h_v] — layout [B, S, v_H]
                    float g = std::exp(gate_data[b * gate_beta_stride_batch + t * v_H + h_v]);
                    float bt = beta_data[b * gate_beta_stride_batch + t * v_H + h_v];

                    // Decay state: state *= g
                    for (size_t d = 0; d < D; d++) {
                        local_state[d] *= g;
                    }

                    // h_k = dot(state, k)
                    float h_k = dot_product(local_state.data(), k_vec.data(), D);

                    // delta: v_val = value[b, t, h_v, d_v] - h_k
                    float v_val = v_data[b * v_stride_batch + t * v_H * Dv + h_v * Dv + d_v] - h_k;

                    // Update state: state += k * (v_val * beta)
                    float update_scale = v_val * bt;
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

}  // namespace ov::op::internal
