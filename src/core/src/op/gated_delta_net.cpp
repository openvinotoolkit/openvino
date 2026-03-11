// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace {

// Validates input rank and type for a node input.
inline void input_check(const ov::Node* node,
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

namespace ov {
namespace op {

GatedDeltaNet::GatedDeltaNet(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void GatedDeltaNet::validate_and_infer_types() {
    OV_OP_SCOPE(GatedDeltaNet_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 6, "GatedDeltaNet expects 6 inputs, but it has ", get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 1, "key", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 2, "value", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 3, "recurrent_state", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 4, "gate", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 5, "beta", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});

    // batch, seq_len, head_num, head_size
    const auto& query_ps = get_input_partial_shape(0);
    const auto& key_ps = get_input_partial_shape(1);
    const auto& value_ps = get_input_partial_shape(2);
    const auto& state_ps = get_input_partial_shape(3);
    const auto& gate_ps = get_input_partial_shape(4);
    const auto& beta_ps = get_input_partial_shape(5);

    const auto q_head_num = query_ps[2];
    const auto k_head_num = key_ps[2];
    const auto v_head_num = value_ps[2];

    const auto k_head_size = key_ps[3];
    const auto q_head_size = query_ps[3];
    const auto v_head_size = value_ps[3];

    NODE_VALIDATION_CHECK(this,
                          q_head_num.compatible(k_head_num) && q_head_num.compatible(v_head_num),
                          "The number of heads in query key and value should be the same, but got ",
                          q_head_num,
                          " and ",
                          k_head_num,
                          ".");

    NODE_VALIDATION_CHECK(this,
                          k_head_size.compatible(q_head_size),
                          "The head size in key and query should be the same, but got ",
                          k_head_size,
                          " and ",
                          v_head_size,
                          ".");

    const auto gate_head_num = gate_ps[2];
    const auto beta_head_num = beta_ps[2];

    NODE_VALIDATION_CHECK(this,
                          gate_head_num.compatible(beta_head_num) && gate_head_num.compatible(q_head_num),
                          "The number of heads in gate, beta, and query should be the same, but got ",
                          gate_head_num,
                          " and ",
                          beta_head_num,
                          ".");

    // [batch, v_head_nums, k_head_size, v_head_size]
    const auto state_head_num = state_ps[1];
    const auto state_hidden_size_0 = state_ps[2];
    const auto state_hidden_size_1 = state_ps[3];
    NODE_VALIDATION_CHECK(this,
                          state_head_num.compatible(v_head_num),
                          "The number of heads in recurrent_state and value should be the same, but got ",
                          state_head_num,
                          " and ",
                          v_head_num,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          state_hidden_size_0.compatible(k_head_size),
                          "The [-2] dim in shape of recurrent_state and key should be the same, but got ",
                          state_hidden_size_0,
                          " and ",
                          k_head_size,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          state_hidden_size_1.compatible(v_head_size),
                          "The [-1] dim in shape of recurrent_state and value should be the same, but got ",
                          state_hidden_size_1,
                          " and ",
                          v_head_size,
                          ".");
    // output has the same shape and type as input value, output state has the same shape and type as input
    // recurrent_state
    set_output_type(0, get_input_element_type(2), value_ps);
    set_output_type(1, get_input_element_type(3), state_ps);
}

bool GatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GatedDeltaNet_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("fuse_qk_l2norm", m_config.fuse_qk_l2norm);
    visitor.on_attribute("fuse_q_scale", m_config.fuse_q_scale);
    visitor.on_attribute("q_l2_norm_eps", m_config.q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_config.k_l2_norm_eps);
    visitor.finish_structure();
    return true;
}

std::shared_ptr<ov::Node> GatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    auto cloned = std::make_shared<GatedDeltaNet>(new_args);
    cloned->m_config = m_config;
    return cloned;
}

}  // namespace op
}  // namespace ov
