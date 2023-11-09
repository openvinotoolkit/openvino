// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdp.hpp"

#include <algorithm>

#include "transformations/itt.hpp"

ov::intel_cpu::ScaledDotProductAttentionNode::ScaledDotProductAttentionNode(const OutputVector& args, const Config& cfg)
    : Op(args),
      m_config(cfg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::ScaledDotProductAttentionNode::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::ScaledDotProductAttentionNode>(new_args, m_config);
}

void ov::intel_cpu::ScaledDotProductAttentionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionNode_validate_and_infer_types);
    auto input_num = get_input_size();
    // [B, H, L1, S]
    auto q_ps = get_input_partial_shape(0);
    // [B, H, L0, S]
    auto past_kv_ps = get_input_partial_shape(input_num - 1);

    NODE_VALIDATION_CHECK(this, m_config.output_BLHxS == false);
    NODE_VALIDATION_CHECK(this, q_ps.size() >= 3);
    if (past_kv_ps.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, q_ps.size() == past_kv_ps.size());
        for (size_t i = 0; i < q_ps.size(); i++) {
            if (i == q_ps.size() - 2)
                continue;
            NODE_VALIDATION_CHECK(this, q_ps[i].compatible(past_kv_ps[i]));
        }
        past_kv_ps[q_ps.size() - 2] += q_ps[q_ps.size() - 2];
    }
    set_output_type(0, get_input_element_type(0), q_ps);
    set_output_type(1, get_input_element_type(input_num - 1), past_kv_ps);
    set_output_type(2, get_input_element_type(input_num - 1), past_kv_ps);
}

bool ov::intel_cpu::ScaledDotProductAttentionNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("output_BLHxS", m_config.output_BLHxS);
    visitor.on_attribute("fuse_causal_attn", m_config.fuse_causal_attn);
    visitor.on_attribute("is_causal", m_config.is_causal);
    visitor.finish_structure();
    return true;
}
