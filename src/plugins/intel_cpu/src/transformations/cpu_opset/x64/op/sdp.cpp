// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdp.hpp"

#include <algorithm>

#include "transformations/itt.hpp"
#include "openvino/op/util/variable.hpp"

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
    auto input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this, m_config.qkv_merged == true);
    NODE_VALIDATION_CHECK(this, m_config.past_key_var);
    NODE_VALIDATION_CHECK(this, m_config.past_value_var);
    NODE_VALIDATION_CHECK(this, m_config.output_BLHxS);

    auto batch = input_pshape[0];
    auto seq_len = input_pshape[1];
    auto h3s_len = input_pshape[2];

    NODE_VALIDATION_CHECK(this, h3s_len == m_config.num_heads * 3 * m_config.num_states_per_head);

    input_pshape[2] = m_config.num_heads * m_config.num_states_per_head;
    set_output_type(0, get_input_element_type(0), input_pshape);
}

bool ov::intel_cpu::ScaledDotProductAttentionNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("qkv_merged", m_config.qkv_merged);
    visitor.on_attribute("input_trans0213", m_config.input_trans0213);
    visitor.on_attribute("cos_is_raw3d", m_config.cos_is_raw3d);
    visitor.on_attribute("sin_is_raw3d", m_config.sin_is_raw3d);
    visitor.on_attribute("output_BLHxS", m_config.output_BLHxS);
    visitor.on_attribute("ndims", m_config.rope_ndims);
    visitor.on_attribute("gather_position_arg_id", m_config.gather_position_arg_id);
    visitor.on_attribute("past_key_var", m_config.past_key_var);
    visitor.on_attribute("past_value_var", m_config.past_value_var);
    visitor.finish_structure();
    return true;
}
