// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.hpp"

#include "transformations/itt.hpp"
namespace ov {

template <>
EnumNames<ov::intel_cpu::LLMMLPNode::ACT_FN>& EnumNames<ov::intel_cpu::LLMMLPNode::ACT_FN>::get() {
    static auto enum_names = EnumNames<ov::intel_cpu::LLMMLPNode::ACT_FN>(
        "op::intel_cpu::LLMMLPNode::ACT_FN",
        {{"GELU", ov::intel_cpu::LLMMLPNode::ACT_FN::GELU}, {"SILU", ov::intel_cpu::LLMMLPNode::ACT_FN::SILU}});
    return enum_names;
}

std::ostream& operator<<(std::ostream& os, const ov::intel_cpu::LLMMLPNode::ACT_FN& type) {
    return os << as_string(type);
}

namespace intel_cpu {

bool LLMMLPNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(LLMMLPNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("act", m_config.act);
    visitor.on_attribute("gate_up_quantized", m_config.gate_up_quantized);
    visitor.on_attribute("down_quantized", m_config.down_quantized);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("up_size", m_config.up_size);
    visitor.on_attribute("gate_up_combined", m_config.gate_up_combined);
    visitor.finish_structure();
    return true;
}

void LLMMLPNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(LLMMLPNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    size_t expect_input_size = 4;
    if (m_config.gate_up_quantized) expect_input_size += 2;
    if (m_config.down_quantized) expect_input_size += 1;
    NODE_VALIDATION_CHECK(this, input_size == expect_input_size);

    const auto& ishape = get_input_partial_shape(0);
    const auto& itype = get_input_element_type(0);

    const auto& w_down_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this, ishape.rank().is_static() && ishape.rank() == 3, "feature shape rank must be 3");
    const auto batch = ishape[0];
    const auto length = ishape[1];
    const auto feature = ishape[2];
    NODE_VALIDATION_CHECK(this, feature.is_static());
    NODE_VALIDATION_CHECK(this, itype.is_real(), "feature data type must be real");

    auto oshape = ishape;
    oshape[oshape.size() - 1] = w_down_shape[0];
    set_output_type(0, itype, oshape);
}

std::shared_ptr<Node> LLMMLPNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LLMMLPNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LLMMLPNode>(new_args, m_config);
}
}  // namespace intel_cpu
}  // namespace ov
