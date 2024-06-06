// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_qkv_proj.hpp"

#include "transformations/itt.hpp"
namespace ov {
namespace intel_cpu {

bool LLMQKVProjNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(LLMQKVProjNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.finish_structure();
    return true;
}

void LLMQKVProjNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(LLMQKVProjNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto& ishape = get_input_partial_shape(0);
    const auto& itype = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this, ishape.rank().is_static() && ishape.rank() == 3, "feature shape rank must be 3");
    const auto batch = ishape[0];
    const auto length = ishape[1];
    const auto feature = ishape[2];
    NODE_VALIDATION_CHECK(this, feature.is_static() && feature.get_length() == m_config.hidden_size);

    NODE_VALIDATION_CHECK(this, input_size == 4);
    set_output_size(3);
    set_output_type(0, itype, ishape);
    set_output_type(1, itype, ishape);
    set_output_type(2, itype, ishape);
}

std::shared_ptr<Node> LLMQKVProjNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LLMQKVProjNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LLMQKVProjNode>(new_args, m_config);
}
}  // namespace intel_cpu
}  // namespace ov
