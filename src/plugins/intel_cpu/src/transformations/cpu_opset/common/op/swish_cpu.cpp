// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swish_cpu.hpp"

#include "transformations/itt.hpp"

ov::intel_cpu::SwishNode::SwishNode(const ov::Output<ov::Node>& input, const float alpha)
    : Op({input}),
      m_alpha(alpha) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::SwishNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(SwishNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::SwishNode>(new_args.at(0), m_alpha);
}

bool ov::intel_cpu::SwishNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(SwishNode_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void ov::intel_cpu::SwishNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(SwishNode_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

float ov::intel_cpu::SwishNode::get_alpha() const {
    return m_alpha;
}
