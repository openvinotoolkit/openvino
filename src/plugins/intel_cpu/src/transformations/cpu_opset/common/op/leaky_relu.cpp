// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "leaky_relu.hpp"

#include "transformations/itt.hpp"

ov::intel_cpu::LeakyReluNode::LeakyReluNode(const ov::Output<ov::Node>& data,
                                            const float& negative_slope,
                                            const ov::element::Type output_type)
    : Op({data}),
      m_negative_slope(negative_slope),
      m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::LeakyReluNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LeakyReluNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::LeakyReluNode>(new_args.at(0), m_negative_slope, m_output_type);
}

void ov::intel_cpu::LeakyReluNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(LeakyReluNode_validate_and_infer_types);
    set_output_type(0,
                    m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type,
                    get_input_partial_shape(0));
}

bool ov::intel_cpu::LeakyReluNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(LeakyReluNode_visit_attributes);
    visitor.on_attribute("negative_slope", m_negative_slope);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
