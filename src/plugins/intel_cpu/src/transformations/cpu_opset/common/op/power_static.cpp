// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "power_static.hpp"

#include "transformations/itt.hpp"

ov::intel_cpu::PowerStaticNode::PowerStaticNode(const ov::Output<Node>& data,
                                                const float& power,
                                                const float& scale,
                                                const float& shift,
                                                const ov::element::Type output_type)
    : Op({data}),
      scale(scale),
      power(power),
      shift(shift),
      m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::PowerStaticNode::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(PowerStaticNode_clone_with_new_inputs);
    if (new_args.size() != 1) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }

    return std::make_shared<ov::intel_cpu::PowerStaticNode>(new_args.at(0),
                                                            this->power,
                                                            this->scale,
                                                            this->shift,
                                                            this->m_output_type);
}

void ov::intel_cpu::PowerStaticNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(PowerStaticNode_validate_and_infer_types);
    set_output_type(0,
                    m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type,
                    get_input_partial_shape(0));
}

bool ov::intel_cpu::PowerStaticNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(PowerStaticNode_visit_attributes);
    visitor.on_attribute("scale", scale);
    visitor.on_attribute("power", power);
    visitor.on_attribute("shift", shift);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
