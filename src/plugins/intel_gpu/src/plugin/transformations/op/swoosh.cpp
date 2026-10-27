// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/swoosh.hpp"

#include "openvino/core/attribute_visitor.hpp"

namespace ov::intel_gpu::op {

Swoosh::Swoosh(const ov::Output<Node>& data, float offset, float bias)
    : Op({data}), m_offset(offset), m_bias(bias) {
    validate_and_infer_types();
}

void Swoosh::validate_and_infer_types() {
    const bool is_swoosh_l = m_offset == swoosh_constants::swoosh_l_offset && m_bias == swoosh_constants::swoosh_l_bias;
    const bool is_swoosh_r = m_offset == swoosh_constants::swoosh_r_offset && m_bias == swoosh_constants::swoosh_r_bias;
    OPENVINO_ASSERT(is_swoosh_l || is_swoosh_r, "Unsupported Swoosh constants");
    OPENVINO_ASSERT(get_input_element_type(0).is_real(), "Swoosh requires a floating-point input");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool Swoosh::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("offset", m_offset);
    visitor.on_attribute("bias", m_bias);
    return true;
}

std::shared_ptr<ov::Node> Swoosh::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Swoosh>(new_args.at(0), m_offset, m_bias);
}

}  // namespace ov::intel_gpu::op
