// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/moe.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace v16 {

MOE::MOE(const OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

const MOE::Config& MOE::get_config() const {
    return m_config;
}

void MOE::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOE::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v16_MOE_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<MOE>(new_args, m_config);
}

void MOE::validate_and_infer_types() {
    OV_OP_SCOPE(v16_MOE_validate_and_infer_types);
    // At minimum we need 2 inputs: hidden_states and router_logits
    OPENVINO_ASSERT(get_input_size() >= 2, "MOE must have at least 2 inputs whereas it has ", get_input_size());

    // For now, just do basic validation. The input layout validation can be more flexible
    // to allow incremental building during pattern matching
    // Expected inputs:
    // 0: hidden_states
    // 1: router_logits
    // 2+: expert constants (flexible layout during construction)

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOE::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_MOE_visit_attributes);

    // visitor.on_attribute("expert_type", m_config.expert_type);
    // TODO: Add adapter

    visitor.on_attribute("expert_alpha", m_config.expert_alpha);
    visitor.on_attribute("expert_beta", m_config.expert_beta);

    return true;
}

}  // namespace v16
}  // namespace op
}  // namespace ov
