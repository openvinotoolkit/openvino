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
    visitor.start_structure("config");

    visitor.on_attribute("topk", m_config.topk);
    visitor.on_attribute("expert_num", m_config.expert_num);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("intermediate_size", m_config.intermediate_size);
    visitor.on_attribute("group_size", m_config.group_size);
    visitor.on_attribute("weight_type", m_config.weight_type);
    visitor.on_attribute("scale_type", m_config.scale_type);
    visitor.on_attribute("zp_type", m_config.zp_type);
    visitor.finish_structure();
    
    return true;
}

std::shared_ptr<ov::op::v0::Constant> MOE::get_expert_const(size_t expert_idx, size_t weight_type, size_t const_type) const {
    OPENVINO_ASSERT(expert_idx < m_config.expert_num, "Expert index out of range");
    OPENVINO_ASSERT(weight_type < 3, "Weight type must be 0 (gate), 1 (up), or 2 (down)");
    OPENVINO_ASSERT(const_type < 3, "Const type must be 0 (weight), 1 (scale), or 2 (zp)");

    // Calculate input index based on expert and weight/const type
    // Input layout: [hidden_states, router_logits, expert0_gate_weight, expert0_gate_scale?, expert0_gate_zp?, 
    //                expert0_up_weight, expert0_up_scale?, expert0_up_zp?, expert0_down_weight, expert0_down_scale?, expert0_down_zp?, ...]
    
    size_t base_idx = 2; // Start after hidden_states and router_logits
    
    // For now, assume simple layout: weight, scale?, zp? for each of gate, up, down
    size_t constants_per_weight_type = 1; // Just weights for now, will need to extend for scales/zps
    if (m_config.scale_type != ov::element::dynamic) constants_per_weight_type++;
    if (m_config.zp_type != ov::element::dynamic) constants_per_weight_type++;
    
    size_t constants_per_expert = 3 * constants_per_weight_type; // 3 weight types * constants per type
    
    size_t expert_base = base_idx + expert_idx * constants_per_expert;
    size_t weight_base = expert_base + weight_type * constants_per_weight_type;
    size_t input_idx = weight_base + const_type;
    
    if (input_idx >= get_input_size()) {
        return nullptr; // Constant not provided (e.g., scale or zp for non-quantized weights)
    }
    
    auto input_node = get_input_node_shared_ptr(input_idx);
    return ov::as_type_ptr<ov::op::v0::Constant>(input_node);
}

}  // namespace v16
}  // namespace op
}  // namespace ov
