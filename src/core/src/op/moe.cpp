// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/moe.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace internal {

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
    OV_OP_SCOPE(internal_MOE_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<MOE>(new_args, m_config);
}

void MOE::validate_and_infer_types() {
    OV_OP_SCOPE(internal_MOE_validate_and_infer_types);
    if (m_config.expert_type == Expert_type::GEMM2_BIAS_SWIGLU_CLAMP) {
        NODE_VALIDATION_CHECK(this,
                              m_config.activation_type == Activation_type::SWIGLU,
                              "GEMM2_BIAS_SWIGLU_CLAMP expert type only supports SWIGLU activation");
    }

    // Check that routing_weights and router_topk_output_indices have the same shape
    const auto& routing_weights_shape = get_input_partial_shape(1);
    const auto& topk_indices_shape = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this,
                          routing_weights_shape == topk_indices_shape,
                          "routing_weights shape ",
                          routing_weights_shape,
                          " must be compatible with router_topk_output_indices shape ",
                          topk_indices_shape);

    // Check that all expert weight inputs (index >= 3) have the same first dimension (num_experts).
    const size_t base_inputs_count = (m_config.expert_type == Expert_type::GEMM3_SWIGLU) ? 6 : 7;
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(3).is_static(), "Weights must have static shape.");
    const auto& num_experts = get_input_shape(3)[0];
    for (size_t i = 4; i < base_inputs_count; i++) {
        const auto& ps = get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(this, ps.is_static(), "Weights must have static shape.");
        // Note: dynamic element type means empty zero point input
        if (get_input_element_type(i) != ov::element::dynamic) {
            NODE_VALIDATION_CHECK(this,
                                  num_experts == get_input_shape(i)[0],
                                  "All weight inputs must have the same first dimension (num_experts).");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOE::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_MOE_visit_attributes);
    visitor.on_attribute("expert_type", m_config.expert_type);
    visitor.on_attribute("expert_alpha", m_config.expert_alpha);
    visitor.on_attribute("expert_beta", m_config.expert_beta);
    visitor.on_attribute("gate_idx", m_config.gate_idx);
    visitor.on_attribute("activation_type", m_config.activation_type);

    return true;
}

}  // namespace internal
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::internal::MOE::Expert_type& type) {
    return s << as_string(type);
}

std::ostream& operator<<(std::ostream& s, const ov::op::internal::MOE::Activation_type& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<ov::op::internal::MOE::Expert_type>& EnumNames<ov::op::internal::MOE::Expert_type>::get() {
    static auto enum_names = EnumNames<ov::op::internal::MOE::Expert_type>(
        "ov::op::internal::MOE::Expert_type",
        {
            {"gemm2_bias_swiglu_clamp", ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP},
            {"gemm3_swiglu", ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU},
        });
    return enum_names;
}

template <>
OPENVINO_API EnumNames<ov::op::internal::MOE::Activation_type>&
EnumNames<ov::op::internal::MOE::Activation_type>::get() {
    static auto enum_names = EnumNames<ov::op::internal::MOE::Activation_type>(
        "ov::op::internal::MOE::Activation_type",
        {
            {"swiglu", ov::op::internal::MOE::Activation_type::SWIGLU},
            {"geglu_tanh", ov::op::internal::MOE::Activation_type::GEGLU_TANH},
            {"geglu_erf", ov::op::internal::MOE::Activation_type::GEGLU_ERF},
        });
    return enum_names;
}
}  // namespace ov
