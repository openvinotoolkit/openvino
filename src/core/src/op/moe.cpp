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
    // TODO: Add inputs validation

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool MOE::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_MOE_visit_attributes);

    visitor.on_attribute("expert_type", m_config.expert_type);
    visitor.on_attribute("expert_alpha", m_config.expert_alpha);
    visitor.on_attribute("expert_beta", m_config.expert_beta);

    return true;
}

}  // namespace v16
}  // namespace op

std::ostream& operator<<(std::ostream& s, const ov::op::v16::MOE::Expert_type& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<ov::op::v16::MOE::Expert_type>& EnumNames<ov::op::v16::MOE::Expert_type>::get() {
    static auto enum_names = EnumNames<ov::op::v16::MOE::Expert_type>(
        "ov::op::v16::MOE::Expert_type",
        {
            {"gemm2_bias_swiglu_clamp", ov::op::v16::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP},
            {"gemm2_bias_gelu", ov::op::v16::MOE::Expert_type::GEMM3_SWIGLU},
        });
    return enum_names;
}
}  // namespace ov
