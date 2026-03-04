// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_compressed.hpp"

namespace ov::op::internal {

MOECompressed::MOECompressed(const OutputVector& args, const Config& config) : MOE(args, config), m_config(config) {
    constructor_validate_and_infer_types();
}

const MOECompressed::Config& MOECompressed::get_config() const {
    return m_config;
}

void MOECompressed::set_config(const Config& config) {
    m_config = config;
}

std::shared_ptr<ov::Node> MOECompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOECompressed>(new_args, m_config);
}

void MOECompressed::validate_and_infer_types() {
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;

    set_output_type(0, output_type, get_input_partial_shape(0));
}

bool MOECompressed::visit_attributes(ov::AttributeVisitor& visitor) {
    MOE::visit_attributes(visitor);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("inter_size", m_config.inter_size);
    visitor.on_attribute("num_expert", m_config.num_expert);
    visitor.on_attribute("num_shared_expert", m_config.num_shared_expert);
    visitor.on_attribute("top_k", m_config.top_k);
    visitor.on_attribute("group_size", m_config.group_size);
    visitor.on_attribute("has_batch_dim", m_config.has_batch_dim);
    visitor.on_attribute("has_zp", m_config.has_zp);
    visitor.on_attribute("out_type", m_config.out_type);
    visitor.on_attribute("routing_type", m_config.routing_type);
    return true;
}

std::ostream& operator<<(std::ostream& s, const MOECompressed::RoutingType& type) {
    return s << as_string(type);
}

}  // namespace ov::op::internal

namespace ov {
using RoutingType = ov::op::internal::MOECompressed::RoutingType;
template <>
EnumNames<RoutingType>& EnumNames<RoutingType>::get() {
    static auto enum_names = EnumNames<RoutingType>("MOECompressed::RoutingType",
                                                    {
                                                        {"softmax", RoutingType::SOFTMAX},
                                                        {"sigmoid_bias", RoutingType::SIGMOID_BIAS},
                                                    });
    return enum_names;
}

}  // namespace ov
