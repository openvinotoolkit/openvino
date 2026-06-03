// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_router_fused.hpp"

namespace ov::intel_gpu::op {

MoERouterFused::MoERouterFused(const ov::OutputVector& args, const Config& config)
    : ov::op::Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

void MoERouterFused::validate_and_infer_types() {
    const size_t expected_inputs = (m_config.routing_type == RoutingType::SIGMOID_BIAS) ? 3 : 1;
    NODE_VALIDATION_CHECK(this, get_input_size() == expected_inputs,
                          "MoERouterFused: expected ", expected_inputs,
                          " inputs, got ", get_input_size());

    const auto& input_type = get_input_element_type(0);
    const auto& input_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          input_pshape.rank().is_static() && (input_pshape.rank().get_length() == 2 || input_pshape.rank().get_length() == 3),
                          "MoERouterFused expects input of rank 2 or 3, got rank ",
                          input_pshape.rank());

    ov::PartialShape out_shape = input_pshape;
    if (input_pshape.rank().is_static()) {
        *out_shape.rbegin() = static_cast<int64_t>(m_config.top_k);
    }

    // Output 0: topk_weights [num_tokens, top_k] — same type as input
    set_output_type(0, input_type, out_shape);
    // Output 1: topk_indices [num_tokens, top_k] — i32
    set_output_type(1, ov::element::i32, out_shape);
}

std::shared_ptr<ov::Node> MoERouterFused::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MoERouterFused>(new_args, m_config);
}

bool MoERouterFused::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_expert", m_config.num_expert);
    visitor.on_attribute("top_k", m_config.top_k);
    visitor.on_attribute("routing_type", m_config.routing_type);
    return true;
}

std::ostream& operator<<(std::ostream& s, const MoERouterFused::RoutingType& type) {
    return s << as_string(type);
}

}  // namespace ov::intel_gpu::op

namespace ov {
using RoutingType = ov::intel_gpu::op::MoERouterFused::RoutingType;
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
