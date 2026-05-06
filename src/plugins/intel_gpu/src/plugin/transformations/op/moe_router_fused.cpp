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

    auto input_type = get_input_element_type(0);
    auto input_pshape = get_input_partial_shape(0);

    ov::PartialShape out_shape;
    if (input_pshape.rank().is_static() && input_pshape.rank().get_length() >= 1) {
        out_shape = ov::PartialShape{input_pshape[0], static_cast<int64_t>(m_config.top_k)};
    } else {
        out_shape = ov::PartialShape::dynamic(2);
    }

    // Output 0: topk_weights [num_tokens, top_k] — same type as input
    set_output_type(0, input_type, out_shape);
    // Output 1: topk_indices [num_tokens, top_k] — u32
    set_output_type(1, ov::element::u32, out_shape);
}

std::shared_ptr<ov::Node> MoERouterFused::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MoERouterFused>(new_args, m_config);
}

bool MoERouterFused::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_expert", m_config.num_expert);
    visitor.on_attribute("top_k", m_config.top_k);
    int rt = static_cast<int>(m_config.routing_type);
    visitor.on_attribute("routing_type", rt);
    return true;
}

}  // namespace ov::intel_gpu::op
