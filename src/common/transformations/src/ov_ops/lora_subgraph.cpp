// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/lora_subgraph.hpp"

namespace ov {
namespace op {
namespace internal {

LoraSubgraph::LoraSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body) : SubGraphOp(args) {
    OPENVINO_ASSERT(args.size() == 5, "LoraSubgraph must have 5 inputs");
    SubGraphOp::set_function(body);
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LoraSubgraph::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<LoraSubgraph>(new_args, get_function()->clone());
}

void LoraSubgraph::validate_and_infer_types() {
    const auto& body = get_function();
    OPENVINO_ASSERT(body, "LoraSubgraph must have initialized body");
    validate_and_infer_type_body(body, m_input_descriptions[0]);
    for (size_t i = 0; i < get_output_size(); ++i)
        set_output_type(i, body->get_output_element_type(i), body->get_output_partial_shape(i));
}

}  // namespace internal
}  // namespace op
}  // namespace ov
