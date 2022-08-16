// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interaction.hpp"
#include "../itt.hpp"

ov::intel_cpu::InteractionNode::InteractionNode(const OutputVector& args) :
    Op(args) {
    validate_and_infer_types();
}

ov::intel_cpu::InteractionNode::InteractionNode(const NodeVector& args) :
    InteractionNode(as_output_vector(args)) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::InteractionNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(InteractionNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::InteractionNode>(new_args);
    throw ngraph::ngraph_error("Unsupported number of arguments for FullyConnected operation");
}

void ov::intel_cpu::InteractionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(InteractionNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto& dense_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
        dense_pshape.rank().is_static() &&
        dense_pshape.rank() == 2 &&
        dense_pshape.is_static(),
        "feature shape must be static");
    const auto batch = dense_pshape[0].get_length();
    const auto feature = dense_pshape[1].get_length();
    for (size_t i = 1; i < input_size; i++) {
        const auto& sparse_pshape = get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(this,
            sparse_pshape.rank().is_static() &&
            sparse_pshape.rank() == 2,
            "sparse shape must be static");
        NODE_VALIDATION_CHECK(this,
            batch == sparse_pshape[0].get_length() &&
            feature == sparse_pshape[1].get_length(),
            "all shape must be same");
    }
    const auto batch_size = dense_pshape[0];
    const auto feature_size = dense_pshape[1];
    int64_t output_feature_size = input_size * (input_size - 1) / 2 + feature_size.get_length();
    auto output_type = m_output_type == ngraph::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, PartialShape{batch_size, output_feature_size});
    return;
}

bool ov::intel_cpu::InteractionNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(InteractionNode_visit_attributes);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
