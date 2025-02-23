// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interaction.hpp"

#include "transformations/itt.hpp"

ov::intel_cpu::InteractionNode::InteractionNode(const OutputVector& args) : Op(args) {
    validate_and_infer_types();
}

ov::intel_cpu::InteractionNode::InteractionNode(const NodeVector& args) : InteractionNode(as_output_vector(args)) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::InteractionNode::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(InteractionNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::InteractionNode>(new_args);
}

void ov::intel_cpu::InteractionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(InteractionNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    const auto& dense_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          dense_pshape.rank().is_static() && dense_pshape.rank() == 2,
                          "feature shape rank must be 2");
    const auto batch = dense_pshape[0];
    const auto feature = dense_pshape[1];
    for (size_t i = 1; i < input_size; i++) {
        const auto& sparse_pshape = get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(this,
                              sparse_pshape.rank().is_static() && sparse_pshape.rank() == 2,
                              "sparse shape must be static");
        NODE_VALIDATION_CHECK(this,
                              batch.compatible(sparse_pshape[0]) && feature.compatible(sparse_pshape[1]),
                              "dense & sparse shape must be compatible");
    }

    Dimension output_feature_size;
    // only set output when feature is static
    if (feature.is_static()) {
        output_feature_size = input_size * (input_size - 1) / 2 + feature.get_length();
    }
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    m_output_type = output_type;
    PartialShape output_shape = ov::PartialShape::dynamic(2);
    output_shape[0] = batch;
    output_shape[1] = output_feature_size;
    set_output_type(0, output_type, output_shape);
    return;
}

bool ov::intel_cpu::InteractionNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(InteractionNode_visit_attributes);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
