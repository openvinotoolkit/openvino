// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::FullyConnectedNode::FullyConnectedNode(const ngraph::Output<Node>& A,
                                                     const ngraph::Output<Node>& B,
                                                     const ngraph::Rank& output_rank,
                                                     const ngraph::element::Type output_type)
    : Op({A, B}), m_output_rank(output_rank), m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::FullyConnectedNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FullyConnectedNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<ov::intel_cpu::FullyConnectedNode>(new_args.at(0), new_args.at(1), m_output_rank, m_output_type);
}

void ov::intel_cpu::FullyConnectedNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(FullyConnectedNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 2,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 2.");

    // Weights shape: [O, I1, ..., Im];
    // O - output channels dimensions, Ik - input channels dimensions
    const auto weights_pshape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
        weights_pshape.is_static(),
        "Weights pshape must be static");
    const auto weights_shape = weights_pshape.to_shape();

    NODE_VALIDATION_CHECK(this,
        weights_pshape.size() > 0,
        "Weights rank must be greater than 0");

    const auto o_channels = weights_pshape[0];

    // Activations shape: [B1, ..., Bn, I1, ..., Im];
    // Bi - batch dimensions, Ik - input channels dimensions
    const auto activations_pshape = get_input_partial_shape(0);

    // Result shape: [B1, ..., Bn, O]
    ngraph::PartialShape output_pshape;
    if (activations_pshape.rank().is_static()) {
        size_t output_channels_dimensions_count = weights_shape.size() - 1;
        for (size_t i = 0; i < activations_pshape.size() - output_channels_dimensions_count; ++i) {
            output_pshape.push_back(activations_pshape[i]);
        }
        output_pshape.push_back(o_channels);

        NODE_VALIDATION_CHECK(this,
            m_output_rank.is_static(),
            "Output rank must be static if activations rank is static.");

        while (output_pshape.rank().get_length() < m_output_rank.get_length()) {
            output_pshape.insert(output_pshape.begin(), 1);
        }
    } else {
        output_pshape = ngraph::PartialShape::dynamic();
    }

    auto output_type = m_output_type == ngraph::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, output_pshape);
}

bool ov::intel_cpu::FullyConnectedNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(FullyConnectedNode_visit_attributes);
    visitor.on_attribute("out-rank", m_output_rank);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
