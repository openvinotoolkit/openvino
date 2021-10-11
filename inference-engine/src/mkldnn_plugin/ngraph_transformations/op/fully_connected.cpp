// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

constexpr ngraph::NodeTypeInfo MKLDNNPlugin::FullyConnectedNode::type_info;

MKLDNNPlugin::FullyConnectedNode::FullyConnectedNode(const ngraph::Output<Node>& A,
                                                     const ngraph::Output<Node>& B,
                                                     const ngraph::Rank& output_rank,
                                                     const ngraph::element::Type output_type)
    : Op({A, B}), m_output_rank(output_rank), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

MKLDNNPlugin::FullyConnectedNode::FullyConnectedNode(const ngraph::Output<Node>& A,
                                                     const ngraph::Output<Node>& B,
                                                     const ngraph::Output<Node>& C,
                                                     const ngraph::Rank& output_rank,
                                                     const ngraph::element::Type output_type)
    : Op({A, B, C}), m_output_rank(output_rank), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::FullyConnectedNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<MKLDNNPlugin::FullyConnectedNode>(new_args.at(0), new_args.at(1), m_output_rank, m_output_type);
    } else if (new_args.size() == 3) {
        return std::make_shared<MKLDNNPlugin::FullyConnectedNode>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_rank, m_output_type);
    }

    throw ngraph::ngraph_error("Unsupported number of arguments for FullyConnected operation");
}

void MKLDNNPlugin::FullyConnectedNode::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 2 || input_size == 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 2 or 3.");

    const auto output_size = get_output_size();
    NODE_VALIDATION_CHECK(this,
        output_size == 1,
        "Number of outputs is incorrect. Current value is: ",
        output_size,
        ", expected: 1.");

    // weights_shape: [O, I]
    const auto weights_pshape = get_input_partial_shape(1);
    const auto weights_rank = weights_pshape.rank();
    NODE_VALIDATION_CHECK(this,
        weights_rank.is_static() && weights_rank.get_length() == 2,
        "Weights have incorrect rank. Current value is: ",
        weights_rank,
        ", expected: 2.");

    const auto o_channels = weights_pshape[0];
    if (input_size == 3) {
        const auto bias_shape = get_input_partial_shape(2);
        const auto expected_bias_shape = ngraph::PartialShape{ o_channels };
        NODE_VALIDATION_CHECK(this,
            bias_shape == expected_bias_shape,
            "Bias shape is incorrect. Current value is: ",
            bias_shape,
            ", expected: ",
            expected_bias_shape,
            ".");
    }

    // activations_shape: [..., I]
    const auto activations_pshape = get_input_partial_shape(0);
    auto output_shape = activations_pshape;

    if (output_shape.rank().is_static()) {
        if (m_output_rank.is_static()) {
            while (output_shape.rank().get_length() < m_output_rank.get_length()) {
                output_shape.insert(output_shape.begin(), 1);
            }
        }

        const auto i_channels = weights_pshape[1];
        if (activations_pshape[activations_pshape.size() - 1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                activations_pshape[activations_pshape.size() - 1] == i_channels,
                "Activations pshape has incorrect input_channels value. Current value is: ",
                activations_pshape[activations_pshape.size() - 1],
                ", expected: ",
                i_channels,
                ".");
        }
        output_shape[output_shape.size() - 1] = o_channels;
    }

    auto output_type = m_output_type == ngraph::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, output_shape);
}

bool MKLDNNPlugin::FullyConnectedNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    if (m_output_rank.is_static()) {
        std::int64_t value = m_output_rank.get_length();
        visitor.on_attribute("out-rank", value);
    }
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
