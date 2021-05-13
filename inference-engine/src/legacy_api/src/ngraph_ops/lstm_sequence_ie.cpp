// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/lstm_sequence_ie.hpp"
#include "ngraph/op/util/recurrent_sequence.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::LSTMSequenceIE, "LSTMSequenceIE", 5);

op::LSTMSequenceIE::LSTMSequenceIE(const Output<Node> &X,
                                   const Output<Node> &H_t,
                                   const Output<Node> &C_t,
                                   const Output<Node> &seq_lengths,
                                   const Output<Node> &WR,
                                   const Output<Node> &B,
                                   std::size_t hidden_size,
                                   ngraph::op::RecurrentSequenceDirection direction,
                                   const std::vector<std::string> &activations,
                                   const std::vector<float> &activations_alpha,
                                   const std::vector<float> &activations_beta,
                                   float clip,
                                   int64_t seq_axis)
        : RNNCellBase({X, H_t, C_t, seq_lengths, WR, B}, hidden_size, clip, activations, activations_alpha, activations_beta),
          m_direction(direction),
          m_seq_axis(seq_axis) {
    constructor_validate_and_infer_types();
}

void op::LSTMSequenceIE::validate_and_infer_types() {
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
            set_output_type(2, get_input_element_type(0), PartialShape::dynamic());
            return;
        }
    }
    // rank validation
    auto x_pshape = get_input_partial_shape(0);
    auto h_state_pshape = get_input_partial_shape(1);
    auto c_state_pshape = get_input_partial_shape(2);
    auto seq_lengths_pshape = get_input_partial_shape(3);
    auto wr_pshape = get_input_partial_shape(4);
    auto b_pshape = get_input_partial_shape(5);

    std::vector<ngraph::PartialShape> pshapes = {x_pshape, h_state_pshape, c_state_pshape,
                                                 seq_lengths_pshape, wr_pshape, b_pshape};
    std::vector<std::string> in_names = {"X", "H", "C", "seq_lengths", "WR", "B"};
    // num_direction dimension should be squeezed, we don't support bidirectional case
    std::vector<size_t> ranks = {3, 2, 2, 1, 2, 1};
    for (size_t i = 0; i < pshapes.size(); ++i) {
        NGRAPH_CHECK((pshapes[i].rank().get_length() == static_cast<int64_t>(ranks[i])),
                     "LSTMSequenceIE ",
                     in_names[i],
                     " input rank is not correct.");
    }

    element::Type arg_type = get_input_element_type(0);
    PartialShape output_shape_0{PartialShape::dynamic(3)};
    PartialShape output_shape_1{PartialShape::dynamic(2)};
    if (get_input_partial_shape(0).is_static()) {
        size_t batch_size = get_input_partial_shape(0).get_shape()[1 - m_seq_axis];
        size_t seq_length = get_input_partial_shape(0).get_shape()[m_seq_axis];
        if (m_seq_axis == 1)
            output_shape_0 = Shape{batch_size, seq_length, m_hidden_size};
        else
            output_shape_0 = Shape{seq_length, batch_size, m_hidden_size};
        output_shape_1 = Shape{batch_size, m_hidden_size};
    }
    set_output_type(0, arg_type, output_shape_0);
    set_output_type(1, arg_type, output_shape_1);
    set_output_type(2, arg_type, output_shape_1);
}

bool ngraph::op::LSTMSequenceIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("direction", m_direction);
    visitor.on_attribute("axis", m_seq_axis);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

shared_ptr<Node> op::LSTMSequenceIE::clone_with_new_inputs(const OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::LSTMSequenceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
            new_args.at(4), new_args.at(5), m_hidden_size, m_direction, m_activations, m_activations_alpha, m_activations_beta,
            m_clip, m_seq_axis);
}
