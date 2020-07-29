// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/gru_sequence_ie.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GRUSequenceIE::type_info;

op::GRUSequenceIE::GRUSequenceIE(const Output<Node>& X, const Output<Node>& H_t, const Output<Node>& sequence_lengths,
                                 const Output<Node>& WR, const Output<Node>& B, std::size_t hidden_size,
                                 op::RecurrentSequenceDirection direction, const std::vector<std::string>& activations,
                                 const std::vector<float>& activations_alpha, const std::vector<float>& activations_beta,
                                 float clip, bool linear_before_reset)
        : Op({X, H_t, sequence_lengths, WR, B}),
          RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta),
          m_direction(direction),
          m_linear_before_reset(linear_before_reset) {
    constructor_validate_and_infer_types();
}

void op::GRUSequenceIE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape output_shape_0{PartialShape::dynamic(4)};
    PartialShape output_shape_1{PartialShape::dynamic(3)};
    if (get_input_partial_shape(0).is_static()) {
        int64_t batch_size = get_input_partial_shape(0).get_shape()[0];
        output_shape_0 = {batch_size,
                          m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1,
                          Dimension::dynamic(),
                          static_cast<int64_t>(m_hidden_size)};

        output_shape_1 = {batch_size,
                          m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1,
                          static_cast<int64_t>(m_hidden_size)};

        const auto seq_len_in = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
                input_value(3).get_node_shared_ptr());
        if (seq_len_in) {
            auto seq_len = seq_len_in->cast_vector<size_t>()[0];
            output_shape_0[2] = seq_len;
        }
    }
    set_output_type(0, arg_type, output_shape_0);
    set_output_type(1, arg_type, output_shape_1);
}

shared_ptr<Node> op::GRUSequenceIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::GRUSequenceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                          new_args.at(4), m_hidden_size, m_direction, m_activations,
                                          m_activations_alpha, m_activations_beta, m_clip, m_linear_before_reset);
}
