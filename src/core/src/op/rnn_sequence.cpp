// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rnn_sequence.hpp"

#include <memory>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/recurrent_sequence.hpp"
#include "rnn_sequence_shape_inference.hpp"

namespace ov {
op::v5::RNNSequence::RNNSequence() : m_direction(op::RecurrentSequenceDirection::FORWARD) {}

op::v5::RNNSequence::RNNSequence(const Output<Node>& X,
                                 const Output<Node>& H_t,
                                 const Output<Node>& sequence_lengths,
                                 const Output<Node>& W,
                                 const Output<Node>& R,
                                 const Output<Node>& B,
                                 std::size_t hidden_size,
                                 op::RecurrentSequenceDirection direction,
                                 const std::vector<std::string>& activations,
                                 const std::vector<float>& activations_alpha,
                                 const std::vector<float>& activations_beta,
                                 float clip)
    : RNNCellBase({X, H_t, sequence_lengths, W, R, B},
                  hidden_size,
                  clip,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_direction(direction) {
    constructor_validate_and_infer_types();
}

void op::v5::RNNSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v5_RNNSequence_validate_and_infer_types);

    auto result_et = element::dynamic;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(3)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)),
                          "Element types for X, initial_hidden_state, W, R and B inputs do not "
                          "match.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 5; ++i)
        set_input_is_relevant_to_shape(i);

    // Set output size, type and shape
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
}

bool op::v5::RNNSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_RNNSequence_visit_attributes);
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

std::shared_ptr<Node> op::v5::RNNSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_RNNSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v5::RNNSequence>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 new_args.at(5),
                                                 m_hidden_size,
                                                 m_direction,
                                                 m_activations,
                                                 m_activations_alpha,
                                                 m_activations_beta,
                                                 m_clip);
}
}  // namespace ov
