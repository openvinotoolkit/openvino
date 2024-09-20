// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_sequence.hpp"

#include "itt.hpp"
#include "lstm_sequence_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/recurrent_sequence.hpp"

namespace ov {
OPENVINO_SUPPRESS_DEPRECATED_START
op::v0::LSTMSequence::LSTMSequence(const Output<Node>& X,
                                   const Output<Node>& initial_hidden_state,
                                   const Output<Node>& initial_cell_state,
                                   const Output<Node>& sequence_lengths,
                                   const Output<Node>& W,
                                   const Output<Node>& R,
                                   const Output<Node>& B,
                                   const Output<Node>& P,
                                   const std::int64_t hidden_size,
                                   const LSTMSequence::direction lstm_direction,
                                   LSTMWeightsFormat weights_format,
                                   const std::vector<float> activations_alpha,
                                   const std::vector<float> activations_beta,
                                   const std::vector<std::string> activations,
                                   const float clip_threshold,
                                   const bool input_forget)
    : RNNCellBase({X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B, P},
                  hidden_size,
                  clip_threshold,
                  activations,
                  activations_alpha,
                  activations_beta),
      m_direction(lstm_direction),
      m_input_forget(input_forget),
      m_weights_format(weights_format) {
    constructor_validate_and_infer_types();
}

op::v0::LSTMSequence::LSTMSequence(const Output<Node>& X,
                                   const Output<Node>& initial_hidden_state,
                                   const Output<Node>& initial_cell_state,
                                   const Output<Node>& sequence_lengths,
                                   const Output<Node>& W,
                                   const Output<Node>& R,
                                   const Output<Node>& B,
                                   const std::int64_t hidden_size,
                                   const LSTMSequence::direction lstm_direction,
                                   LSTMWeightsFormat weights_format,
                                   const std::vector<float>& activations_alpha,
                                   const std::vector<float>& activations_beta,
                                   const std::vector<std::string>& activations,
                                   const float clip_threshold,
                                   const bool input_forget)
    : op::v0::LSTMSequence(
          X,
          initial_hidden_state,
          initial_cell_state,
          sequence_lengths,
          W,
          R,
          B,
          Constant::create(element::f32,
                           Shape{(lstm_direction == LSTMSequence::direction::BIDIRECTIONAL ? 2UL : 1UL),
                                 3UL * static_cast<size_t>(hidden_size)},
                           std::vector<float>{0.f}),
          hidden_size,
          lstm_direction,
          weights_format,
          activations_alpha,
          activations_beta,
          activations,
          clip_threshold,
          input_forget) {}

bool op::v0::LSTMSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_LSTMSequence_visit_attributes);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);
    visitor.on_attribute("direction", m_direction);

    visitor.on_attribute("input_forget", m_input_forget);
    visitor.on_attribute("weights_format", m_weights_format);
    return true;
}

std::shared_ptr<Node> op::v0::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 8) {
        return std::make_shared<op::v0::LSTMSequence>(new_args.at(0),  // X
                                                      new_args.at(1),  // initial_hidden_state
                                                      new_args.at(2),  // initial_cell_state
                                                      new_args.at(3),  // sequence_lengths
                                                      new_args.at(4),  // W
                                                      new_args.at(5),  // R
                                                      new_args.at(6),  // B
                                                      new_args.at(7),  // P
                                                      m_hidden_size,
                                                      m_direction,
                                                      m_weights_format,
                                                      m_activations_alpha,
                                                      m_activations_beta,
                                                      m_activations,
                                                      m_clip,
                                                      m_input_forget);
    } else if (new_args.size() == 7) {
        return std::make_shared<op::v0::LSTMSequence>(new_args.at(0),  // X
                                                      new_args.at(1),  // initial_hidden_state
                                                      new_args.at(2),  // initial_cell_state
                                                      new_args.at(3),  // sequence_lengths
                                                      new_args.at(4),  // W
                                                      new_args.at(5),  // R
                                                      new_args.at(6),  // B
                                                      m_hidden_size,
                                                      m_direction,
                                                      m_weights_format,
                                                      m_activations_alpha,
                                                      m_activations_beta,
                                                      m_activations,
                                                      m_clip,
                                                      m_input_forget);
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

void op::v0::LSTMSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v0_LSTMSequence_validate_and_infer_types);
    auto result_et = element::dynamic;

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not match.");

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 6; ++i)
        set_input_is_relevant_to_shape(i);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);

    // Set output size, type and shape
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
    set_output_type(2, result_et, output_shapes[2]);
}
OPENVINO_SUPPRESS_DEPRECATED_END

bool op::v5::LSTMSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_LSTMSequence_visit_attributes);
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

std::shared_ptr<Node> op::v5::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 7) {
        return std::make_shared<op::v5::LSTMSequence>(new_args.at(0),  // X
                                                      new_args.at(1),  // initial_hidden_state
                                                      new_args.at(2),  // initial_cell_state
                                                      new_args.at(3),  // sequence_lengths
                                                      new_args.at(4),  // W
                                                      new_args.at(5),  // R
                                                      new_args.at(6),  // B
                                                      m_hidden_size,
                                                      m_direction,
                                                      m_activations_alpha,
                                                      m_activations_beta,
                                                      m_activations,
                                                      m_clip);
    } else {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

void op::v5::LSTMSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v5_LSTMSequence_validate_and_infer_types);

    auto result_et = element::dynamic;

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not match.");

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 6; ++i)
        set_input_is_relevant_to_shape(i);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto output_shapes = shape_infer(this, input_shapes);

    // Set output size, type and shape
    set_output_type(0, result_et, output_shapes[0]);
    set_output_type(1, result_et, output_shapes[1]);
    set_output_type(2, result_et, output_shapes[2]);
}
}  // namespace ov
