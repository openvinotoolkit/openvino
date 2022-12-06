// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/lstm_sequence.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/util/recurrent_sequence.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset4.hpp"

using namespace ngraph;
using namespace std;

BWDCMP_RTTI_DEFINITION(op::v0::LSTMSequence);
BWDCMP_RTTI_DEFINITION(op::v5::LSTMSequence);

op::v0::LSTMSequence::LSTMSequence()
    : Op(),
      m_activations_alpha(),
      m_activations_beta(),
      m_activations(),
      m_clip_threshold(),
      m_direction(),
      m_hidden_size(),
      m_input_forget(),
      m_weights_format() {}

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
    : Op({X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B, P}),
      m_activations_alpha(activations_alpha),
      m_activations_beta(activations_beta),
      m_activations(activations),
      m_clip_threshold(clip_threshold),
      m_direction(lstm_direction),
      m_hidden_size(hidden_size),
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
    visitor.on_attribute("clip", m_clip_threshold);
    visitor.on_attribute("direction", m_direction);

    visitor.on_attribute("input_forget", m_input_forget);
    visitor.on_attribute("weights_format", m_weights_format);
    return true;
}

shared_ptr<Node> op::v0::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 8) {
        return make_shared<op::v0::LSTMSequence>(new_args.at(0),  // X
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
                                                 m_clip_threshold,
                                                 m_input_forget);
    } else if (new_args.size() == 7) {
        return make_shared<op::v0::LSTMSequence>(new_args.at(0),  // X
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
                                                 m_clip_threshold,
                                                 m_input_forget);
    } else {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

void op::v0::LSTMSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v0_LSTMSequence_validate_and_infer_types);
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(2, get_input_element_type(0), ov::PartialShape::dynamic());
            return;
        }
    }
    std::vector<ov::PartialShape> input_param{};

    auto lstm_seq_gates_count = 4;
    auto lstm_seq_peepholes_count = 3;
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto merged_num_directions = Dimension::dynamic();
    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(this, get_input_size() > 0, "The number of inputs of the LSTMSequence op cannot be zero.");
    // Copy all inputs without peephole and initial_cell_state information for further validation
    for (size_t i = 0; i < get_input_size() - 1; i++) {
        // exclude initial_cell_state from the loop
        if (i != 2) {
            input_param.push_back(get_input_partial_shape(i));
        }
    }

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& sl_pshape = get_input_partial_shape(3);
    const auto& w_pshape = get_input_partial_shape(4);
    const auto& r_pshape = get_input_partial_shape(5);
    const auto& b_pshape = get_input_partial_shape(6);
    const auto& p_pshape = get_input_partial_shape(7);

    ngraph::op::util::validate_seq_input_rank_dimension(input_param);

    // Validate rank and dimension for initial_cell_state input
    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().is_static()),
                          "LSTMSequence input tensor initial_cell_state shall have static rank.");

    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().get_length() == 3),
                          "LSTMSequence input tensor initial_cell_state shall have dimension 3D.");

    // Validate rank and dimension for P input
    NODE_VALIDATION_CHECK(this, (p_pshape.rank().is_static()), "LSTMSequence input tensor P shall have static rank.");

    NODE_VALIDATION_CHECK(this,
                          (p_pshape.rank().get_length() == 2),
                          "LSTMSequence input tensor P shall have dimension 2D.");

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not "
                          "match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                          "Parameter batch_size not matched in LSTMSequence.");

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[2]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                          "Parameter hidden_size not matched LSTMSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, ct_pshape[1]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
                          "Parameter num_directions not matched in LSTMSequence.");

    auto valid_num_directions = 0;
    if (m_direction == op::RecurrentSequenceDirection::FORWARD ||
        m_direction == op::RecurrentSequenceDirection::REVERSE) {
        valid_num_directions = 1;
    } else if (m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        valid_num_directions = 2;
    } else {
        // Guard for potential future extension of RecurrentSequenceDirection enum
        NODE_VALIDATION_CHECK(this, false, "Parameter direction must be FORWARD or REVERSE or BIDIRECTIONAL.");
    }

    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_num_directions, merged_num_directions, valid_num_directions),
                          "Parameter 'num_directions' doesn't match with direction '",
                          m_direction,
                          "' in LSTMSequence. Expected ",
                          valid_num_directions,
                          ", actual ",
                          merged_num_directions);

    // Validate hidden_size value for W, R, B and P inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  w_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in P input. Current value is: ",
                                  w_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (r_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  r_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in R input. Current value is: ",
                                  r_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (b_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  b_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (p_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  p_pshape[1].compatible(merged_hidden_size * lstm_seq_peepholes_count),
                                  "Parameter hidden_size mistmatched in P input. Current value is: ",
                                  p_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_peepholes_count,
                                  ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);
    set_input_is_relevant_to_shape(4);
    set_input_is_relevant_to_shape(5);
    set_input_is_relevant_to_shape(6);

    // Set output size, type and shape
    set_output_size(3);
    set_output_type(0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
    set_output_type(2, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
}

bool ngraph::op::v5::LSTMSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_LSTMSequence_visit_attributes);
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

shared_ptr<Node> op::v5::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 7) {
        return make_shared<op::v5::LSTMSequence>(new_args.at(0),  // X
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
        throw ngraph_error("Incorrect number of new arguments");
    }
}

void op::v5::LSTMSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v5_LSTMSequence_validate_and_infer_types);
    for (const auto& input : inputs()) {
        if (input.get_partial_shape().rank().is_dynamic()) {
            set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), ov::PartialShape::dynamic());
            set_output_type(2, get_input_element_type(0), ov::PartialShape::dynamic());
            return;
        }
    }
    std::vector<ov::PartialShape> input_param{};

    auto lstm_seq_gates_count = 4;
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto merged_num_directions = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Copy all inputs without initial_cell_state information for further validation
    for (size_t i = 0; i < get_input_size(); i++) {
        // exclude initial_cell_state from the loop
        if (i != 2) {
            input_param.push_back(get_input_partial_shape(i));
        }
    }

    // Get input partial shape for all inputs
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& sl_pshape = get_input_partial_shape(3);
    const auto& w_pshape = get_input_partial_shape(4);
    const auto& r_pshape = get_input_partial_shape(5);
    const auto& b_pshape = get_input_partial_shape(6);

    ngraph::op::util::validate_seq_input_rank_dimension(input_param);

    // Validate rank and dimension for initial_cell_state input
    NODE_VALIDATION_CHECK(this,
                          (ct_pshape.rank().get_length() == 3),
                          "LSTMSequence input tensor initial_cell_state shall have dimension 3D.");

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(2)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(4)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(5)) &&
                              element::Type::merge(result_et, result_et, get_input_element_type(6)),
                          "Element types for X, initial_hidden_state, initial_cell_state, W, R and B inputs do "
                          "not "
                          "match.");

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
                              Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
                          "Parameter batch_size not matched in LSTMSequence.");

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[2]) &&
                              Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                          "Parameter hidden_size not matched LSTMSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, ct_pshape[1]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
                              Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
                          "Parameter num_directions not matched in LSTMSequence.");

    auto valid_num_directions = 0;
    if (m_direction == op::RecurrentSequenceDirection::FORWARD ||
        m_direction == op::RecurrentSequenceDirection::REVERSE) {
        valid_num_directions = 1;
    } else if (m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        valid_num_directions = 2;
    } else {
        // Guard for potential future extension of RecurrentSequenceDirection enum
        NODE_VALIDATION_CHECK(this, false, "Parameter direction must be FORWARD or REVERSE or BIDIRECTIONAL.");
    }

    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_num_directions, merged_num_directions, valid_num_directions),
                          "Parameter 'num_directions' doesn't match with direction '",
                          m_direction,
                          "' in LSTMSequence. Expected ",
                          valid_num_directions,
                          ", actual ",
                          merged_num_directions);

    // Validate hidden_size value for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  w_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in W input. Current value is: ",
                                  w_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (r_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  r_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in R input. Current value is: ",
                                  r_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }

        if (b_pshape[1].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  b_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[1].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * lstm_seq_gates_count,
                                  ".");
        }
    }

    // Mark inputs which are relevant to output parameters
    for (size_t i = 0; i <= 6; ++i)
        set_input_is_relevant_to_shape(i);

    // Set output size, type and shape
    set_output_size(3);
    set_output_type(0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
    set_output_type(2, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
}
