//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/lstm_sequence.hpp"
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "ngraph/op/util/recurrent_sequence.hpp"

using namespace ngraph;
using namespace std;

NGRAPH_SUPPRESS_DEPRECATED_START

NGRAPH_RTTI_DEFINITION(op::v0::LSTMSequence, "LSTMSequence", 0);
NGRAPH_RTTI_DEFINITION(op::v5::LSTMSequence, "LSTMSequence", 5);

op::v0::LSTMSequence::LSTMSequence()
    : FusedOp()
    , m_activations_alpha()
    , m_activations_beta()
    , m_activations()
    , m_clip_threshold()
    , m_direction()
    , m_hidden_size()
    , m_input_forget()
    , m_weights_format()
{
}

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
    : FusedOp({X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B, P})
    , m_activations_alpha(activations_alpha)
    , m_activations_beta(activations_beta)
    , m_activations(activations)
    , m_clip_threshold(clip_threshold)
    , m_direction(lstm_direction)
    , m_hidden_size(hidden_size)
    , m_input_forget(input_forget)
    , m_weights_format(weights_format)
{
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
          Constant::create(
              element::f32,
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
          input_forget)
{
}

bool op::v0::LSTMSequence::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_LSTMSequence_visit_attributes);
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

OutputVector op::v0::LSTMSequence::decompose_op() const
{
    OutputVector results;
    if (m_direction == direction::FORWARD || m_direction == direction::REVERSE)
    {
        results = lstm_pass(m_direction == direction::REVERSE);
    }
    if (m_direction == direction::BIDIRECTIONAL)
    {
        OutputVector fwd_results{lstm_pass()};
        OutputVector rev_results{lstm_pass(true)};

        // Stack together respective outputs from both forward and reverse passess.
        shared_ptr<Node> Y{
            make_shared<opset1::Concat>(OutputVector{fwd_results.at(0), rev_results.at(0)}, 1)};
        shared_ptr<Node> Y_h{
            make_shared<opset1::Concat>(OutputVector{fwd_results.at(1), rev_results.at(1)}, 1)};
        shared_ptr<Node> Y_c{
            make_shared<opset1::Concat>(OutputVector{fwd_results.at(2), rev_results.at(2)}, 1)};
        results = OutputVector{Y, Y_h, Y_c};
    }
    return results;
}

shared_ptr<Node> op::v0::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 8)
    {
        return make_shared<op::v0::LSTMSequence>(new_args.at(0), // X
                                                 new_args.at(1), // initial_hidden_state
                                                 new_args.at(2), // initial_cell_state
                                                 new_args.at(3), // sequence_lengths
                                                 new_args.at(4), // W
                                                 new_args.at(5), // R
                                                 new_args.at(6), // B
                                                 new_args.at(7), // P
                                                 m_hidden_size,
                                                 m_direction,
                                                 m_weights_format,
                                                 m_activations_alpha,
                                                 m_activations_beta,
                                                 m_activations,
                                                 m_clip_threshold,
                                                 m_input_forget);
    }
    else if (new_args.size() == 7)
    {
        return make_shared<op::v0::LSTMSequence>(new_args.at(0), // X
                                                 new_args.at(1), // initial_hidden_state
                                                 new_args.at(2), // initial_cell_state
                                                 new_args.at(3), // sequence_lengths
                                                 new_args.at(4), // W
                                                 new_args.at(5), // R
                                                 new_args.at(6), // B
                                                 m_hidden_size,
                                                 m_direction,
                                                 m_weights_format,
                                                 m_activations_alpha,
                                                 m_activations_beta,
                                                 m_activations,
                                                 m_clip_threshold,
                                                 m_input_forget);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

shared_ptr<Node> op::v0::LSTMSequence::get_masked_node(const Output<Node>& data,
                                                       int32_t time_step,
                                                       size_t batch_axis,
                                                       const Output<Node>& default_value) const
{
    Output<Node> mask_value = default_value;
    // Create zero mask value node.
    if (!mask_value.get_node_shared_ptr())
    {
        mask_value = opset1::Constant::create(data.get_element_type(),
                                              data.get_shape(),
                                              vector<float>(shape_size(data.get_shape()), 0.f));
    }

    // Create predicate nodes. The condition is whether current time step value
    // is greater than sequence length for respective batch inputs.
    shared_ptr<Node> curr_time_step_node = opset1::Constant::create(
        element::i32, data.get_shape(), vector<int32_t>(shape_size(data.get_shape()), time_step));

    Output<Node> batch_seq_length = builder::opset1::legacy_broadcast_for_binary_operation(
        curr_time_step_node, input_value(3).get_node_shared_ptr(), batch_axis);

    // Create mask node deciding whether or not to mask batch data.
    shared_ptr<Node> mask_condition =
        make_shared<opset1::Greater>(curr_time_step_node, batch_seq_length);

    // Select values depnding on mask_condition.
    // Select(<condition>, <true_value>, <false_value>)
    return make_shared<opset1::Select>(mask_condition, mask_value, data);
}

OutputVector op::v0::LSTMSequence::lstm_pass(bool is_reverse) const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ INPUTS ------
    // X - The input tensor. [batch_size, seq_length, input_size]
    // W - The weight tensor. [num_directions, 4*hidden_size, input_size]
    // R - The recurrence weight tensor. [num_directions, 4*hidden_size, hidden_size]
    // B - The bias tensor for input gate. [num_directions, 8*hidden_size]
    // P - The weight tensor for peepholes. [num_directions, 3*hidde_size]
    // ------ ACRONYMS ------
    // i - input gate
    // o - output gate
    // f - forget gate
    // c - cell gate
    // t - time step (t-1 means previous time step)
    // ------ VARIABLE NAMES ------
    // H_t     - Hidden state vector at current time step. [batch_size, num_directions, hidden_size]
    // C_t     - Cell state vector at current time step. [batch_size, num_directions, hidden_size]
    // h_list  - The list of hidden states at all processed time steps.

    NodeVector h_list;
    shared_ptr<Node> X = input_value(0).get_node_shared_ptr();
    shared_ptr<Node> H_t = prepare_input(input_value(1), is_reverse, 1);
    shared_ptr<Node> C_t = prepare_input(input_value(2), is_reverse, 1);
    shared_ptr<Node> seq_lengths = input_value(3).get_node_shared_ptr();
    shared_ptr<Node> W = prepare_input(input_value(4), is_reverse);
    shared_ptr<Node> R = prepare_input(input_value(5), is_reverse);
    shared_ptr<Node> B = prepare_input(input_value(6), is_reverse);
    shared_ptr<Node> P = prepare_input(input_value(7), is_reverse);

    if (is_reverse)
    {
        X = make_shared<opset1::ReverseSequence>(X, seq_lengths, 0 /*batch_axis*/, 1 /*seq_axis*/);
    }

    OutputVector in_seqs = builder::opset1::split(X, X->get_shape().at(1), 1);

    for (auto& in_x : in_seqs)
    {
        // Remove empty dim, after above split.
        in_x = builder::opset1::squeeze(in_x, {1});
    }

    int32_t time_step{1};
    for (const auto& in_x : in_seqs)
    {
        shared_ptr<Node> lstm_cell = make_shared<opset1::LSTMCell>(in_x,
                                                                   H_t,
                                                                   C_t,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   P,
                                                                   m_hidden_size,
                                                                   m_weights_format,
                                                                   m_activations,
                                                                   m_activations_alpha,
                                                                   m_activations_beta,
                                                                   m_clip_threshold,
                                                                   m_input_forget);

        Output<Node> H = lstm_cell->output(0);
        Output<Node> C = lstm_cell->output(1);

        // Expand tensors with empty outermost dim, so we can later concatenate
        // them.
        // Mask hidden state tensor in order to handle mixed sequence lengths.
        // This results in zeroing out values in batches with sequence shorter
        // than current time_step.
        h_list.push_back(get_masked_node(builder::opset1::expand_dims(H, 1), time_step, 0));
        // Reference implementation in ONNX Runtime doesn't mask values of Y_h
        // and Y_c outputs, thus here we make sure that only appropriate batches
        // (in respect to its sequence length) are updated. Those batches which
        // has shorter sequences preserve the last value.
        H_t = get_masked_node(H, time_step, 0, H_t);
        C_t = get_masked_node(C, time_step, 0, C_t);
        time_step++;
    }
    // The tensor that concats all the intermediate output values of the hidden.
    // It has shape [batch_size, seq_length, hidden_size]
    shared_ptr<Node> Y{make_shared<opset1::Concat>(h_list, 1)};

    // Get back the original order of the output data.
    if (is_reverse)
    {
        Y = make_shared<opset1::ReverseSequence>(Y, seq_lengths, 0 /*batch_axis*/, 1 /*seq_axis*/);
    }

    // Expand Y so that it has expected shape:
    // [batch_size, num_directions, seq_length, hidden_size]
    Y = builder::opset1::expand_dims(Y, 1);

    // expand H_t and C_t so that it has expected shape:
    // [ batch_size, num_directions, hidden_size]
    auto Y_h = builder::opset1::expand_dims(H_t, 1);
    auto Y_c = builder::opset1::expand_dims(C_t, 1);
    return {Y, Y_h, Y_c};
}

shared_ptr<Node> op::v0::LSTMSequence::prepare_input(Output<Node> node,
                                                     bool is_reverse,
                                                     size_t num_direction_axis) const
{
    // In bidirectional mode inputs are stacked together, so we must split them.
    Output<Node> tmp = node;
    if (m_direction == direction::BIDIRECTIONAL)
    {
        tmp = builder::opset1::split(node, 2, num_direction_axis).at(is_reverse ? 1 : 0);
    }
    // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
    return builder::opset1::squeeze(tmp, {num_direction_axis});
}

void op::v0::LSTMSequence::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_LSTMSequence_validate_and_infer_types);
    std::vector<ngraph::PartialShape> input_param{};

    auto lstm_seq_gates_count = 4;
    auto lstm_seq_peepholes_count = 3;
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto merged_num_directions = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Copy all inputs without peephole and initial_cell_state information for further
    // validation
    for (size_t i = 0; i < get_input_size() - 1; i++)
    {
        // exclude initial_cell_state from the loop
        if (i != 2)
        {
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
    NODE_VALIDATION_CHECK(
        this, (p_pshape.rank().is_static()), "LSTMSequence input tensor P shall have static rank.");

    NODE_VALIDATION_CHECK(this,
                          (p_pshape.rank().get_length() == 2),
                          "LSTMSequence input tensor P shall have dimension 2D.");

    // Validate input types and save result for output type
    NODE_VALIDATION_CHECK(
        this,
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
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
        "Parameter batch_size not matched in LSTMSequence.");

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[2]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
        "Parameter hidden_size not matched LSTMSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, ct_pshape[1]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
        "Parameter num_directions not matched in LSTMSequence.");

    // Validate hidden_size value for W, R, B and P inputs
    if (merged_hidden_size.is_static())
    {
        if (w_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                w_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                "Parameter hidden_size mistmatched in P input. Current value is: ",
                w_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * lstm_seq_gates_count,
                ".");
        }

        if (r_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                r_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                "Parameter hidden_size mistmatched in R input. Current value is: ",
                r_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * lstm_seq_gates_count,
                ".");
        }

        if (b_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                b_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                "Parameter hidden_size mistmatched in B input. Current value is: ",
                b_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * lstm_seq_gates_count,
                ".");
        }

        if (p_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
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
    set_output_type(
        0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
    set_output_type(2, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
}

bool ngraph::op::v5::LSTMSequence::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v5_LSTMSequence_visit_attributes);
    visitor.on_attribute("direction", m_direction);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

shared_ptr<Node> op::v5::LSTMSequence::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v5_LSTMSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 7)
    {
        return make_shared<op::v5::LSTMSequence>(new_args.at(0), // X
                                                 new_args.at(1), // initial_hidden_state
                                                 new_args.at(2), // initial_cell_state
                                                 new_args.at(3), // sequence_lengths
                                                 new_args.at(4), // W
                                                 new_args.at(5), // R
                                                 new_args.at(6), // B
                                                 m_hidden_size,
                                                 m_direction,
                                                 m_activations_alpha,
                                                 m_activations_beta,
                                                 m_activations,
                                                 m_clip);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

void op::v5::LSTMSequence::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v5_LSTMSequence_validate_and_infer_types);
    for (const auto& input : inputs())
    {
        if (input.get_partial_shape().rank().is_dynamic())
        {
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
            set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
            set_output_type(2, get_input_element_type(0), PartialShape::dynamic());
            return;
        }
    }
    std::vector<ngraph::PartialShape> input_param{};

    auto lstm_seq_gates_count = 4;
    auto merged_batch_size = Dimension::dynamic();
    auto merged_hidden_size = Dimension::dynamic();
    auto merged_num_directions = Dimension::dynamic();
    auto result_et = element::dynamic;

    // Copy all inputs without initial_cell_state information for further validation
    for (size_t i = 0; i < get_input_size(); i++)
    {
        // exclude initial_cell_state from the loop
        if (i != 2)
        {
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
    NODE_VALIDATION_CHECK(
        this,
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
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, x_pshape[0]) &&
            Dimension::merge(merged_batch_size, merged_batch_size, sl_pshape[0]),
        "Parameter batch_size not matched in LSTMSequence.");

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_hidden_size, merged_hidden_size, ht_pshape[2]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, ct_pshape[2]) &&
            Dimension::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
        "Parameter hidden_size not matched LSTMSequence.");

    // Merge num_directions dimension across all inputs to evaluate output dimension
    NODE_VALIDATION_CHECK(
        this,
        Dimension::merge(merged_num_directions, merged_num_directions, ht_pshape[1]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, ct_pshape[1]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, w_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, r_pshape[0]) &&
            Dimension::merge(merged_num_directions, merged_num_directions, b_pshape[0]),
        "Parameter num_directions not matched in LSTMSequence.");

    // Validate hidden_size value for W, R, B inputs
    if (merged_hidden_size.is_static())
    {
        if (w_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                w_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                "Parameter hidden_size mistmatched in W input. Current value is: ",
                w_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * lstm_seq_gates_count,
                ".");
        }

        if (r_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                r_pshape[1].compatible(merged_hidden_size * lstm_seq_gates_count),
                "Parameter hidden_size mistmatched in R input. Current value is: ",
                r_pshape[1].get_length(),
                ", expected: ",
                merged_hidden_size.get_length() * lstm_seq_gates_count,
                ".");
        }

        if (b_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
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
    set_output_type(
        0, result_et, {merged_batch_size, merged_num_directions, x_pshape[1], merged_hidden_size});
    set_output_type(1, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
    set_output_type(2, result_et, {merged_batch_size, merged_num_directions, merged_hidden_size});
}
