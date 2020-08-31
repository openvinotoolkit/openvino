////*****************************************************************************
//// Copyright 2020 Intel Corporation
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////     http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
////*****************************************************************************
//
//#pragma once
//
//#include <cmath>
//#include <ngraph/runtime/reference/lstm_cell.hpp
//
//namespace ngraph
//{
//    namespace runtime
//    {
//        namespace reference
//        {
//            template <typename T>
//            void lstm_sequence(const T* X,
//                           const Shape& X_shape,
//                           const T* H,
//                           const Shape& H_shape,
//                           const T* C,
//                           const Shape& C_shape,
//                           const T* W,
//                           const Shape& W_shape,
//                           const T* R,
//                           const Shape& R_shape,
//                           const T* B,
//                           const Shape& B_shape,
//                           T* out_Ht,
//                           T* out_Ct,
//                           const std::string& activation_f,
//                           const std::string& activation_g,
//                           const std::string& activation_h,
//                           float clip)
//            {
//                OutputVector op::v0::LSTMSequence::decompose_op() const
//                {
//                    OutputVector results;
//                    if (m_direction == direction::FORWARD || m_direction == direction::REVERSE)
//                    {
//                        results = lstm_pass(m_direction == direction::REVERSE);
//                    }
//                    if (m_direction == direction::BIDIRECTIONAL)
//                    {
//                        OutputVector fwd_results{lstm_pass()};
//                        OutputVector rev_results{lstm_pass(true)};
//
//                        // Stack together respective outputs from both forward and reverse passess.
//                        shared_ptr<Node> Y{
//                                make_shared<opset1::Concat>(OutputVector{fwd_results.at(0), rev_results.at(0)}, 1)};
//                        shared_ptr<Node> Y_h{
//                                make_shared<opset1::Concat>(OutputVector{fwd_results.at(1), rev_results.at(1)}, 1)};
//                        shared_ptr<Node> Y_c{
//                                make_shared<opset1::Concat>(OutputVector{fwd_results.at(2), rev_results.at(2)}, 1)};
//                        results = OutputVector{Y, Y_h, Y_c};
//                    }
//                    return results;
//                }
//                OutputVector op::v0::LSTMSequence::lstm_pass(bool is_reverse) const
//                {
//                    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
//                    // The names used below are analogous to the one used in ONNX documentation.
//                    //
//                    // ------ INPUTS ------
//                    // X - The input tensor. [batch_size, seq_length, input_size]
//                    // W - The weight tensor. [num_directions, 4*hidden_size, input_size]
//                    // R - The recurrence weight tensor. [num_directions, 4*hidden_size, hidden_size]
//                    // B - The bias tensor for input gate. [num_directions, 8*hidden_size]
//                    // ------ ACRONYMS ------
//                    // i - input gate
//                    // o - output gate
//                    // f - forget gate
//                    // c - cell gate
//                    // t - time step (t-1 means previous time step)
//                    // ------ VARIABLE NAMES ------
//                    // H_t     - Hidden state vector at current time step. [batch_size, num_directions, hidden_size]
//                    // C_t     - Cell state vector at current time step. [batch_size, num_directions, hidden_size]
//                    // h_list  - The list of hidden states at all processed time steps.
//
//                    NodeVector h_list;
//                    shared_ptr<Node> X = input_value(0).get_node_shared_ptr();
//                    shared_ptr<Node> H_t = prepare_input(input_value(1), is_reverse, 1);
//                    shared_ptr<Node> C_t = prepare_input(input_value(2), is_reverse, 1);
//                    shared_ptr<Node> seq_lengths = input_value(3).get_node_shared_ptr();
//                    shared_ptr<Node> W = prepare_input(input_value(4), is_reverse);
//                    shared_ptr<Node> R = prepare_input(input_value(5), is_reverse);
//                    shared_ptr<Node> B = prepare_input(input_value(6), is_reverse);
//
//                    if (is_reverse)
//                    {
//                        X = make_shared<opset1::ReverseSequence>(X, seq_lengths, 0 /*batch_axis*/, 1 /*seq_axis*/);
//                    }
//
//                    OutputVector in_seqs = builder::opset1::split(X, X->get_shape().at(1), 1);
//
//                    for (auto& in_x : in_seqs)
//                    {
//                        // Remove empty dim, after above split.
//                        in_x = builder::opset1::squeeze(in_x, {1});
//                    }
//
//                    int32_t time_step{1};
//                    for (const auto& in_x : in_seqs)
//                    {
//                        shared_ptr<Node> lstm_cell = make_shared<opset1::LSTMCell>(in_x,
//                                                                                   H_t,
//                                                                                   C_t,
//                                                                                   W,
//                                                                                   R,
//                                                                                   B,
//                                                                                   m_hidden_size,
//                                                                                   m_activations,
//                                                                                   m_activations_alpha,
//                                                                                   m_activations_beta,
//                                                                                   m_clip);
//
//                        Output<Node> H = lstm_cell->output(0);
//                        Output<Node> C = lstm_cell->output(1);
//
//                        // Expand tensors with empty outermost dim, so we can later concatenate
//                        // them.
//                        // Mask hidden state tensor in order to handle mixed sequence lengths.
//                        // This results in zeroing out values in batches with sequence shorter
//                        // than current time_step.
//                        h_list.push_back(get_masked_node(builder::opset1::expand_dims(H, 1), time_step, 0));
//                        // Reference implementation in ONNX Runtime doesn't mask values of Y_h
//                        // and Y_c outputs, thus here we make sure that only appropriate batches
//                        // (in respect to its sequence length) are updated. Those batches which
//                        // has shorter sequences preserve the last value.
//                        H_t = get_masked_node(H, time_step, 0, H_t);
//                        C_t = get_masked_node(C, time_step, 0, C_t);
//                        time_step++;
//                    }
//                    // The tensor that concats all the intermediate output values of the hidden.
//                    // It has shape [batch_size, seq_length, hidden_size]
//                    shared_ptr<Node> Y{make_shared<opset1::Concat>(h_list, 1)};
//
//                    // Get back the original order of the output data.
//                    if (is_reverse)
//                    {
//                        Y = make_shared<opset1::ReverseSequence>(Y, seq_lengths, 0 /*batch_axis*/, 1 /*seq_axis*/);
//                    }
//
//                    // Expand Y so that it has expected shape:
//                    // [batch_size, num_directions, seq_length, hidden_size]
//                    Y = builder::opset1::expand_dims(Y, 1);
//
//                    // expand H_t and C_t so that it has expected shape:
//                    // [ batch_size, num_directions, hidden_size]
//                    auto Y_h = builder::opset1::expand_dims(H_t, 1);
//                    auto Y_c = builder::opset1::expand_dims(C_t, 1);
//                    return {Y, Y_h, Y_c};
//                }
//
//                shared_ptr<Node> op::v0::LSTMSequence::prepare_input(Output<Node> node,
//                                                                     bool is_reverse,
//                                                                     size_t num_direction_axis) const
//                {
//                    // In bidirectional mode inputs are stacked together, so we must split them.
//                    Output<Node> tmp = node;
//                    if (m_direction == direction::BIDIRECTIONAL)
//                    {
//                        tmp = builder::opset1::split(node, 2, num_direction_axis).at(is_reverse ? 1 : 0);
//                    }
//                    // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
//                    return builder::opset1::squeeze(tmp, {num_direction_axis});
//                }
//                shared_ptr<Node> op::v0::LSTMSequence::get_masked_node(const Output<Node>& data,
//                                                                       int32_t time_step,
//                                                                       size_t batch_axis,
//                                                                       const Output<Node>& default_value) const
//                {
//                    Output<Node> mask_value = default_value;
//                    // Create zero mask value node.
//                    if (!mask_value.get_node_shared_ptr())
//                    {
//                        mask_value = opset1::Constant::create(data.get_element_type(),
//                                                              data.get_shape(),
//                                                              vector<float>(shape_size(data.get_shape()), 0.f));
//                    }
//
//                    // Create predicate nodes. The condition is whether current time step value
//                    // is greater than sequence length for respective batch inputs.
//                    shared_ptr<Node> curr_time_step_node = opset1::Constant::create(
//                            element::i32, data.get_shape(), vector<int32_t>(shape_size(data.get_shape()), time_step));
//
//                    Output<Node> batch_seq_length = builder::opset1::legacy_broadcast_for_binary_operation(
//                            curr_time_step_node, input_value(3).get_node_shared_ptr(), batch_axis);
//
//                    // Create mask node deciding whether or not to mask batch data.
//                    shared_ptr<Node> mask_condition =
//                            make_shared<opset1::Greater>(curr_time_step_node, batch_seq_length);
//
//                    // Select values depnding on mask_condition.
//                    // Select(<condition>, <true_value>, <false_value>)
//                    return make_shared<opset1::Select>(mask_condition, mask_value, data);
//                }l
//            }
//        }
//    }
//}
