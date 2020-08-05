//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"
#include "ngraph/frontend/onnx_import/core/null_node.hpp"
#include "recurrent.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace recurrent
        {
            OpInputMap::OpInputMap(const onnx_import::Node& node, std::size_t gates_count)
            {
                const auto& ng_inputs = node.get_ng_inputs();

                m_map[OpInput::X] = ng_inputs.at(0);
                m_map[OpInput::W] = ng_inputs.at(1);
                m_map[OpInput::R] = ng_inputs.at(2);

                const auto el_type = ng_inputs.at(0).get_element_type();

                const auto x_pshape = m_map[OpInput::X].get_partial_shape();
                const auto w_pshape = m_map[OpInput::W].get_partial_shape();
                const auto r_pshape = m_map[OpInput::R].get_partial_shape();
                NGRAPH_CHECK(x_pshape.rank().is_static() && x_pshape[0].is_static() &&
                                 x_pshape[1].is_static(),
                             "RecurrentSequence input X must have static \"seq_length\" and "
                             "\"batch_size\" dimensions.");
                NGRAPH_CHECK(w_pshape.rank().is_static() && w_pshape[0].is_static(),
                             "RecurrentSequence input W must have static \"num_directions\" "
                             "(outermost) dimension.");
                NGRAPH_CHECK(r_pshape.rank().is_static() && r_pshape[2].is_static(),
                             "RecurrentSequence input R must have static \"hidden_size\" "
                             "(innermost) dimension.");

                const std::size_t hidden_size = m_map[OpInput::R].get_shape().back();
                const std::size_t batch_size = m_map[OpInput::X].get_shape().at(1);
                const std::size_t num_directions = m_map[OpInput::W].get_shape().front();

                if (ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs.at(3)))
                {
                    auto bias = ng_inputs.at(3);
                    auto split_bias = builder::opset1::split(bias, 2, 1);
                    m_map[OpInput::B] = split_bias.at(0) + split_bias.at(1);
                }
                else
                {
                    m_map[OpInput::B] = std::make_shared<default_opset::Constant>(
                        el_type, Shape{num_directions, gates_count * hidden_size}, 0.f);
                }
                if (ng_inputs.size() > 4 && !ngraph::op::is_null(ng_inputs.at(4)))
                {
                    m_map[OpInput::SEQ_LENGTHS] = ng_inputs.at(4);
                }
                else
                {
                    m_map[OpInput::SEQ_LENGTHS] = std::make_shared<default_opset::Constant>(
                        element::i32, Shape{batch_size}, m_map[OpInput::X].get_shape().at(0));
                }
                // The initial value of the hidden.
                if (ng_inputs.size() > 5 && !ngraph::op::is_null(ng_inputs.at(5)))
                {
                    m_map[OpInput::INIT_H] = ng_inputs.at(5);
                }
                else
                {
                    m_map[OpInput::INIT_H] = std::make_shared<default_opset::Constant>(
                        el_type, Shape{num_directions, batch_size, hidden_size}, 0.f);
                }
            }

            OpInputMap::OpInputMap(container_type&& map)
                : m_map(std::move(map))
            {
            }

            Output<ngraph::Node>& OpInputMap::at(const OpInput& key) { return m_map.at(key); }
            const Output<ngraph::Node>& OpInputMap::at(const OpInput& key) const
            {
                return m_map.at(key);
            }

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            OpAttributes::OpAttributes(const Node& node)
                : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")}
                , m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)}
                // Recurrent Operators which have more activation functions should override
                // this value in constructor of respective Attributes' struct.
                , m_activations{node.get_attribute_value<std::vector<std::string>>("activations",
                                                                                   {"tanh"})}
                // Default values for activation functions are same as for corresponding
                // ONNX operator.
                , m_activations_alpha{node.get_attribute_value<std::vector<float>>(
                      "activation_alpha", std::vector<float>{})}
                , m_activations_beta{node.get_attribute_value<std::vector<float>>(
                      "activation_beta", std::vector<float>{})}
            {
                m_clip_threshold = std::abs(m_clip_threshold);
                std::string direction =
                    ngraph::to_lower(node.get_attribute_value<std::string>("direction", "forward"));
                m_direction = ngraph::as_enum<ngraph::op::RecurrentSequenceDirection>(direction);
            }

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sequence Computations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            RecurrentSequence::RecurrentSequence(OpInputMap& args,
                                                 ngraph::op::RecurrentSequenceDirection direction)
                : m_args(args)
                , m_direction(direction)
            {
            }

            OutputVector RecurrentSequence::run_sequence(const RecurrentCellFunction& kernel)
            {
                OutputVector results;
                if (m_direction == ngraph::op::RecurrentSequenceDirection::FORWARD ||
                    m_direction == ngraph::op::RecurrentSequenceDirection::REVERSE)
                {
                    results = recurrent_sequence_pass(
                        kernel, m_direction == ngraph::op::RecurrentSequenceDirection::REVERSE);
                }
                else if (m_direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                {
                    OutputVector fwd_results{recurrent_sequence_pass(kernel)};
                    OutputVector rev_results{recurrent_sequence_pass(kernel, true)};

                    // Stack together respective outputs from both forward and reverse passess.
                    std::shared_ptr<ngraph::Node> Y{std::make_shared<default_opset::Concat>(
                        OutputVector{fwd_results.at(0), rev_results.at(0)}, 1)};
                    results.push_back(Y);

                    std::shared_ptr<ngraph::Node> Y_h{std::make_shared<default_opset::Concat>(
                        OutputVector{fwd_results.at(1), rev_results.at(1)}, 0)};
                    results.push_back(Y_h);
                }
                else
                {
                    throw ngraph_error(
                        "RecurrentSequence: unhandled direction mode during decomposition.");
                }
                return results;
            }

            OutputVector
                RecurrentSequence::recurrent_sequence_pass(const RecurrentCellFunction& kernel,
                                                           bool is_reverse)
            {
                OutputVector h_list;

                // back-up nodes which we may later modify.
                Output<ngraph::Node> orig_W = m_args.at(OpInput::W);
                Output<ngraph::Node> orig_R = m_args.at(OpInput::R);
                Output<ngraph::Node> orig_B = m_args.at(OpInput::B);

                Output<ngraph::Node> X = m_args.at(OpInput::X);
                Output<ngraph::Node> H_t = prepare_input(m_args.at(OpInput::INIT_H), is_reverse);
                Output<ngraph::Node> W = prepare_input(m_args.at(OpInput::W), is_reverse);
                Output<ngraph::Node> R = prepare_input(m_args.at(OpInput::R), is_reverse);
                Output<ngraph::Node> B = prepare_input(m_args.at(OpInput::B), is_reverse);
                Output<ngraph::Node> seq_lengths = m_args.at(OpInput::SEQ_LENGTHS);

                m_args.at(OpInput::W) = W;
                m_args.at(OpInput::R) = R;
                m_args.at(OpInput::B) = B;

                if (is_reverse)
                {
                    X = std::make_shared<default_opset::ReverseSequence>(
                        X, seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                }

                OutputVector in_seq_steps = builder::opset1::split(X, X.get_shape().at(0));

                for (auto& in_x : in_seq_steps)
                {
                    // remove first empty dim, after above split.
                    in_x = builder::opset1::squeeze(in_x);
                }

                int32_t time_step{1};
                for (const auto& in_x : in_seq_steps)
                {
                    Output<ngraph::Node> H = kernel(m_args, in_x, H_t);

                    // Expand tensors with empty outermost dim, so we can later concatenate
                    // them.
                    // Mask hidden state tensor in order to handle mixed sequence lengths.
                    // This results in zeroing out values in batches with sequence shorter
                    // than current time_step.
                    h_list.push_back(
                        get_masked_node(builder::opset1::expand_dims(H), time_step, 1));

                    // Here we make sure that only appropriate batches (with respect to its sequence
                    // length) are updated. Those batches which has shorter sequences preserve
                    // the last value.
                    H_t = get_masked_node(H, time_step, 0, H_t);
                    time_step++;
                }

                // Get back original nodes.
                m_args.at(OpInput::W) = orig_W;
                m_args.at(OpInput::R) = orig_R;
                m_args.at(OpInput::B) = orig_B;

                // The tensor that concats all the intermediate output values of the hidden.
                // It has shape [seq_length, batch_size, hidden_size]
                std::shared_ptr<ngraph::Node> Y{std::make_shared<default_opset::Concat>(h_list, 0)};

                // Get back the original order of the output data.
                if (is_reverse)
                {
                    Y = std::make_shared<default_opset::ReverseSequence>(
                        Y, seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                }

                // Expand Y so that it has expected shape:
                // [seq_length, num_directions, batch_size, hidden_size]
                Y = builder::opset1::expand_dims(Y, 1);

                // Expand H_t so that it has expected shape:
                // [num_directions, batch_size, hidden_size]
                auto Y_h = builder::opset1::expand_dims(H_t);

                return {Y, Y_h};
            }

            std::shared_ptr<ngraph::Node>
                RecurrentSequence::get_masked_node(const Output<ngraph::Node>& data,
                                                   int32_t time_step,
                                                   size_t batch_axis,
                                                   const Output<ngraph::Node>& default_value) const
            {
                Output<ngraph::Node> mask_value = default_value;
                // Create zero mask value node.
                if (!mask_value.get_node_shared_ptr())
                {
                    mask_value = std::make_shared<default_opset::Constant>(
                        data.get_element_type(), data.get_shape(), 0.f);
                }

                // Create predicate nodes. The condition is whether current time step value
                // is greater than sequence length for respective batch inputs.
                std::shared_ptr<ngraph::Node> curr_time_step_node =
                    std::make_shared<default_opset::Constant>(
                        element::i32, data.get_shape(), time_step);

                Output<ngraph::Node> batch_seq_length =
                    builder::opset1::legacy_broadcast_for_binary_operation(
                        curr_time_step_node, m_args.at(OpInput::SEQ_LENGTHS), batch_axis);

                // Create mask node deciding whether or not to mask batch data.
                std::shared_ptr<ngraph::Node> mask_condition =
                    std::make_shared<default_opset::Greater>(curr_time_step_node, batch_seq_length);

                // Select values depnding on mask_condition.
                // Select(<condition>, <true_value>, <false_value>)
                return std::make_shared<default_opset::Select>(mask_condition, mask_value, data);
            }

            std::shared_ptr<ngraph::Node>
                RecurrentSequence::prepare_input(Output<ngraph::Node> node, bool is_reverse) const
            {
                // In bidirectional mode inputs are stacked together, so we must split them.
                Output<ngraph::Node> tmp = node;
                if (m_direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                {
                    tmp = builder::opset1::split(node, 2).at(is_reverse ? 1 : 0);
                }
                // Since we work in forward pass mode, we can squeeze `num_directions` axis from
                // input.
                return builder::opset1::squeeze(tmp);
            }

        } // recurrent
    }     // onnx_import
} // ngraph
