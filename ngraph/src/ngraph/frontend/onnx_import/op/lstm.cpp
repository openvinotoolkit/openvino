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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "lstm.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/frontend/onnx_import/op/lstm.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/lstm_sequence.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMInput
                {
                    LSTM_INPUT_X,
                    LSTM_INPUT_W,
                    LSTM_INPUT_R,
                    LSTM_INPUT_B,
                    LSTM_INPUT_SEQ_LENGTHS,
                    LSTM_INPUT_INIT_H,
                    LSTM_INPUT_INIT_C,
                    LSTM_INPUT_P
                };

                struct LSTMNgInputMap
                {
                    using container_type = std::map<LSTMInput, std::shared_ptr<ngraph::Node>>;
                    using iterator = typename container_type::iterator;

                    explicit LSTMNgInputMap(const Node& node)
                    {
                        const auto& ng_inputs = node.get_ng_inputs();
                        // We have input, output, forget and cell gates
                        constexpr std::size_t gates_count{4};
                        // Peepholes add additional connections to input, output and forget gates.
                        constexpr std::size_t peepholes_count{3};

                        // ----- Mandatory inputs ------
                        // Packed input sequences. Shape: [seq_length, batch_size, input_size]
                        m_map[LSTMInput::LSTM_INPUT_X] = ng_inputs.at(0);
                        // Weight tensor for the gates.
                        // Shape: [num_directions, 4*hidden_size, input_size]
                        m_map[LSTMInput::LSTM_INPUT_W] = ng_inputs.at(1);
                        // The recurrence weight tensor.
                        // Shape: [num_directions, 4*hidden_size, hidden_size]
                        m_map[LSTMInput::LSTM_INPUT_R] = ng_inputs.at(2);

                        const std::size_t hidden_size =
                            m_map[LSTMInput::LSTM_INPUT_R]->get_shape().back();
                        const std::size_t batch_size =
                            m_map[LSTMInput::LSTM_INPUT_X]->get_shape().at(1);
                        const std::size_t num_directions =
                            m_map[LSTMInput::LSTM_INPUT_W]->get_shape().front();

                        // ------ Optional inputs ------
                        // The bias tensor for input gate. Shape [num_directions, 4*hidden_size]
                        if (ng_inputs.size() > 3 && !is_type<NullNode>(ng_inputs.at(3)))
                        {
                            auto bias = ng_inputs.at(3);
                            auto split_bias = builder::opset1::split(bias, 2, 1);
                            m_map[LSTMInput::LSTM_INPUT_B] = split_bias.at(0) + split_bias.at(1);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_B] = default_opset::Constant::create(
                                element::f32,
                                Shape{num_directions, gates_count * hidden_size},
                                std::vector<float>(num_directions * gates_count * hidden_size,
                                                   0.f));
                        }
                        // The lengths of the sequences in a batch. Shape [batch_size]
                        if (ng_inputs.size() > 4 && !is_type<NullNode>(ng_inputs.at(4)))
                        {
                            m_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ng_inputs.at(4);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] =
                                default_opset::Constant::create(
                                    element::i32,
                                    Shape{batch_size},
                                    std::vector<std::int32_t>(
                                        batch_size,
                                        m_map[LSTMInput::LSTM_INPUT_X]->get_shape().at(0)));
                        }
                        // The initial value of the hidden.
                        // Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 5 && !is_type<NullNode>(ng_inputs.at(5)))
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] = default_opset::Constant::create(
                                element::f32,
                                Shape{num_directions, batch_size, hidden_size},
                                std::vector<float>(num_directions * batch_size * hidden_size, 0.f));
                        }
                        // The initial value of the cell.
                        // Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 6 && !is_type<NullNode>(ng_inputs.at(6)))
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] = default_opset::Constant::create(
                                element::f32,
                                Shape{num_directions, batch_size, hidden_size},
                                std::vector<float>(num_directions * batch_size * hidden_size, 0.f));
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() > 7 && !is_type<NullNode>(ng_inputs.at(7)))
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = default_opset::Constant::create(
                                element::f32,
                                Shape{num_directions, peepholes_count * hidden_size},
                                std::vector<float>(num_directions * peepholes_count * hidden_size,
                                                   0.f));
                        }
                    }

                    std::shared_ptr<ngraph::Node>& at(const LSTMInput& key)
                    {
                        return m_map.at(key);
                    }
                    container_type m_map;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                struct LSTMAttributes
                {
                    explicit LSTMAttributes(const Node& node)
                        : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")}
                        , m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)}
                        , m_activations{node.get_attribute_value<std::vector<std::string>>(
                              "activations", {"sigmoid", "tanh", "tanh"})}
                        // Default values for activation functions are same as for corresponding
                        // ONNX operator.
                        , m_activation_alpha{node.get_attribute_value<std::vector<float>>(
                              "activation_alpha", std::vector<float>{})}
                        , m_activation_beta{node.get_attribute_value<std::vector<float>>(
                              "activation_beta", std::vector<float>{})}
                        , m_input_forget{static_cast<bool>(
                              node.get_attribute_value<std::int64_t>("input_forget", 0))}
                    {
                        m_clip_threshold = std::abs(m_clip_threshold);
                        std::string direction = ngraph::to_lower(
                            node.get_attribute_value<std::string>("direction", "forward"));
                        NGRAPH_CHECK(direction == "bidirectional" || direction == "forward" ||
                                         direction == "reverse",
                                     "Provided direction: ",
                                     direction,
                                     " is invalid");
                        if (direction == "forward")
                        {
                            m_direction = default_opset::LSTMSequence::direction::FORWARD;
                        }
                        else if (direction == "reverse")
                        {
                            m_direction = default_opset::LSTMSequence::direction::REVERSE;
                        }
                        else // (direction == "bidirectional")
                        {
                            m_direction = default_opset::LSTMSequence::direction::BIDIRECTIONAL;
                        }
                    }

                    ngraph::op::LSTMSequence::direction m_direction;
                    std::int64_t m_hidden_size;
                    float m_clip_threshold;
                    std::vector<std::string> m_activations;
                    std::vector<float> m_activation_alpha;
                    std::vector<float> m_activation_beta;
                    bool m_input_forget;
                };

            } // anonymous namespace

            namespace set_1
            {
                NodeVector lstm(const Node& node)
                {
                    LSTMNgInputMap input_map{node};
                    LSTMAttributes attributes{node};

                    auto lstmSequence = std::make_shared<default_opset::LSTMSequence>(
                        input_map.at(LSTMInput::LSTM_INPUT_X),
                        input_map.at(LSTMInput::LSTM_INPUT_INIT_H),
                        input_map.at(LSTMInput::LSTM_INPUT_INIT_C),
                        input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                        input_map.at(LSTMInput::LSTM_INPUT_W),
                        input_map.at(LSTMInput::LSTM_INPUT_R),
                        input_map.at(LSTMInput::LSTM_INPUT_B),
                        input_map.at(LSTMInput::LSTM_INPUT_P),
                        attributes.m_hidden_size,
                        attributes.m_direction,
                        ngraph::op::LSTMWeightsFormat::IOFC,
                        attributes.m_activation_alpha,
                        attributes.m_activation_beta,
                        attributes.m_activations,
                        attributes.m_clip_threshold,
                        attributes.m_input_forget);
                    return {std::make_shared<ngraph::opset0::GetOutputElement>(lstmSequence, 0),
                            std::make_shared<ngraph::opset0::GetOutputElement>(lstmSequence, 1),
                            std::make_shared<ngraph::opset0::GetOutputElement>(lstmSequence, 2)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
