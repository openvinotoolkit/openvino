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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/enum_names.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/lstm_sequence.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "op/lstm.hpp"

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
                    explicit LSTMNgInputMap(const Node& node)
                    {
                        const auto& ng_inputs = node.get_ng_inputs();
                        // We have input, output, forget and cell gates
                        constexpr std::size_t gates_count{4};

                        // ----- Mandatory inputs ------
                        // Packed input sequences.
                        // ONNX Shape: [seq_length, batch_size, input_size]
                        // OpenVino Shape: [batch_size, seq_length, input_size]
                        m_input_map[LSTMInput::LSTM_INPUT_X] =
                            builder::opset1::reorder_axes(ng_inputs.at(0), {1, 0, 2});

                        // Weight tensor for the gates.
                        // Shape: [num_directions, 4*hidden_size, input_size]
                        m_input_map[LSTMInput::LSTM_INPUT_W] =
                            ngraph::op::util::convert_lstm_node_format(
                                ng_inputs.at(1),
                                ngraph::op::util::LSTMWeightsFormat::IOFC,
                                ngraph::op::util::LSTMWeightsFormat::FICO,
                                1);

                        // The recurrence weight tensor.
                        // Shape: [num_directions, 4*hidden_size, hidden_size]
                        m_input_map[LSTMInput::LSTM_INPUT_R] =
                            ngraph::op::util::convert_lstm_node_format(
                                ng_inputs.at(2),
                                ngraph::op::util::LSTMWeightsFormat::IOFC,
                                ngraph::op::util::LSTMWeightsFormat::FICO,
                                1);

                        // Get dimensions needed for default inputs creation
                        auto shape_of_x = std::make_shared<default_opset::ShapeOf>(
                            m_input_map[LSTMInput::LSTM_INPUT_X]);
                        auto axes =
                            default_opset::Constant::create(element::Type_t::i32, Shape{1}, {0});
                        auto batch_size_node = std::make_shared<default_opset::Gather>(
                            shape_of_x,
                            default_opset::Constant::create(element::Type_t::i32, Shape{1}, {0}),
                            axes);
                        auto seq_length_node = std::make_shared<default_opset::Gather>(
                            shape_of_x,
                            default_opset::Constant::create(element::Type_t::i32, Shape{1}, {1}),
                            axes);

                        auto shape_of_r = std::make_shared<default_opset::ShapeOf>(
                            m_input_map[LSTMInput::LSTM_INPUT_R]);
                        auto num_directions_node = std::make_shared<default_opset::Gather>(
                            shape_of_r,
                            default_opset::Constant::create(element::Type_t::i32, Shape{1}, {0}),
                            axes);
                        auto hidden_size_node = std::make_shared<default_opset::Gather>(
                            shape_of_r,
                            default_opset::Constant::create(element::Type_t::i32, Shape{1}, {2}),
                            axes);

                        // ------ Optional inputs ------
                        // `B` - The bias tensor for input gate.
                        // ONNX Shape: [num_directions, 8*hidden_size]
                        // OpenVino Shape: [num_directions, 4*hidden_size]
                        if (ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs.at(3)))
                        {
                            auto bias = ng_inputs.at(3);
                            auto split_bias = builder::opset1::split(bias, 2, 1);
                            NGRAPH_SUPPRESS_DEPRECATED_START
                            m_input_map[LSTMInput::LSTM_INPUT_B] =
                                std::make_shared<default_opset::Add>(split_bias.at(0),
                                                                     split_bias.at(1));
                            NGRAPH_SUPPRESS_DEPRECATED_END
                            m_input_map[LSTMInput::LSTM_INPUT_B] =
                                ngraph::op::util::convert_lstm_node_format(
                                    m_input_map[LSTMInput::LSTM_INPUT_B],
                                    ngraph::op::util::LSTMWeightsFormat::IOFC,
                                    ngraph::op::util::LSTMWeightsFormat::FICO,
                                    1);
                        }
                        else
                        {
                            auto b_shape = std::make_shared<default_opset::Concat>(
                                OutputVector{num_directions_node,
                                             std::make_shared<default_opset::Multiply>(
                                                 default_opset::Constant::create(
                                                     element::Type_t::i64, Shape{1}, {gates_count}),
                                                 hidden_size_node)},
                                0);
                            m_input_map[LSTMInput::LSTM_INPUT_B] =
                                std::make_shared<default_opset::Broadcast>(
                                    default_opset::Constant::create(
                                        m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(),
                                        Shape{},
                                        {0}),
                                    b_shape);
                        }
                        // `sequence_lens`- The lengths of the sequences in a batch.
                        // Shape: [batch_size]
                        if (ng_inputs.size() > 4 && !ngraph::op::is_null(ng_inputs.at(4)))
                        {
                            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ng_inputs.at(4);
                        }
                        else
                        {
                            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] =
                                std::make_shared<default_opset::Broadcast>(seq_length_node,
                                                                           batch_size_node);
                        }
                        // `initial_h` - The initial value of the hidden.
                        // ONNX Shape: [num_directions, batch_size, hidden_size]
                        // OpenVino Shape: [batch_size, num_directions, hidden_size]
                        if (ng_inputs.size() > 5 && !ngraph::op::is_null(ng_inputs.at(5)))
                        {
                            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] =
                                builder::opset1::reorder_axes(ng_inputs.at(5), {1, 0, 2});
                        }
                        else
                        {
                            auto init_h_shape = std::make_shared<default_opset::Concat>(
                                OutputVector{
                                    batch_size_node, num_directions_node, hidden_size_node},
                                0);
                            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] =
                                std::make_shared<default_opset::Broadcast>(
                                    default_opset::Constant::create(
                                        m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(),
                                        Shape{},
                                        {0}),
                                    init_h_shape);
                        }
                        // `initial_c` - The initial value of the cell.
                        // ONNX Shape: [num_directions, batch_size, hidden_size]
                        // OpenVino Shape: [batch_size, num_directions, hidden_size]
                        if (ng_inputs.size() > 6 && !ngraph::op::is_null(ng_inputs.at(6)))
                        {
                            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] =
                                builder::opset1::reorder_axes(ng_inputs.at(6), {1, 0, 2});
                        }
                        else
                        {
                            auto init_c_shape = std::make_shared<default_opset::Concat>(
                                OutputVector{
                                    batch_size_node, num_directions_node, hidden_size_node},
                                0);
                            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] =
                                std::make_shared<default_opset::Broadcast>(
                                    default_opset::Constant::create(
                                        m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(),
                                        Shape{},
                                        {0}),
                                    init_c_shape);
                        }
                        // `P` - The weight tensor for peepholes.
                        // Peepholes input is not supported by OpenVino
                        if (ng_inputs.size() > 7 && !ngraph::op::is_null(ng_inputs.at(7)))
                        {
                            NGRAPH_WARN
                                << (node)
                                << " Input `P` (peepholes) is not supported and will be ignored ";
                        }
                    }

                    Output<ngraph::Node>& at(const LSTMInput& key) { return m_input_map.at(key); }
                    std::map<LSTMInput, Output<ngraph::Node>> m_input_map;
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

                        m_direction =
                            ngraph::as_enum<ngraph::op::RecurrentSequenceDirection>(direction);

                        if (m_input_forget != 0)
                        {
                            NGRAPH_WARN << (node)
                                        << " Attribute `input_forget` is not supported "
                                           "and will be ignored ";
                        }
                    }

                    ngraph::op::RecurrentSequenceDirection m_direction;
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
                OutputVector lstm(const Node& node)
                {
                    LSTMNgInputMap input_map{node};
                    LSTMAttributes attributes{node};

                    auto lstm_sequence = std::make_shared<default_opset::LSTMSequence>(
                        input_map.at(LSTMInput::LSTM_INPUT_X),
                        input_map.at(LSTMInput::LSTM_INPUT_INIT_H),
                        input_map.at(LSTMInput::LSTM_INPUT_INIT_C),
                        input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                        input_map.at(LSTMInput::LSTM_INPUT_W),
                        input_map.at(LSTMInput::LSTM_INPUT_R),
                        input_map.at(LSTMInput::LSTM_INPUT_B),
                        attributes.m_hidden_size,
                        attributes.m_direction,
                        attributes.m_activation_alpha,
                        attributes.m_activation_beta,
                        attributes.m_activations,
                        attributes.m_clip_threshold);

                    const auto Y = lstm_sequence->output(0);
                    const auto Y_h = lstm_sequence->output(1);
                    const auto Y_c = lstm_sequence->output(2);

                    return {builder::opset1::reorder_axes(Y, {2, 1, 0, 3}),
                            builder::opset1::reorder_axes(Y_h, {1, 0, 2}),
                            builder::opset1::reorder_axes(Y_c, {1, 0, 2})};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
