// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"
#include "paddle_utils.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
namespace {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class LSTMInput {
    LSTM_INPUT_X,
    LSTM_INPUT_W,
    LSTM_INPUT_R,
    LSTM_INPUT_B,
    LSTM_INPUT_SEQ_LENGTHS,
    LSTM_INPUT_INIT_H,
    LSTM_INPUT_INIT_C,
    LSTM_INPUT_P
};

struct LSTMNgInputMap {
    explicit LSTMNgInputMap(const NodeContext& node, Output<Node>& prev_output, int layer) {
        auto input_x = reorder_axes(prev_output, {1, 0, 2});
        //[begin. end)
        auto weight_list = node.get_ng_inputs("WeightList");
        auto weight_begin = weight_list.begin();
        auto weight_end = std::next(weight_begin, weight_list.size() / 2);
        auto bias_begin = weight_end;
        int bidirect_len = node.get_attribute<bool>("is_bidirec") ? 4 : 2;
        int layer_weight_start = layer * bidirect_len;
        int layer_weight_end = bidirect_len + layer * bidirect_len;
        int layer_bias_start = layer * bidirect_len;
        int layer_bias_end = layer * bidirect_len + bidirect_len;
        OutputVector layer_input_weight;
        OutputVector layer_hidden_weight;
        OutputVector layer_weight_bias;
        OutputVector layer_hidden_bias;

        m_input_map[LSTMInput::LSTM_INPUT_X] = input_x;
        // Parsing W R B
        auto axis_const = std::make_shared<opset6::Constant>(element::i64, Shape{}, 0);
        for (int i = layer_weight_start; i < layer_weight_end; i++) {
            auto weight_node = std::next(weight_begin, i);
            if (i & 0x1)
                layer_hidden_weight.push_back(std::make_shared<opset6::Unsqueeze>(*weight_node, axis_const));
            else
                layer_input_weight.push_back(std::make_shared<opset6::Unsqueeze>(*weight_node, axis_const));
        }

        for (int i = layer_bias_start; i < layer_bias_end; i++) {
            auto weight_node = std::next(bias_begin, i);

            if (i & 0x1)
                layer_hidden_bias.push_back(std::make_shared<opset6::Unsqueeze>(*weight_node, axis_const));
            else
                layer_weight_bias.push_back(std::make_shared<opset6::Unsqueeze>(*weight_node, axis_const));
        }

        auto input_weight = std::make_shared<opset6::Concat>(layer_input_weight, 0);
        auto hidden_weight = std::make_shared<opset6::Concat>(layer_hidden_weight, 0);
        auto weight_bias = std::make_shared<opset6::Concat>(layer_weight_bias, 0);
        auto hidden_bias = std::make_shared<opset6::Concat>(layer_hidden_bias, 0);
        auto bias = std::make_shared<opset6::Add>(weight_bias, hidden_bias);
        m_input_map[LSTMInput::LSTM_INPUT_W] =
            ov::op::util::convert_lstm_node_format(input_weight,
                                                   ov::op::util::LSTMWeightsFormat::IFCO,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);
        m_input_map[LSTMInput::LSTM_INPUT_R] =
            ov::op::util::convert_lstm_node_format(hidden_weight,
                                                   ov::op::util::LSTMWeightsFormat::IFCO,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);
        m_input_map[LSTMInput::LSTM_INPUT_B] =
            ov::op::util::convert_lstm_node_format(bias,
                                                   ov::op::util::LSTMWeightsFormat::IFCO,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        // Get dimensions needed for default inputs creation
        // Parsing init hidden state
        auto shape_of_x = std::make_shared<opset6::ShapeOf>(input_x);

        auto axes = opset6::Constant::create(element::i64, Shape{1}, {0});

        auto batch_size_node =
            std::make_shared<opset6::Gather>(shape_of_x, opset6::Constant::create(element::i64, Shape{1}, {0}), axes);

        if (node.has_input("SequenceLength")) {
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = node.get_input("SequenceLength");
        } else {
            auto seq_length_node =
                std::make_shared<opset6::Gather>(shape_of_x,
                                                 opset6::Constant::create(element::i64, Shape{1}, {1}),
                                                 axes);
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] =
                std::make_shared<opset6::Broadcast>(seq_length_node, batch_size_node);
        }

        auto init_states = node.get_ng_inputs("PreState");
        // 0 for init_h, 1 for init_cell, update bidirect_len for init states
        bidirect_len = node.get_attribute<bool>("is_bidirec") ? 2 : 1;

        auto h_begin = opset6::Constant::create(element::i64, {1}, {layer * bidirect_len});
        auto h_end = opset6::Constant::create(element::i64, Shape{1}, {layer * bidirect_len + bidirect_len});
        auto c_begin = opset6::Constant::create(element::i64, {1}, {layer * bidirect_len});
        auto c_end = opset6::Constant::create(element::i64, {1}, {layer * bidirect_len + bidirect_len});

        m_input_map[LSTMInput::LSTM_INPUT_INIT_H] =
            reorder_axes(std::make_shared<opset6::StridedSlice>(init_states[0],
                                                                h_begin,
                                                                h_end,
                                                                std::vector<int64_t>{0},
                                                                std::vector<int64_t>{0}),
                         {1, 0, 2});
        m_input_map[LSTMInput::LSTM_INPUT_INIT_C] =
            reorder_axes(std::make_shared<opset6::StridedSlice>(init_states[1],
                                                                c_begin,
                                                                c_end,
                                                                std::vector<int64_t>{0},
                                                                std::vector<int64_t>{0}),
                         {1, 0, 2});
    }

    Output<ov::Node>& at(const LSTMInput& key) {
        return m_input_map.at(key);
    }

    std::map<LSTMInput, Output<ov::Node>> m_input_map;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct LSTMAttributes {
    explicit LSTMAttributes(const NodeContext& node)
        : m_direction(node.get_attribute<bool>("is_bidirec") ? ov::op::RecurrentSequenceDirection::BIDIRECTIONAL
                                                             : ov::op::RecurrentSequenceDirection::FORWARD),
          m_hidden_size(node.get_attribute<int32_t>("hidden_size")),
          m_layers(node.get_attribute<int32_t>("num_layers"))

              {};

    ov::op::RecurrentSequenceDirection m_direction;
    int32_t m_hidden_size;
    int32_t m_layers;
};
}  // namespace
NamedOutputs lstm(const NodeContext& node) {
    auto mode = node.get_attribute<std::string>("mode");
    PADDLE_OP_CHECK(node, mode == "LSTM", "RNN only support LSTM now");
    auto prev_inputs = node.get_ng_inputs("Input");
    Output<Node> prev_output = prev_inputs[0];
    LSTMAttributes attrs(node);
    OutputVector final_h;
    OutputVector final_c;
    auto axis_const = std::make_shared<opset6::Constant>(element::i64, Shape{}, 0);
    for (int i = 0; i < attrs.m_layers; i++) {
        LSTMNgInputMap input_map(node, prev_output, i);
        auto lstm_sequence = std::make_shared<opset6::LSTMSequence>(input_map.at(LSTMInput::LSTM_INPUT_X),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_INIT_H),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_INIT_C),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_W),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_R),
                                                                    input_map.at(LSTMInput::LSTM_INPUT_B),
                                                                    attrs.m_hidden_size,
                                                                    attrs.m_direction);
        prev_output = reorder_axes(lstm_sequence->output(0), {2, 0, 1, 3});
        auto out_shape = opset6::Constant::create(element::i64, Shape{3}, {0, 0, -1});
        prev_output = std::make_shared<opset6::Reshape>(prev_output, out_shape, true);

        final_h.push_back(reorder_axes(lstm_sequence->output(1), {1, 0, 2}));
        final_c.push_back(reorder_axes(lstm_sequence->output(2), {1, 0, 2}));
    }

    NamedOutputs named_outputs;
    named_outputs["Out"] = {prev_output};
    named_outputs["State"] = {std::make_shared<opset6::Concat>(final_h, 0),
                              std::make_shared<opset6::Concat>(final_c, 0)};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
