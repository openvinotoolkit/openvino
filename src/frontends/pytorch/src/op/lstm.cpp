// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "helper_ops/packed_sequence.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
enum RnnVariant { LSTM, GRU, RNN, RNN_RELU, RNN_TANH };

Output<Node> convert_data_format(ov::pass::NodeRegistry& rg, RnnVariant variant, const Output<Node>& node) {
    Output<Node> res;
    switch (variant) {
    case RnnVariant::LSTM:
        res = ov::op::util::convert_lstm_node_format(node, ov::op::util::LSTMWeightsFormat::IFCO);
        break;
    case RnnVariant::GRU:
        res = ov::op::util::convert_lstm_peepholes_format(node, ov::op::util::LSTMPeepholesFormat::IFO);
        break;
    default:
        res = node;
        break;
    }
    const auto axis_const = rg.make<v0::Constant>(element::i32, ov::Shape{}, 0);
    return rg.make<v0::Unsqueeze>(res, axis_const);
}

Output<Node> format_bias(ov::pass::NodeRegistry& rg,
                         RnnVariant variant,
                         const Output<Node>& b_ih,
                         const Output<Node>& b_hh) {
    Output<Node> res;
    if (variant == RnnVariant::GRU) {
        const auto one = v0::Constant::create(element::i32, Shape{}, {1});
        const auto bias_ih = convert_data_format(rg, variant, b_ih);
        const auto bias_hh = convert_data_format(rg, variant, b_hh);
        const auto split_bias_ih = rg.make<v1::Split>(bias_ih, one, 3);
        const auto split_bias_hh = rg.make<v1::Split>(bias_hh, one, 3);
        const auto wr_z_bias = rg.make<v1::Add>(split_bias_ih->output(0), split_bias_hh->output(0));
        const auto wr_r_bias = rg.make<v1::Add>(split_bias_ih->output(1), split_bias_hh->output(1));
        // The result has shape: [num_directions, 4 * hidden_size]
        // and data layout: [ [Wb_z + Rb_z], [Wb_r + Rb_r], [Wb_h], [Rb_h], ]
        res =
            rg.make<v0::Concat>(OutputVector{wr_z_bias, wr_r_bias, split_bias_ih->output(2), split_bias_hh->output(2)},
                                1);
    } else {
        res = rg.make<v1::Add>(b_ih, b_hh);
        res = convert_data_format(rg, variant, res);
    }
    return res;
}

OutputVector generic_rnn(ov::pass::NodeRegistry& rg,
                         RnnVariant variant,
                         const Output<Node>& input,
                         const std::deque<Output<Node>>& initial_states,
                         const std::deque<Output<Node>>& all_weights,
                         bool has_biases,
                         int64_t num_layers,
                         bool bidirectional,
                         bool batch_first,
                         const Output<Node>& batch_sizes = {}) {
    std::string rnn_activation;
    if (variant == RnnVariant::RNN_RELU) {
        variant = RnnVariant::RNN;
        rnn_activation = "relu";
    } else if (variant == RnnVariant::RNN_TANH) {
        variant = RnnVariant::RNN;
        rnn_activation = "tanh";
    }
    const auto direction =
        bidirectional ? RecurrentSequenceDirection::BIDIRECTIONAL : RecurrentSequenceDirection::FORWARD;
    int64_t weights_per_layer = has_biases ? 4 : 2;
    int64_t mult = bidirectional ? 2 : 1;
    PYTORCH_OP_CONVERSION_CHECK(static_cast<int64_t>(all_weights.size()) == num_layers * weights_per_layer * mult,
                                "Unexpected length of list with weights for rnn operation.");

    const auto w_hh = all_weights[1];
    const auto& w_hh_pshape = w_hh.get_partial_shape();
    PYTORCH_OP_CONVERSION_CHECK(w_hh_pshape.rank().is_static() && w_hh_pshape[1].is_static(), "");
    const auto hidden_size = w_hh_pshape[1].get_length();

    const auto zero = v0::Constant::create(element::i32, Shape{}, {0});
    const auto zero_1d = v0::Constant::create(element::i32, Shape{1}, {0});
    const auto one = v0::Constant::create(element::i32, Shape{}, {1});
    const auto one_1d = v0::Constant::create(element::i32, Shape{1}, {1});
    const auto order_102 = v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});

    OutputVector h_outs;
    OutputVector c_outs;
    Output<Node> h0;
    Output<Node> c0;
    if (variant == RnnVariant::RNN || variant == RnnVariant::GRU) {
        h0 = initial_states[0];
    } else if (variant == RnnVariant::LSTM) {
        h0 = initial_states[0];
        c0 = initial_states[1];
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported rnn variant.");
    }

    Output<Node> prev_output = input;
    if (!batch_first)
        prev_output = rg.make<v1::Transpose>(prev_output, order_102);
    Output<Node> sequence_lens = batch_sizes;

    const auto h_states = rg.make<v1::Split>(h0, zero, num_layers)->outputs();
    OutputVector c_states;
    if (variant == RnnVariant::LSTM) {
        c_states = rg.make<v1::Split>(c0, zero, num_layers)->outputs();
    }

    Output<Node> bias_concat;
    Shape::value_type num_directions = bidirectional ? 2 : 1;
    if (!has_biases) {
        Shape::value_type gates_count = variant == RnnVariant::RNN ? 1 : 4;
        Shape::value_type gates_hidden = gates_count * static_cast<Shape::value_type>(hidden_size);
        bias_concat = rg.make<v0::Constant>(element::i32, Shape{num_directions, gates_hidden}, 0);
        bias_concat = rg.make<v1::ConvertLike>(bias_concat, input);
    }

    for (int64_t i = 0; i < num_layers; i++) {
        Output<Node> weight_ih;
        Output<Node> weight_hh;

        int64_t idx = i * weights_per_layer;
        if (!bidirectional) {
            weight_ih = convert_data_format(rg, variant, all_weights[idx]);
            weight_hh = convert_data_format(rg, variant, all_weights[idx + 1]);
            if (has_biases) {
                const auto& bias_ih = all_weights[idx + 2];
                const auto& bias_hh = all_weights[idx + 3];
                bias_concat = format_bias(rg, variant, bias_ih, bias_hh);
            }
        } else {
            Output<Node> weight_ih_f;
            Output<Node> weight_hh_f;
            Output<Node> weight_ih_b;
            Output<Node> weight_hh_b;
            if (has_biases) {
                weight_ih_f = all_weights[2 * idx];
                weight_hh_f = all_weights[2 * idx + 1];
                const auto& bias_ih_f = all_weights[2 * idx + 2];
                const auto& bias_hh_f = all_weights[2 * idx + 3];
                weight_ih_b = all_weights[2 * idx + 4];
                weight_hh_b = all_weights[2 * idx + 5];
                const auto& bias_ih_b = all_weights[2 * idx + 6];
                const auto& bias_hh_b = all_weights[2 * idx + 7];
                const auto bias_f = format_bias(rg, variant, bias_ih_f, bias_hh_f);
                const auto bias_b = format_bias(rg, variant, bias_ih_b, bias_hh_b);
                bias_concat = rg.make<v0::Concat>(OutputVector{bias_f, bias_b}, 0);
            } else {
                weight_ih_f = all_weights[2 * idx];
                weight_hh_f = all_weights[2 * idx + 1];
                weight_ih_b = all_weights[2 * idx + 2];
                weight_hh_b = all_weights[2 * idx + 3];
            }
            weight_ih_f = convert_data_format(rg, variant, weight_ih_f);
            weight_hh_f = convert_data_format(rg, variant, weight_hh_f);
            weight_ih_b = convert_data_format(rg, variant, weight_ih_b);
            weight_hh_b = convert_data_format(rg, variant, weight_hh_b);
            weight_ih = rg.make<v0::Concat>(OutputVector{weight_ih_f, weight_ih_b}, 0);
            weight_hh = rg.make<v0::Concat>(OutputVector{weight_hh_f, weight_hh_b}, 0);
        }

        const auto shape_of_x = rg.make<v3::ShapeOf>(prev_output, element::i32);
        const auto axes = v0::Constant::create(element::i32, Shape{1}, {0});
        const auto batch_size_node = rg.make<v8::Gather>(shape_of_x, zero_1d, axes);
        if (!sequence_lens.get_node_shared_ptr()) {
            const auto seq_length_node = rg.make<v8::Gather>(shape_of_x, one_1d, axes);
            sequence_lens = rg.make<v3::Broadcast>(seq_length_node, batch_size_node);
        }

        const auto h_state = rg.make<v1::Transpose>(h_states[i], order_102);
        std::shared_ptr<Node> rnn_node;
        if (variant == RnnVariant::GRU) {
            rnn_node = rg.make<v5::GRUSequence>(prev_output,
                                                h_state,
                                                sequence_lens,
                                                weight_ih,
                                                weight_hh,
                                                bias_concat,
                                                hidden_size,
                                                direction,
                                                std::vector<std::string>{"sigmoid", "tanh"},
                                                std::vector<float>{},
                                                std::vector<float>{},
                                                0.f,
                                                true);
        } else if (variant == RnnVariant::LSTM) {
            Output<Node> c_state = rg.make<v1::Transpose>(c_states[i], order_102);
            rnn_node = rg.make<v5::LSTMSequence>(prev_output,
                                                 h_state,
                                                 c_state,
                                                 sequence_lens,
                                                 weight_ih,
                                                 weight_hh,
                                                 bias_concat,
                                                 hidden_size,
                                                 direction);
        } else if (variant == RnnVariant::RNN) {
            rnn_node = rg.make<v5::RNNSequence>(prev_output,
                                                h_state,
                                                sequence_lens,
                                                weight_ih,
                                                weight_hh,
                                                bias_concat,
                                                hidden_size,
                                                direction,
                                                std::vector<std::string>{rnn_activation});
        }
        prev_output = rnn_node->output(0);

        if (bidirectional) {
            const auto order = v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3});
            prev_output = rg.make<v1::Transpose>(prev_output, order);
            const auto new_shape = v0::Constant::create(element::i32, Shape{3}, {0, 0, -1});
            prev_output = rg.make<v1::Reshape>(prev_output, new_shape, true);
        } else {
            prev_output = rg.make<v0::Squeeze>(prev_output, one);
        }

        h_outs.push_back(rnn_node->output(1));
        if (variant == RnnVariant::LSTM)
            c_outs.push_back(rnn_node->output(2));
    }
    if (!batch_first)
        prev_output = rg.make<v1::Transpose>(prev_output, order_102);
    Output<Node> h_res;
    if (h_outs.size() == 1) {
        h_res = h_outs[0];
    } else {
        h_res = rg.make<v0::Concat>(h_outs, 1);
    }

    h_res = rg.make<v1::Transpose>(h_res, order_102);
    if (variant == RnnVariant::RNN || variant == RnnVariant::GRU) {
        return {prev_output, h_res};
    } else if (variant == RnnVariant::LSTM) {
        Output<Node> c_res;
        if (c_outs.size() == 1) {
            c_res = c_outs[0];
        } else {
            c_res = rg.make<v0::Concat>(c_outs, 1);
        }
        c_res = rg.make<v1::Transpose>(c_res, order_102);
        return {prev_output, h_res, c_res};
    }
    PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported rnn variant.");
}

}  // namespace

OutputVector translate_lstm(const NodeContext& context) {
    num_inputs_check(context, 9, 9);
    ov::pass::NodeRegistry rg;
    if (context.get_input_type(3).is<type::List>()) {
        // lstm packed
        // aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int
        // num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
        const auto data = context.get_input(0);
        const auto batch_sizes = context.get_input(1);
        const auto hx = context.get_input(2);
        const auto params = context.get_input(3);
        const auto has_bias = context.const_input<bool>(4);
        const auto num_layers = context.const_input<int64_t>(5);
        // const auto dropout = context.const_input<float>(6); - skip
        const auto train = context.const_input<bool>(7);
        PYTORCH_OP_CONVERSION_CHECK(!train, "LSTM in train mode is not supported.");
        const auto bidirectional = context.const_input<bool>(8);

        const auto initial_states = get_list_as_outputs(hx);
        const auto all_weights = get_list_as_outputs(params);
        const auto res = generic_rnn(rg,
                                     RnnVariant::LSTM,
                                     data,
                                     initial_states,
                                     all_weights,
                                     has_bias,
                                     num_layers,
                                     bidirectional,
                                     false,
                                     batch_sizes);
        context.mark_nodes(rg.get());
        return res;
    } else {
        // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout,
        // bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
        const auto data = context.get_input(0);
        const auto hx = context.get_input(1);
        const auto params = context.get_input(2);
        const auto has_bias = context.const_input<bool>(3);
        const auto num_layers = context.const_input<int64_t>(4);
        // const auto dropout = context.const_input<float>(5); - skip
        const auto train = context.const_input<bool>(6);
        PYTORCH_OP_CONVERSION_CHECK(!train, "LSTM in train mode is not supported.");
        const auto bidirectional = context.const_input<bool>(7);
        const auto batch_first = context.const_input<bool>(8);

        const auto initial_states = get_list_as_outputs(hx);
        const auto all_weights = get_list_as_outputs(params);
        const auto res = generic_rnn(rg,
                                     RnnVariant::LSTM,
                                     data,
                                     initial_states,
                                     all_weights,
                                     has_bias,
                                     num_layers,
                                     bidirectional,
                                     batch_first);
        context.mark_nodes(rg.get());
        return res;
    }
};

OutputVector translate_gru(const NodeContext& context) {
    num_inputs_check(context, 9, 9);
    ov::pass::NodeRegistry rg;
    // aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout,
    // bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
    const auto input = context.get_input(0);
    const auto hidden = context.get_input(1);
    const auto weight_v = context.get_input(2);
    const auto has_biases = context.const_input<bool>(3);
    const auto num_layers = context.const_input<int64_t>(4);
    // const auto dropout = context.const_input<float>(5); - skip
    const auto train = context.const_input<bool>(6);
    PYTORCH_OP_CONVERSION_CHECK(!train, "GRU in train mode is not supported.");
    const auto bidirectional = context.const_input<bool>(7);
    const auto batch_first = context.const_input<bool>(8);

    const auto weight = get_list_as_outputs(weight_v);
    const auto res =
        generic_rnn(rg, RnnVariant::GRU, input, {hidden}, weight, has_biases, num_layers, bidirectional, batch_first);
    context.mark_nodes(rg.get());
    return res;
};

namespace {
std::map<std::string, RnnVariant> RNN_VARIANT_MAP = {
    {"aten::rnn_tanh", RnnVariant::RNN_TANH},
    {"aten::rnn_relu", RnnVariant::RNN_RELU},
};
}

OutputVector translate_rnn(const NodeContext& context) {
    num_inputs_check(context, 9, 9);
    ov::pass::NodeRegistry rg;
    // aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout,
    // bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
    const auto input = context.get_input(0);
    const auto hidden = context.get_input(1);
    const auto weight_v = context.get_input(2);
    const auto has_biases = context.const_input<bool>(3);
    const auto num_layers = context.const_input<int64_t>(4);
    // const auto dropout = context.const_input<float>(5); - skip
    const auto train = context.const_input<bool>(6);
    PYTORCH_OP_CONVERSION_CHECK(!train, "RNN in train mode is not supported.");
    const auto bidirectional = context.const_input<bool>(7);
    const auto batch_first = context.const_input<bool>(8);

    const auto weight = get_list_as_outputs(weight_v);
    const auto variant_it = RNN_VARIANT_MAP.find(context.get_op_type());
    PYTORCH_OP_CONVERSION_CHECK(variant_it != RNN_VARIANT_MAP.end(), "Unsupported RNN variant.");
    const auto res = generic_rnn(rg,
                                 variant_it->second,
                                 input,
                                 {hidden},
                                 weight,
                                 has_biases,
                                 num_layers,
                                 bidirectional,
                                 batch_first);
    context.mark_nodes(rg.get());
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
