// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
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
Output<Node> reform_weights(const Output<Node>& w,
                            int64_t n,
                            const std::vector<std::pair<int64_t, int64_t>>& intervals,
                            ov::pass::NodeRegistry& rg) {
    OutputVector slices;
    auto step = v0::Constant::create(element::i32, Shape{1}, {1});
    for (const auto& interval : intervals) {
        auto start = v0::Constant::create(element::i32, Shape{1}, {interval.first * n});
        auto stop = v0::Constant::create(element::i32, Shape{1}, {interval.second * n});
        slices.push_back(rg.make<v8::Slice>(w, start, stop, step));
    }
    return rg.make<v0::Concat>(slices, 0);
}

Output<Node> convert_lstm_node_format(const Output<Node>& node, ov::pass::NodeRegistry& rg) {
    const std::vector<size_t> from = {1, 3, 0, 2};
    const std::vector<size_t> to = {0, 1, 2, 3};
    size_t num_gates = 4;
    int64_t axis = 1;

    auto axis_const = rg.make<v0::Constant>(element::i64, ov::Shape{}, axis);
    OutputVector splitted_node = rg.make<v1::Split>(node, axis_const, num_gates)->outputs();
    OutputVector nodes_in_new_format(num_gates);
    for (size_t i = 0; i < num_gates; ++i) {
        nodes_in_new_format[to[from[i]]] = splitted_node[i];
    }
    return rg.make<v0::Concat>(nodes_in_new_format, axis);
}

OutputVector generic_rnn(ov::pass::NodeRegistry& rg,
                         std::string variant,
                         const Output<Node>& input,
                         const std::deque<Output<Node>>& initial_states,
                         const std::deque<Output<Node>>& all_weights,
                         bool has_biases,
                         int64_t num_layers,
                         // float dropout,
                         // bool train,
                         bool bidirectional,
                         bool batch_first,
                         const Output<Node>& batch_sizes = {}) {
    constexpr std::size_t gates_count{4};
    const auto direction =
        bidirectional ? v5::LSTMSequence::direction::BIDIRECTIONAL : v5::LSTMSequence::direction::FORWARD;
    int64_t weights_per_layer = has_biases ? 4 : 2;
    int64_t mult = bidirectional ? 2 : 1;
    FRONT_END_OP_CONVERSION_CHECK(static_cast<int64_t>(all_weights.size()) == num_layers * weights_per_layer * mult,
                                  "Unexpected length of list with weights for rnn operation.");

    auto w_hh = all_weights[1];
    auto w_hh_pshape = w_hh.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(w_hh_pshape.rank().is_static() && w_hh_pshape[1].is_static(), "");
    auto hidden_size = w_hh_pshape[1].get_length();

    auto zero = v0::Constant::create(element::i32, Shape{}, {0});
    auto zero_1d = v0::Constant::create(element::i32, Shape{1}, {0});
    auto one = v0::Constant::create(element::i32, Shape{}, {1});
    auto one_1d = v0::Constant::create(element::i32, Shape{1}, {1});
    auto two_1d = v0::Constant::create(element::i32, Shape{1}, {2});
    auto order_102 = v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});

    OutputVector h_outs;
    OutputVector c_outs;
    Output<Node> h0;
    Output<Node> c0;
    if (variant == "LSTM") {
        h0 = initial_states[0];
        c0 = initial_states[1];
    }

    Output<Node> prev_output = input;
    if (!batch_first)
        prev_output = rg.make<v1::Transpose>(prev_output, order_102);
    Output<Node> sequence_lens = batch_sizes;

    std::vector<std::pair<int64_t, int64_t>> reform_permutation;
    if (variant == "GRU") {
        // pytorch is reset, input, hidden
        // ov is      input, reset, hidden
        reform_permutation = {{1, 2}, {0, 1}, {2, 3}};
    } else if (variant == "LSTM") {
        // pytorch is input, forget, cell, output.
        // ov is      input, output, forget, cell.
        reform_permutation = {{0, 1}, {3, 4}, {1, 3}};
    }

    auto h_states = rg.make<v1::Split>(h0, zero, num_layers)->outputs();
    OutputVector c_states;
    if (variant == "LSTM") {
        c_states = rg.make<v1::Split>(c0, zero, num_layers)->outputs();
    }

    for (int64_t i = 0; i < num_layers; i++) {
        Output<Node> weight_ih;
        Output<Node> weight_hh;
        Output<Node> bias_concat;

        if (!has_biases) {
            auto num_directions_node = bidirectional ? two_1d : one_1d;
            auto hidden_size_node = v0::Constant::create(element::i32, Shape{1}, {hidden_size});
            auto gates_count_node = v0::Constant::create(element::i32, Shape{1}, {gates_count});
            auto gates_hidden = rg.make<v1::Multiply>(gates_count_node, hidden_size_node);
            auto b_shape = rg.make<v0::Concat>(OutputVector{num_directions_node, gates_hidden}, 0);
            auto zero_cl = rg.make<v1::ConvertLike>(zero, input);
            bias_concat = rg.make<v3::Broadcast>(zero_cl, b_shape);
        }

        int64_t idx = i * weights_per_layer;
        if (!bidirectional) {
            weight_ih = all_weights[idx];
            weight_hh = all_weights[idx + 1];
            if (variant == "GRU" || variant == "LSTM") {
                weight_ih = reform_weights(weight_ih, hidden_size, reform_permutation, rg);
                weight_hh = reform_weights(weight_hh, hidden_size, reform_permutation, rg);
            }
            weight_ih = rg.make<v0::Unsqueeze>(weight_ih, zero);
            weight_hh = rg.make<v0::Unsqueeze>(weight_hh, zero);
            if (has_biases) {
                auto bias_ih = all_weights[idx + 2];
                auto bias_hh = all_weights[idx + 3];
                bias_concat = rg.make<v1::Add>(bias_ih, bias_hh);
                if (variant == "GRU" || variant == "LSTM") {
                    bias_concat = reform_weights(bias_concat, hidden_size, reform_permutation, rg);
                }
                bias_concat = rg.make<v0::Unsqueeze>(bias_concat, zero);
            }
        } else {
            Output<Node> weight_ih_f;
            Output<Node> weight_hh_f;
            Output<Node> weight_ih_b;
            Output<Node> weight_hh_b;
            if (has_biases) {
                weight_ih_f = all_weights[2 * idx];
                weight_hh_f = all_weights[2 * idx + 1];
                auto bias_ih_f = all_weights[2 * idx + 2];
                auto bias_hh_f = all_weights[2 * idx + 3];
                weight_ih_b = all_weights[2 * idx + 4];
                weight_hh_b = all_weights[2 * idx + 5];
                auto bias_ih_b = all_weights[2 * idx + 6];
                auto bias_hh_b = all_weights[2 * idx + 7];
                Output<Node> bias_f = rg.make<v1::Add>(bias_ih_f, bias_hh_f);
                Output<Node> bias_b = rg.make<v1::Add>(bias_ih_b, bias_hh_b);
                if (variant == "GRU" || variant == "LSTM") {
                    bias_f = reform_weights(bias_f, hidden_size, reform_permutation, rg);
                    bias_b = reform_weights(bias_b, hidden_size, reform_permutation, rg);
                }
                bias_f = rg.make<v0::Unsqueeze>(bias_f, zero);
                bias_b = rg.make<v0::Unsqueeze>(bias_b, zero);
                bias_concat = rg.make<v0::Concat>(OutputVector{bias_f, bias_b}, 0);
            } else {
                weight_ih_f = all_weights[2 * idx];
                weight_hh_f = all_weights[2 * idx + 1];
                weight_ih_b = all_weights[2 * idx + 2];
                weight_hh_b = all_weights[2 * idx + 3];
            }
            if (variant == "GRU" || variant == "LSTM") {
                weight_ih_f = reform_weights(weight_ih_f, hidden_size, reform_permutation, rg);
                weight_hh_f = reform_weights(weight_hh_f, hidden_size, reform_permutation, rg);
                weight_ih_b = reform_weights(weight_ih_b, hidden_size, reform_permutation, rg);
                weight_hh_b = reform_weights(weight_hh_b, hidden_size, reform_permutation, rg);
            }
            weight_ih_f = rg.make<v0::Unsqueeze>(weight_ih_f, zero);
            weight_hh_f = rg.make<v0::Unsqueeze>(weight_hh_f, zero);
            weight_ih_b = rg.make<v0::Unsqueeze>(weight_ih_b, zero);
            weight_hh_b = rg.make<v0::Unsqueeze>(weight_hh_b, zero);
            weight_ih = rg.make<v0::Concat>(OutputVector{weight_ih_f, weight_ih_b}, 0);
            weight_hh = rg.make<v0::Concat>(OutputVector{weight_hh_f, weight_hh_b}, 0);
        }

        if (!sequence_lens.get_node_shared_ptr()) {
            auto shape_of_x = rg.make<v3::ShapeOf>(prev_output);
            auto axes = v0::Constant::create(element::i32, Shape{1}, {0});
            auto batch_size_node = rg.make<v8::Gather>(shape_of_x, zero_1d, axes);
            auto seq_length_node = rg.make<v8::Gather>(shape_of_x, one_1d, axes);
            sequence_lens = rg.make<v3::Broadcast>(seq_length_node, batch_size_node);
        }

        auto h_state = rg.make<v1::Transpose>(h_states[i], order_102);
        auto c_state = rg.make<v1::Transpose>(c_states[i], order_102);  // TODO: fix for non-LSTM case
        auto lstm_sequence = rg.make<v5::LSTMSequence>(prev_output,
                                                       h_state,
                                                       c_state,
                                                       sequence_lens,
                                                       convert_lstm_node_format(weight_ih, rg),
                                                       convert_lstm_node_format(weight_hh, rg),
                                                       convert_lstm_node_format(bias_concat, rg),
                                                       hidden_size,
                                                       direction);
        prev_output = lstm_sequence->output(0);

        if (bidirectional) {
            auto order = v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3});
            prev_output = rg.make<v1::Transpose>(prev_output, order);
            auto new_shape = v0::Constant::create(element::i32, Shape{3}, {0, 0, -1});
            prev_output = rg.make<v1::Reshape>(prev_output, new_shape, true);
        } else {
            prev_output = rg.make<v0::Squeeze>(prev_output, one);
        }

        h_outs.push_back(lstm_sequence->output(1));
        if (variant == "LSTM")
            c_outs.push_back(lstm_sequence->output(2));
    }
    if (!batch_first)
        prev_output = rg.make<v1::Transpose>(prev_output, order_102);
    Output<Node> h_res = rg.make<v0::Concat>(h_outs, 1);
    h_res = rg.make<v1::Transpose>(h_res, order_102);
    if (variant == "RNN" || variant == "GRU") {
        return {prev_output, h_res};
    } else if (variant == "LSTM") {
        Output<Node> c_res = rg.make<v0::Concat>(c_outs, 1);
        c_res = rg.make<v1::Transpose>(c_res, order_102);
        return {prev_output, h_res, c_res};
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported rnn variant.");
}

}  // namespace

OutputVector translate_lstm(const NodeContext& context) {
    num_inputs_check(context, 9, 9);
    ov::pass::NodeRegistry rg;
    if (context.get_input_type(3).is<type::List>()) {
        // lstm packed
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported lstm variant.");
    } else {
        // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout,
        // bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
        auto data = context.get_input(0);
        const auto hx = context.get_input(1);
        const auto params = context.get_input(2);
        const auto has_bias = context.const_input<bool>(3);
        const auto num_layers = context.const_input<int64_t>(4);
        // const auto dropout = context.const_input<float>(5); - skip
        const auto train = context.const_input<bool>(6);
        FRONT_END_OP_CONVERSION_CHECK(!train, "LSTM in train mode is not supported.");
        const auto bidirectional = context.const_input<bool>(7);
        const auto batch_first = context.const_input<bool>(8);

        auto initial_states = get_list_as_outputs(hx);
        auto all_weights = get_list_as_outputs(params);
        auto res = generic_rnn(rg,
                               "LSTM",
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

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
