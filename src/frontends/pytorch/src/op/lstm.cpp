// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_lstm(const NodeContext& context) {
    num_inputs_check(context, 9, 9);
    const auto data = context.get_input(0);
    ov::Output<ov::Node> hx;
    ov::Output<ov::Node> params;
    bool has_biases;
    int num_layers;
    bool bidirectional;

    if (context.get_input_type(6) == element::boolean) {
        /* Data of shape either [batch, seq, features] or [seq, features, batch]
        aten::lstm(
            Tensor data
            Tuple[Tensor, Tensor] hx,   (state[0], state[1])
            List[Tensor] params,        (_flat_weights)
            bool has_biases,
            int num_layers,
            IGNORED float dropout,
            IGNORED bool train,
            bool bidirectional,
            bool batch_first
        ) -> output, hidden1, hidden2, ...
        */
        hx = context.get_input(1);
        params = context.get_input(2);
        has_biases = context.const_input<bool>(3);
        num_layers = context.const_input<int>(4);
        bidirectional = context.const_input<bool>(7);
        const auto batch_first = context.const_input<bool>(8);
    } else {
        /* Data of shape [seq, features], split seq by batch_sizes
        aten::lstm(
            Tensor data
            Tensor batch_sizes
            Tuple[Tensor, Tensor] hx,   (state[0], state[1])
            List[Tensor] params,        (_flat_weights)
            bool has_biases,
            int num_layers,
            IGNORED float dropout,
            IGNORED bool train,
            bool bidirectional
        ) -> output, hidden1, hidden2, ...
        */
        const auto batch_sizes = context.get_input(1);
        hx = context.get_input(2);
        params = context.get_input(3);
        has_biases = context.const_input<bool>(4);
        num_layers = context.const_input<int>(5);
        bidirectional = context.const_input<bool>(8);
    }
    const auto neg_one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));
    const auto zero_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {0}));
    const auto one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    const auto two_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {2}));
    const auto three_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {3}));
    const auto four_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {4}));

    const auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(data));
    const auto input_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, neg_one_1d));
    const auto input_size_i64 = context.mark_node(std::make_shared<opset10::Convert>(input_size, element::i64));

    const auto h0 = context.mark_node(std::make_shared<opset10::Gather>(hx, zero_1d));
    const auto c0 = context.mark_node(std::make_shared<opset10::Gather>(hx, one_1d));

    ov::Output<ov::Node> Wi, Wh, bi, bh, buffer, lstm_buffer, slice_start, slice_end;
    ov::Output<ov::Node> it, ft, gt, ot, ct = c0, ht = h0;
    // TODO bidirectional <backwards>, input for multilayer fix, projection matrix?
    int layer_offset = 0;
    for (int direction = 0; direction <= bidirectional; direction++) {
        for (int layer = 0; layer < num_layers; layer++) {
            slice_start = context.mark_node(std::make_shared<opset10::Broadcast>(zero_1d, one_1d));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            if (has_biases) {
                Wi = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));
                Wh = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));
                bi = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));
                bh = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));

                buffer = context.mark_node(std::make_shared<opset10::Multiply>(Wi, data));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Multiply>(Wh, ht));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Add>(lstm_buffer, buffer));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Add>(lstm_buffer, bi));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Add>(lstm_buffer, bh));
            } else {
                Wi = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));
                Wh = context.mark_node(std::make_shared<opset10::Gather>(params, layer_offset++));

                buffer = context.mark_node(std::make_shared<opset10::Multiply>(Wi, data));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Multiply>(Wh, ht));
                lstm_buffer = context.mark_node(std::make_shared<opset10::Add>(lstm_buffer, buffer));
            }
            it = context.mark_node(std::make_shared<opset10::Slice>(lstm_buffer, slice_start, slice_end));
            it = context.mark_node(std::make_shared<opset10::Sigmoid>(it));
            slice_start = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_end, input_size_i64));

            ft = context.mark_node(std::make_shared<opset10::Slice>(lstm_buffer, slice_start, slice_end));
            ft = context.mark_node(std::make_shared<opset10::Sigmoid>(ft));
            slice_start = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_end, input_size_i64));

            gt = context.mark_node(std::make_shared<opset10::Slice>(lstm_buffer, slice_start, slice_end));
            gt = context.mark_node(std::make_shared<opset10::Tanh>(gt));
            slice_start = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_end, input_size_i64));

            ot = context.mark_node(std::make_shared<opset10::Slice>(lstm_buffer, slice_start, slice_end));
            ot = context.mark_node(std::make_shared<opset10::Sigmoid>(ot));
            slice_start = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_end, input_size_i64));

            buffer = context.mark_node(std::make_shared<opset10::Multiply>(ft, ct));
            ct = context.mark_node(std::make_shared<opset10::Multiply>(it, gt));
            ct = context.mark_node(std::make_shared<opset10::Add>(buffer, ct));
            slice_start = context.mark_node(std::make_shared<opset10::Add>(slice_start, input_size_i64));
            slice_end = context.mark_node(std::make_shared<opset10::Add>(slice_end, input_size_i64));

            buffer = context.mark_node(std::make_shared<opset10::Tanh>(ct));
            ot = context.mark_node(std::make_shared<opset10::Multiply>(ot, buffer));
        }
    }
    return {it, ht, ct};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
