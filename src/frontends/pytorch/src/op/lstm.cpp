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
    /** Data of shape either [batch, seq, features] or [seq, features, batch]
    Only this constructor is used (batch_sizes ignored since input would have to be a
    PackedSequence instead of data tensor)
    aten::lstm(
        Tensor data
        Tuple[Tensor, Tensor] hx,   (state[0], state[1])
        List[Tensor] params,        (_flat_weights)
        bool has_bias,
        int num_layers,
        IGNORED float dropout,
        IGNORED bool train,
        bool bidirectional,
        bool batch_first
    ) -> output, hidden1, hidden2
    **/
    auto data = context.get_input(0);
    const auto h_c = context.get_input(1);
    const auto params = context.get_input(2);
    const bool has_bias = context.const_input<bool>(3);
    const int64_t num_layers = context.const_input<int64_t>(4);
    const bool bidirectional = context.const_input<bool>(7);
    const bool batch_first = context.const_input<bool>(8);
    const auto direction =
        bidirectional ? opset10::LSTMSequence::direction::BIDIRECTIONAL : opset10::LSTMSequence::direction::FORWARD;

    const auto neg_one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));
    const auto zero_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {0}));
    const auto one_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    const auto four_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {4}));

    const auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(data));
    const auto num_layers_1d = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {num_layers}));
    const auto num_directions_1d =
        context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {bidirectional ? 2 : 1}));

    std::shared_ptr<ov::Node> seq_size, batch_size;
    if (!batch_first) {
        seq_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, zero_1d, zero_1d));
        batch_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, one_1d, zero_1d));
        const auto batch_first_shape =
            context.mark_node(std::make_shared<opset10::Concat>(NodeVector{batch_size, seq_size, neg_one_1d}, 0));
        data = context.mark_node(std::make_shared<opset10::Reshape>(data, batch_first_shape));
    } else {
        seq_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, one_1d, zero_1d));
        batch_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, zero_1d, zero_1d));
    }
    //  Proper way to unpack?
    auto h0 = h_c.get_node_shared_ptr()->output(0);
    auto c0 = h_c.get_node_shared_ptr()->output(1);
    const auto c0_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(c0));
    const auto hidden_size = context.mark_node(std::make_shared<opset10::Gather>(c0_shape, neg_one_1d));
    const auto h_c_shape = context.mark_node(
        std::make_shared<opset10::Concat>(NodeVector{num_layers_1d, batch_size, num_directions_1d, neg_one_1d}, 0));
    h0 = context.mark_node(std::make_shared<opset10::Reshape>(h0, h_c_shape));
    c0 = context.mark_node(std::make_shared<opset10::Reshape>(c0, h_c_shape));
    const auto seq_len = context.mark_node(std::make_shared<opset10::Broadcast>(seq_size, batch_size));

    std::shared_ptr<ov::Node> packed_output;
    ov::Output<ov::Node> hn, cn, wn, rn, bn;
    const auto four_times_hidden_size = context.mark_node(std::make_shared<opset10::Multiply>(four_1d, hidden_size));
    const auto bias_shape =
        context.mark_node(std::make_shared<opset10::Concat>(NodeVector{num_directions_1d, four_times_hidden_size}, 0));
    for (size_t i = 0; i++; i <= num_layers) {
        const auto layer_idx = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {i}));
        const auto layer_params = params.get_node_shared_ptr()->output(i);
        // https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py#L98
        wn = layer_params.get_node_shared_ptr()->output(0);
        rn = layer_params.get_node_shared_ptr()->output(1);
        if (has_bias) {
            const auto bias_weights = layer_params.get_node_shared_ptr()->output(2);
            const auto bias_matrix = layer_params.get_node_shared_ptr()->output(3);
            bn = context.mark_node(std::make_shared<opset10::Add>(bias_weights, bias_matrix));
        } else {
            bn = context.mark_node(std::make_shared<opset10::Broadcast>(zero_1d, bias_shape));
        }

        hn = context.mark_node(std::make_shared<opset10::Gather>(h0, layer_idx, zero_1d));
        cn = context.mark_node(std::make_shared<opset10::Gather>(c0, layer_idx, zero_1d));
        packed_output = context.mark_node(
            std::make_shared<opset10::LSTMSequence>(data, hn, cn, seq_len, wn, rn, bn, hidden_size, direction));
        data = packed_output->output(0);
    }
    hn = packed_output->output(1);
    cn = packed_output->output(2);

    const auto direction_times_num_layers =
        context.mark_node(std::make_shared<opset10::Multiply>(num_directions_1d, num_layers_1d));
    const auto output_h_c_shape = context.mark_node(
        std::make_shared<opset10::Concat>(NodeVector{direction_times_num_layers, batch_size, neg_one_1d}, 0));
    std::shared_ptr<ov::Node> output_shape;
    if (!batch_first) {
        output_shape =
            context.mark_node(std::make_shared<opset10::Concat>(NodeVector{seq_size, batch_size, neg_one_1d}, 0));
    } else {
        output_shape =
            context.mark_node(std::make_shared<opset10::Concat>(NodeVector{batch_size, seq_size, neg_one_1d}, 0));
    }
    data = context.mark_node(std::make_shared<opset10::Reshape>(data, output_shape));
    hn = context.mark_node(std::make_shared<opset10::Reshape>(hn, output_h_c_shape));
    cn = context.mark_node(std::make_shared<opset10::Reshape>(cn, output_h_c_shape));

    return {data, hn, cn};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
