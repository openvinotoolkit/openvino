// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/block_lstm.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_block_lstm_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 9, {"BlockLSTM"});
    auto node_name = node.get_name();

    auto seq_len_max = node.get_input(0);
    auto x = node.get_input(1);
    auto cs_prev = node.get_input(2);
    auto h_prev = node.get_input(3);
    auto weights = node.get_input(4);
    auto wci = node.get_input(5);
    auto wcf = node.get_input(6);
    auto wco = node.get_input(7);
    auto bias = node.get_input(8);

    // retrieve attributes
    auto forget_bias = node.get_attribute<float>("forget_bias", 1.0f);
    auto cell_clip = node.get_attribute<float>("cell_clip", 3.0f);
    auto use_peephole = node.get_attribute<bool>("use_peephole", false);

    TENSORFLOW_OP_VALIDATION(
        node,
        !use_peephole,
        "[TensorFlow Frontend] internal error: BlockLSTM is supported only for false use_peephole");
    TENSORFLOW_OP_VALIDATION(
        node,
        cell_clip == -1.0f,
        "[TensorFlow Frontend] internal error: BlockLSTM is supported only for cell_clip equal to -1");

    // extract hidden_size
    // we assume that this dimension will not be reshaped
    // and this is feasible assumption because it seems ridiculous to reshape in real model
    auto hidden_size = ov::Dimension::dynamic();
    auto w_shape = weights.get_partial_shape();
    auto w_rank = w_shape.rank();
    auto b_shape = bias.get_partial_shape();
    auto b_rank = b_shape.rank();
    if (w_rank.is_static()) {
        hidden_size = w_shape[1].is_static() ? w_shape[1].get_length() / 4 : ov::Dimension::dynamic();
    }
    if (b_rank.is_static()) {
        hidden_size = b_shape[0].is_static() ? b_shape[0].get_length() / 4 : hidden_size;
    }
    TENSORFLOW_OP_VALIDATION(
        node,
        hidden_size.is_static(),
        "[TensorFlow Frontend] internal error: BlockLSTM is supported only for static hidden size");

    // x has a format [timelen, batch_size, input_size]
    // retrieve input_size
    auto x_shape = std::make_shared<v3::ShapeOf>(x, element::i64);
    auto ss_start = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto ss_stop = std::make_shared<v0::Constant>(element::i64, Shape{1}, 3);
    auto ss_step = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto input_size = std::make_shared<v1::StridedSlice>(x_shape,
                                                         ss_start,
                                                         ss_stop,
                                                         ss_step,
                                                         std::vector<int64_t>{0},
                                                         std::vector<int64_t>{0});

    // retrieve the batch size
    // now x is in a format [time_len, batch_size, input_size]
    auto ss_start2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto ss_stop2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto batch_size = std::make_shared<v1::StridedSlice>(x_shape,
                                                         ss_start2,
                                                         ss_stop2,
                                                         ss_step,
                                                         std::vector<int64_t>{0},
                                                         std::vector<int64_t>{0});

    auto hidden_size_const =
        std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{hidden_size.get_length()});

    // adjust weights and bias
    // 1. reshape weights and bias to highlight channel dimension
    auto new_weight_shape = std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 4, -1});
    auto weight_reshape = std::make_shared<v1::Reshape>(weights, new_weight_shape, true);
    auto new_bias_shape = std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{4, -1});
    auto bias_reshape = std::make_shared<v1::Reshape>(bias, new_bias_shape, true);
    // 2. reorder gates icfo --> fico for both weights and biases
    auto reorder_const = std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{2, 0, 1, 3});
    auto weights_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto weights_reorder = std::make_shared<v8::Gather>(weight_reshape, reorder_const, weights_axis);
    auto bias_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto bias_reorder = std::make_shared<v8::Gather>(bias_reshape, reorder_const, bias_axis);
    // 3. shift_const value should be added to the first 1 / 4th part of the biases(f - gate : 0)
    auto shift_const = std::make_shared<v0::Constant>(element::f32, Shape{}, forget_bias);
    auto bias_split_lens = std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 3});
    auto bias_split = std::make_shared<v1::VariadicSplit>(bias_reorder, bias_axis, bias_split_lens);
    auto bias_first_shift = std::make_shared<v1::Add>(bias_split->output(0), shift_const);
    auto bias_shift = std::make_shared<v0::Concat>(OutputVector{bias_first_shift, bias_split->output(1)}, 0);
    // 4. return to the original shapes
    auto new_weight_shape2 = std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
    auto weight_reshape2 = std::make_shared<v1::Reshape>(weights_reorder, new_weight_shape2, true);
    // 5. normalize weights and bias
    auto transpose_order = std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
    auto new_bias_shape2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto weights_normalize = std::make_shared<v1::Transpose>(weight_reshape2, transpose_order);
    auto bias_normalized = std::make_shared<v1::Reshape>(bias_shift, new_bias_shape2, true);
    // 6. split weights into W and R inputs
    auto WR_split_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto WR_split_lens = std::make_shared<v0::Concat>(OutputVector{input_size, hidden_size_const}, 0);
    auto WR_split = std::make_shared<v1::VariadicSplit>(weights_normalize, WR_split_axis, WR_split_lens);
    // 7. unsqueeze weights and bias to have a dimension for a number of directions
    auto num_direct_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto W = std::make_shared<v0::Unsqueeze>(WR_split->output(0), num_direct_axis);
    auto R = std::make_shared<v0::Unsqueeze>(WR_split->output(1), num_direct_axis);
    auto B = std::make_shared<v0::Unsqueeze>(bias_normalized, num_direct_axis);

    // normalize initial hidden and cell states
    auto unsqueeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto init_hidden_state = std::make_shared<v0::Unsqueeze>(h_prev, unsqueeze_axis);
    auto init_cell_state = std::make_shared<v0::Unsqueeze>(cs_prev, unsqueeze_axis);

    // prepare sequence length input for LSTMSequence
    auto seq_len_max_adjusted = std::make_shared<v3::Broadcast>(seq_len_max, batch_size);

    // prepare input data since LSTMSequence accept it in a format [batch_size, time_len, input_size]
    auto x_order = std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto x_adjusted = std::make_shared<v1::Transpose>(x, x_order);

    // create LSTMSequence node and reconnect inputs and normalized weights and bias
    auto lstm_sequence = std::make_shared<v5::LSTMSequence>(x_adjusted,
                                                            init_hidden_state,
                                                            init_cell_state,
                                                            seq_len_max_adjusted,
                                                            W,
                                                            R,
                                                            B,
                                                            hidden_size.get_length(),
                                                            v5::LSTMSequence::direction::FORWARD);

    // adjust output of concatenated of hidden states from LSTMSequence
    // to have it in a format [time_len, batch_size, hidden_size]
    // 1. squeeze extra dimension - num_directions
    auto squeeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto squeeze_output_hidden_states = std::make_shared<v0::Squeeze>(lstm_sequence->output(0), squeeze_axis);
    // 2. transpose the output to rotate batch and time dimensions
    auto output_hidden_states_order =
        std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto output_hidden_states =
        std::make_shared<v1::Transpose>(squeeze_output_hidden_states, output_hidden_states_order)->output(0);
    output_hidden_states.set_names({node_name + ":6"});

    // for other outputs, it uses internal operation BlockLSTM
    auto block_lstm = make_shared<ov::frontend::tensorflow::BlockLSTM>(seq_len_max,
                                                                       x,
                                                                       cs_prev,
                                                                       h_prev,
                                                                       weights,
                                                                       wci,
                                                                       wcf,
                                                                       wco,
                                                                       bias,
                                                                       forget_bias,
                                                                       cell_clip,
                                                                       use_peephole,
                                                                       node.get_decoder());
    ov::OutputVector results = block_lstm->outputs();
    results[6] = output_hidden_states;

    return results;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
