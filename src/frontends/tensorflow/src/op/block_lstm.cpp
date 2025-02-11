// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/op/loop.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace {
void create_decomposed_block_lstm(const Output<Node>& x,
                                  const Output<Node>& h_init,
                                  const Output<Node>& c_init,
                                  const Output<Node>& w,
                                  const Output<Node>& r,
                                  const Output<Node>& b,
                                  const Output<Node>& seq_len_max,
                                  const element::Type& x_type,
                                  const Dimension& hidden_size,
                                  Output<Node>& hs,
                                  Output<Node>& cs) {
    // inputs:
    // x - [time_len, batch_size, input_size] shape
    // h_init - [batch_size, hidden_size] shape
    // c_init - [batch_size, hidden_size] shape
    // w - [4 * hidden_size, input_size] shape
    // r - [4 * hidden_size, input_size] shape
    // b - [4 * hidden_size] shape
    //
    // outputs:
    // hs - [time_len, batch_size, hidden_size] shape
    // cs - [time_len, batch_size, hidden_size] shape
    auto hidden_size_value = hidden_size.get_length();

    // create a body graph with LSTMCell
    auto xi_param =
        std::make_shared<v0::Parameter>(x_type,
                                        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto h_prev_param =
        std::make_shared<v0::Parameter>(x_type, ov::PartialShape{ov::Dimension::dynamic(), hidden_size_value});
    auto c_prev_param =
        std::make_shared<v0::Parameter>(x_type, ov::PartialShape{ov::Dimension::dynamic(), hidden_size_value});
    auto w_param =
        std::make_shared<v0::Parameter>(x_type, ov::PartialShape{4 * hidden_size_value, ov::Dimension::dynamic()});
    auto r_param = std::make_shared<v0::Parameter>(x_type, ov::PartialShape{4 * hidden_size_value, hidden_size_value});
    auto b_param = std::make_shared<v0::Parameter>(x_type, ov::PartialShape{4 * hidden_size_value});

    // adjust xi since it comes after slicing and slicing axis needs to be squeezed
    auto squeeze_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto xi = std::make_shared<v0::Squeeze>(xi_param, squeeze_axis);

    auto lstm_cell = std::make_shared<v4::LSTMCell>(xi,
                                                    h_prev_param,
                                                    c_prev_param,
                                                    w_param,
                                                    r_param,
                                                    b_param,
                                                    static_cast<size_t>(hidden_size_value));

    auto h = lstm_cell->output(0);
    auto c = lstm_cell->output(1);

    // unsqueeze current cell and hidden states
    // for concatenation along time dimension
    auto axis = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 0);
    auto h_concat = std::make_shared<v0::Unsqueeze>(h, axis)->output(0);
    auto c_concat = std::make_shared<v0::Unsqueeze>(c, axis)->output(0);
    auto body_condition = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);

    ov::ParameterVector body_params({xi_param, h_prev_param, c_prev_param, w_param, r_param, b_param});
    ov::OutputVector body_results({body_condition, h, c, h_concat, c_concat});
    auto lstm_body = std::make_shared<ov::Model>(body_results, body_params);

    // create Loop node and put lstm body graph inside
    // it will represent BlockLSTM operation
    auto execution_cond = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, true);
    auto seq_len_max_shape = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, 1);
    auto new_seq_len_max = std::make_shared<v1::Reshape>(seq_len_max, seq_len_max_shape, false);
    auto loop_node = std::make_shared<v5::Loop>(new_seq_len_max, execution_cond);

    loop_node->set_function(lstm_body);
    loop_node->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    // set inputs for Loop
    // x input will be sliced for each time step
    loop_node->set_sliced_input(xi_param, x, 0, 1, 1, -1, 0);
    // set back edges for cell and hidden states
    // since they are changing through timeline
    loop_node->set_merged_input(h_prev_param, h_init, h);
    loop_node->set_merged_input(c_prev_param, c_init, c);

    loop_node->set_invariant_input(w_param, w);
    loop_node->set_invariant_input(r_param, r);
    loop_node->set_invariant_input(b_param, b);

    // set external outputs for Loop node
    // concatenated cell and hidden states from all time steps
    hs = loop_node->get_concatenated_slices(h_concat, 0, 1, 1, -1, 0);
    cs = loop_node->get_concatenated_slices(c_concat, 0, 1, 1, -1, 0);

    // clarify shapes inside body graphs and on loop outputs
    loop_node->validate_and_infer_types();

    // compute time_len it is needed for further padding
    // of concatenated cell and hidden states
    auto x_shape = std::make_shared<v3::ShapeOf>(x, element::i64);
    auto ss_start = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto ss_stop = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto ss_step = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto time_len = std::make_shared<v1::StridedSlice>(x_shape,
                                                       ss_start,
                                                       ss_stop,
                                                       ss_step,
                                                       std::vector<int64_t>{0},
                                                       std::vector<int64_t>{0});

    // since seq_len_max can be less that time length
    // output tensors needs to be padded
    auto h_init_shape = std::make_shared<v3::ShapeOf>(h_init, element::i64);
    auto dummy_size = std::make_shared<v1::Subtract>(time_len, new_seq_len_max);
    auto dummy_tensor_shape = make_shared<v0::Concat>(OutputVector{dummy_size, h_init_shape}, 0);
    auto zero_element = create_same_type_const_scalar<int32_t>(x, 0);
    auto dummy_tensor = make_shared<v3::Broadcast>(zero_element, dummy_tensor_shape);
    hs = make_shared<v0::Concat>(OutputVector{hs, dummy_tensor}, 0);
    cs = make_shared<v0::Concat>(OutputVector{cs, dummy_tensor}, 0);
}
}  // namespace

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
    auto W = WR_split->output(0);
    auto R = WR_split->output(1);
    auto B = bias_normalized;

    ov::Output<ov::Node> hs, cs;
    auto x_type = x.get_element_type();
    TENSORFLOW_OP_VALIDATION(node,
                             x_type.is_static(),
                             "[TensorFlow Frontend] internal error: BlockLSTM is supported only for x of static type");
    create_decomposed_block_lstm(x, h_prev, cs_prev, W, R, B, seq_len_max, x_type, hidden_size, hs, cs);
    cs.set_names({node_name + ":1"});
    hs.set_names({node_name + ":6"});

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
    results[1] = cs;
    results[6] = hs;

    return results;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
