// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/gru_block_cell.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_gru_block_cell_op(const ov::frontend::tensorflow::NodeContext& node) {
    // GRUBlockCell computes the GRU cell forward propagation for 1 time step
    // Inputs:
    // 0) x: Input to the GRU cell
    // 1) h_prev: State input from the previous GRU cell
    // 2) w_ru: Weight matrix for the reset and update gate
    // 3) w_c: Weight matrix for the cell connection gate
    // 4) b_ru: Bias vector for the reset and update gate
    // 5) b_c: Bias vector for the cell connection gate
    //
    // Outputs:
    // 0) r: Output of the reset gate
    // 1) u: Output of the update gate
    // 2) c: Output of the cell connection gate
    // 3) h: Current state of the GRU cell
    default_op_checks(node, 6, {"GRUBlockCell"});
    auto node_name = node.get_name();
    auto x = node.get_input(0);
    auto h_prev = node.get_input(1);
    auto w_ru = node.get_input(2);
    auto w_c = node.get_input(3);
    auto b_ru = node.get_input(4);
    auto b_c = node.get_input(5);

    // try to deduce hidden_size
    // 1. use h_prev_shape
    auto hidden_size = ov::Dimension::dynamic();
    auto h_prev_shape = h_prev.get_partial_shape();
    auto h_prev_rank = h_prev_shape.rank();
    if (h_prev_rank.is_static()) {
        hidden_size = h_prev_shape[1].is_static() ? h_prev_shape[1].get_length() : hidden_size;
    }
    auto w_ru_shape = w_ru.get_partial_shape();
    auto w_ru_rank = w_ru_shape.rank();
    if (w_ru_rank.is_static()) {
        hidden_size = w_ru_shape[1].is_static() ? w_ru_shape[1].get_length() / 2 : hidden_size;
    }
    // 3. use w_c shape
    auto w_c_shape = w_c.get_partial_shape();
    auto w_c_rank = w_c_shape.rank();
    if (w_c_rank.is_static()) {
        hidden_size = w_c_shape[1].is_static() ? w_c_shape[1].get_length() : hidden_size;
    }
    // 3. use b_ru shape
    auto b_ru_shape = b_ru.get_partial_shape();
    auto b_ru_rank = b_ru_shape.rank();
    if (b_ru_rank.is_static()) {
        hidden_size = b_ru_shape[0].is_static() ? b_ru_shape[0].get_length() / 2 : hidden_size;
    }
    // 4. use b_c shape
    auto b_c_shape = b_c.get_partial_shape();
    auto b_c_rank = b_c_shape.rank();
    if (b_c_rank.is_static()) {
        hidden_size = b_c_shape[0].is_static() ? b_c_shape[0].get_length() : hidden_size;
    }

    TENSORFLOW_OP_VALIDATION(
        node,
        hidden_size.is_static(),
        "[TensorFlow Frontend] internal error: GRUBlockCell is supported only for static hidden size");

    // retrive input_size and hidden_size
    auto x_shape = std::make_shared<v3::ShapeOf>(x, element::i64);
    auto ss_start = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto ss_end = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto ss_step = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto input_size = std::make_shared<v8::Slice>(x_shape, ss_start, ss_end, ss_step);
    auto h_prev_shape_graph = std::make_shared<v3::ShapeOf>(h_prev, element::i64);
    auto hidden_size_graph = std::make_shared<v8::Slice>(h_prev_shape_graph, ss_start, ss_end, ss_step);

    // prepare weights input
    // TensorFlow provides weights in a format w_ru and w_c, where
    // z or u - update, r - reset, c or h - hidden (connection)
    // OpenVINO GRUCell accepts weights in a format w_zrh (or w_urс)
    // 1. split w_ru into w_r and w_u
    auto split_w_ru = std::make_shared<v1::Split>(w_ru, std::make_shared<v0::Constant>(element::i64, Shape{}, 1), 2);
    // 2. concatenate different parts of weights into w_zrh (or w_urс)
    auto w_urc = std::make_shared<v0::Concat>(OutputVector{split_w_ru->output(1), split_w_ru->output(0), w_c}, 1);

    // prepare bias in the same way
    auto split_b_ru = std::make_shared<v1::Split>(b_ru, std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 2);
    auto b_urc = std::make_shared<v0::Concat>(OutputVector{split_b_ru->output(1), split_b_ru->output(0), b_c}, 0);

    // transpose weights
    // the current shape - [input_size + hidden_size, 3 * hidden_size]
    // we need the shape [3 * hidden_size, input_size + hidden_size]
    // in order to split WR into W and R
    auto transpose_order = std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
    auto w_urc_transpose = std::make_shared<v1::Transpose>(w_urc, transpose_order);

    // split combined weights WR into W and R
    auto split_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto split_nums = std::make_shared<v0::Concat>(OutputVector{input_size, hidden_size_graph}, 0);
    auto split_WR = std::make_shared<v1::VariadicSplit>(w_urc_transpose, split_axis, split_nums);

    auto gru_cell = std::make_shared<v3::GRUCell>(x,
                                                  h_prev,
                                                  split_WR->output(0),
                                                  split_WR->output(1),
                                                  b_urc,
                                                  hidden_size.get_length());
    // preserve names of the node and the output tensor
    gru_cell->output(0).set_names({node_name + ":3"});

    auto gru_block_cell_node = make_shared<GRUBlockCell>(x, h_prev, w_ru, w_c, b_ru, b_c, node.get_decoder());
    gru_block_cell_node->output(0).set_names({node_name + ":0"});
    gru_block_cell_node->output(1).set_names({node_name + ":1"});
    gru_block_cell_node->output(2).set_names({node_name + ":2"});
    ov::OutputVector results = gru_block_cell_node->outputs();
    results[3] = gru_cell->output(0);

    return results;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
