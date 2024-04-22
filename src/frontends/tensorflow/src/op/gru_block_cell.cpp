// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/gru_block_cell.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"

using namespace std;
using namespace ov;
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
    auto x = node.get_input(0);
    auto h_prev = node.get_input(1);
    auto w_ru = node.get_input(2);
    auto w_c = node.get_input(3);
    auto b_ru = node.get_input(4);
    auto b_c = node.get_input(5);

    auto gru_block_cell_node = make_shared<GRUBlockCell>(x, h_prev, w_ru, w_c, b_ru, b_c, node.get_decoder());
    set_node_name(node.get_name(), gru_block_cell_node);
    return gru_block_cell_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
