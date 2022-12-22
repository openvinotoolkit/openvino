// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/block_lstm.hpp"

#include "ngraph/validation_util.hpp"
#include "op_table.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_block_lstm_op(const NodeContext& node) {
    default_op_checks(node, 9, {"BlockLSTM"});
    auto seq_len_max = node.get_input(0);
    auto x = node.get_input(1);
    auto cs_prev = node.get_input(2);
    auto h_prev = node.get_input(3);
    auto w = node.get_input(4);
    auto wci = node.get_input(5);
    auto wcf = node.get_input(6);
    auto wco = node.get_input(7);
    auto b = node.get_input(8);

    // retrieve attributes
    auto forget_bias = node.get_attribute<float>("forget_bias");
    auto cell_clip = node.get_attribute<float>("cell_clip");
    auto use_peephole = node.get_attribute<bool>("use_peephole");

    auto block_lstm = make_shared<ov::frontend::tensorflow::BlockLSTM>(seq_len_max,
                                                                       x,
                                                                       cs_prev,
                                                                       h_prev,
                                                                       w,
                                                                       wci,
                                                                       wcf,
                                                                       wco,
                                                                       b,
                                                                       forget_bias,
                                                                       cell_clip,
                                                                       use_peephole,
                                                                       node.get_decoder());
    set_node_name(node.get_name(), block_lstm);
    return block_lstm->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
