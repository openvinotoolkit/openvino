// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_split_op(const NodeContext& node) {
    auto axes = node.get_input(0);
    auto input = node.get_input(1);
    auto num_split = node.get_attribute<int64_t>("num_split");

    auto res = make_shared<Split>(input, axes, num_split);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_split_v_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto split_lengths = node.get_input(1);
    auto split_dims = node.get_input(2);

    // todo(itikhono): double check split_lengths and split_dims are in supported form here
    auto res = make_shared<VariadicSplit>(input, split_dims, split_lengths);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
