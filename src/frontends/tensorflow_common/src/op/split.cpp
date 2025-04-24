// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "common_op_table.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_split_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Split", "SPLIT"});
    auto axis = node.get_input(0);
    auto value = node.get_input(1);
    auto num_split = node.get_attribute<int64_t>("num_split");

    auto split = make_shared<v1::Split>(value, axis, num_split);
    set_node_name(node.get_name(), split);
    return split->outputs();
}

OutputVector translate_split_v_op(const NodeContext& node) {
    default_op_checks(node, 3, {"SplitV", "SPLIT_V"});
    auto value = node.get_input(0);
    auto size_splits = node.get_input(1);
    auto axis = node.get_input(2);

    auto splitv = make_shared<v1::VariadicSplit>(value, axis, size_splits);
    set_node_name(node.get_name(), splitv);
    return splitv->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
