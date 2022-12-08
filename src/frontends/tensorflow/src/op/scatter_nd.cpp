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
OutputVector translate_scatter_nd_op(const NodeContext& node) {
    auto input_indices = node.get_input(0);
    auto updates = node.get_input(1);
    auto shape = node.get_input(2);

    auto input_data = make_shared<opset8::Constant>(updates.get_element_type(), Shape{1}, 0);
    auto broadcast = make_shared<opset8::Broadcast>(input_data, shape);
    auto res = make_shared<opset8::ScatterNDUpdate>(broadcast, input_indices, updates);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
