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

OutputVector translate_gather_op(const NodeContext& node) {
    auto ng_input = node.get_input(0);
    auto ng_input_indices = node.get_input(1);
    auto ng_axis = make_shared<Constant>(element::i64, Shape{}, 0);
    auto res = make_shared<Gather>(ng_input, ng_input_indices, ng_axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_gather_v2_op(const NodeContext& node) {
    auto ng_input = node.get_input(0);
    auto ng_input_coords = node.get_input(1);
    auto ng_axis = node.get_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto res = make_shared<Gather>(ng_input, ng_input_coords, ng_axis, batch_dims);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_gather_nd_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto input_indices = node.get_input(1);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto res = make_shared<GatherND>(input, input_indices, batch_dims);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov