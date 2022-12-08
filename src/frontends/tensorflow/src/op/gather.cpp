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
OutputVector translate_basic_gather_op(const NodeContext& node, const ov::Output<ov::Node>& axis, int64_t batch_dims) {
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, op_type + " must have at least two inputs.");
    auto input = node.get_input(0);
    auto input_indices = node.get_input(1);
    auto res = make_shared<Gather>(input, input_indices, axis, batch_dims);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_gather_op(const NodeContext& node) {
    // Gather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is always equal to 0
    auto axis = make_shared<Constant>(element::i64, Shape{}, 0);
    return translate_basic_gather_op(node, axis, 0);
}

OutputVector translate_resource_gather_op(const NodeContext& node) {
    // ResourceGather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is an attribute and can vary
    auto axis = make_shared<Constant>(element::i64, Shape{}, 0);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_v2_op(const NodeContext& node) {
    // ResourceGather has three inputs: data, indices, and axis by which data is sliced
    // batch_dims is an attribute and can vary
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 3, "GatherV2 must have at least three inputs.");
    auto axis = node.get_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_nd_op(const NodeContext& node) {
    // GatherND has two inputs: data and indices
    // batch_dims is always equal to 0
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