// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_basic_gather_op(const NodeContext& node, const ov::Output<ov::Node>& axis, int64_t batch_dims) {
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, op_type + " must have at least two inputs.");
    auto params = node.get_input(0);
    auto indices = node.get_input(1);
    auto gather = make_shared<v8::Gather>(params, indices, axis, batch_dims);
    set_node_name(node.get_name(), gather);
    return {gather};
}

OutputVector translate_gather_op(const NodeContext& node) {
    // Gather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is always equal to 0
    default_op_checks(node, 2, {"Gather"});
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    return translate_basic_gather_op(node, axis, 0);
}

OutputVector translate_resource_gather_op(const NodeContext& node) {
    // ResourceGather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is an attribute and can vary
    default_op_checks(node, 2, {"ResourceGather"});
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_v2_op(const NodeContext& node) {
    // GatherV2 has three inputs: data, indices, and axis by which data is sliced
    // batch_dims is an attribute and can vary
    default_op_checks(node, 3, {"GatherV2"});
    auto axis = node.get_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_nd_op(const NodeContext& node) {
    // GatherND has two inputs: data and indices
    // batch_dims is always equal to 0
    default_op_checks(node, 2, {"GatherNd", "GATHER_ND"});
    auto input = node.get_input(0);
    auto input_indices = node.get_input(1);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto gather_nd = make_shared<v8::GatherND>(input, input_indices, batch_dims);
    set_node_name(node.get_name(), gather_nd);
    return {gather_nd};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
