// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

template <typename T>
OutputVector translate_direct_reduce_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Any", "All", "EuclideanNorm", "Max", "Mean", "Min", "Prod", "Sum"});
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto reduce_op = make_shared<T>(input, axis, keep_dims);
    set_node_name(node.get_name(), reduce_op);
    return {reduce_op};
}

template OutputVector translate_direct_reduce_op<ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceSum>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<ReduceL2>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
