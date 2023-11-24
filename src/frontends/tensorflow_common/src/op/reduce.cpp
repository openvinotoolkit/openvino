// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"

using namespace std;
using namespace ov::op;

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

template OutputVector translate_direct_reduce_op<v1::ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceSum>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v4::ReduceL2>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
