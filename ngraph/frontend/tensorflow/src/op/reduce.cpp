// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateReduceOp(const NodeContext& node,
                               std::function<Output<Node>(Output<Node>, Output<Node>, const bool)> create_ng_node) {
    auto input = node.get_ng_input(0);
    auto reduction_axes = node.get_ng_input(1);
    auto tf_keep_dims = node.get_attribute<bool>("keep_dims", false);
    return {create_ng_node(input, reduction_axes, tf_keep_dims)};
}

template <typename T>
OutputVector TranslateDirectReduceOp(const NodeContext& node) {
    // ensure its either an arithmetic or a logical reduction
    if (!(std::is_base_of<ov::op::util::ArithmeticReduction, T>::value ||
          std::is_base_of<ov::op::util::LogicalReduction, T>::value)) {
        TF_OP_VALIDATION_CHECK(node,
                               false,
                               "Expected node to be either a valid logical or arithmetic reduction "
                               "type");
    }
    return TranslateReduceOp(node,
                             [&node](Output<Node> ng_input, Output<Node> ng_reduction_axes, const bool keep_dims) {
                                 auto res = make_shared<T>(ng_input, ng_reduction_axes, keep_dims);
                                 SetNodeNames(node.get_name(), res);
                                 return res;
                             });
}

template OutputVector TranslateDirectReduceOp<ReduceLogicalOr>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceLogicalAnd>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceMax>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceMean>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceMin>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceProd>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<ReduceSum>(const NodeContext& node);
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov