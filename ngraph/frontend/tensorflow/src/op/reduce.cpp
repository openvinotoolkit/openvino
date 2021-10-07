// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateReduceOp(const NodeContext& node,
                               std::function<Output<Node>(Output<Node>, Output<Node>, const bool)> create_ng_node) {
    Output<Node> input = node.get_ng_input(0);
    Output<Node> reduction_axes = node.get_ng_input(0);
    auto tf_keep_dims = node.get_attribute<bool>("keep_dims", false);
    Output<Node> ng_node = create_ng_node(input, reduction_axes, tf_keep_dims);

    return {ng_node};
}

template <typename T>
OutputVector TranslateDirectReduceOp(const NodeContext& node) {
    // ensure its either an arithmetic or a logical reduction
    if (!(std::is_base_of<ngraph::op::util::ArithmeticReduction, T>::value ||
          std::is_base_of<ngraph::op::util::LogicalReduction, T>::value)) {
        throw errors::InvalidArgument("Expected node to be either a valid logical or arithmetic reduction "
                                      "type");
    }
    return TranslateReduceOp(node,
                             [&node](Output<Node> ng_input, Output<Node> ng_reduction_axes, const bool keep_dims) {
                                 return ConstructNgNode<T>(node.get_name(), ng_input, ng_reduction_axes, keep_dims);
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
}  // namespace ngraph