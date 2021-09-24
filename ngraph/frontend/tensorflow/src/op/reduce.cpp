// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateReduceOp(const NodeContext& node,
                               std::function<Output<Node>(Output<Node>, Output<Node>, const bool)> create_ng_node) {
    Output<Node> ng_input = node.get_ng_input(0);
    auto tf_keep_dims = node.get_attribute<bool>("keep_dims", false);

    std::vector<int64_t> axes;
    GetStaticInputVector(node, 1, &axes);

    Shape input_shape = ng_input.get_shape();
    size_t input_rank = input_shape.size();

    TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

    std::vector<size_t> ng_reduction_axes_vect(axes.size());
    std::transform(axes.begin(), axes.end(), ng_reduction_axes_vect.begin(), [input_rank](int idx) {
        return idx + (idx < 0 ? (int)input_rank : 0);
    });
    auto ng_reduction_axes = ConstructNgNode<opset::Constant>(node.get_name(),
                                                              element::i64,
                                                              Shape{ng_reduction_axes_vect.size()},
                                                              ng_reduction_axes_vect);

    Output<Node> ng_node = create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);

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

template OutputVector TranslateDirectReduceOp<opset::ReduceLogicalOr>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceLogicalAnd>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceMax>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceMean>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceMin>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceProd>(const NodeContext& node);
template OutputVector TranslateDirectReduceOp<opset::ReduceSum>(const NodeContext& node);
}  // namespace ngraph_bridge
}  // namespace tensorflow