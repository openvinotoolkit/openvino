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

// Helper function to translate a unary op.
//
// Parameters:
//
//    TFNodeDecoder* op                   - TF op being translated. Must have one input.
//    const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>& static_input_map
//                               - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//
//    std::function<Output<Node>(Output<Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (Output<Node> n) {
//                           return
//                           (Output<Multiply>(n,n));
//                       });
//  }
OutputVector TranslateUnaryOp(const NodeContext& op, std::function<Output<Node>(Output<Node>)> create_unary_op) {
    Output<Node> ng_input = op.get_ng_input(0);
    auto ng_node = create_unary_op(ng_input);
    if (ng_node != ng_input) {
        SetTracingInfo(op.get_name(), ng_node);
    }
    // SaveNgOp(ng_op_map, node.get_name(), ng_node);
    // return Status::OK();
    return {ng_node};
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<op::Abs>(n, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
OutputVector TranslateUnaryOp(const NodeContext& node) {
    return TranslateUnaryOp(node, [&node](Output<Node> n) {
        return ConstructNgNode<T>(node.get_name(), n);
    });
}

template OutputVector TranslateUnaryOp<Abs>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Acos>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Acosh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Asin>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Asinh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Atan>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Atanh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Ceiling>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Cos>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Cosh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Exp>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Floor>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Log>(const NodeContext& node);
template OutputVector TranslateUnaryOp<LogicalNot>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Negative>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Relu>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sigmoid>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sin>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sinh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sign>(const NodeContext& node);
template OutputVector TranslateUnaryOp<SoftPlus>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Tan>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Tanh>(const NodeContext& node);

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
