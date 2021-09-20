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

// Helper function to translate a unary op.
//
// Parameters:
//
//    TFNodeDecoder* op                   - TF op being translated. Must have one input.
//    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map
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
//                           (Output<opset::Multiply>(n,n));
//                       });
//  }
OutputVector TranslateUnaryOp(const NodeContext& op, std::function<Output<Node>(Output<Node>)> create_unary_op) {
    Output<Node> ng_input = op.get_ng_input(0);
    auto ng_node = create_unary_op(ng_input);
    if (ng_node != ng_input) {
        Builder::SetTracingInfo(op.get_name(), ng_node);
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

}  // namespace ngraph_bridge
}  // namespace tensorflow