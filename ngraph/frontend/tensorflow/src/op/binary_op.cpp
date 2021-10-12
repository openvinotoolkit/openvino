// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

// Helper function to translate a binary op
// Parameters:
//
//    TFNodeDecoder* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const ov::frontend::tf::detail::TensorWrapper*>& static_input_map - the static input
//    map Builder::OpMap& ng_op_map  - The TF-to-nGraph op map. std::function<Output<Node>(Output<Node>,
//    Output<Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](Output<Node> ng_input1, Output<Node>
//         ng_input2) {
//           auto ng_diff = Output<Subtract>(input1,
//           input2);
//           return Output<Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

namespace ov {
namespace frontend {
namespace tf {
namespace op {
OutputVector TranslateBinaryOp(const NodeContext& node,
                               std::function<Output<Node>(Output<Node>&, Output<Node>&)> create_binary_op) {
    Output<Node> ng_lhs = node.get_ng_input(0), ng_rhs = node.get_ng_input(1);
    auto ng_node = create_binary_op(ng_lhs, ng_rhs);

    // TODO do we need it?
    /*    if (ng_node != ng_lhs && ng_node != ng_rhs) {
            Builder::SetTracingInfo(node.get_name(), ng_node);
        }*/
    return {ng_node};
}

OutputVector TranslateFloorDivOp(const NodeContext& node) {
    auto floordiv_fn = [&node](Output<Node> x, Output<Node> y) {
        return ConstructNgNode<Floor>(node.get_name(), ConstructNgNode<Divide>(node.get_name(), x, y));
    };
    return TranslateBinaryOp(node, floordiv_fn);
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<Add>(op,
//    static_input_map,
//    ng_op_map));
//  }
//

template <typename T>
OutputVector TranslateBinaryOp(const NodeContext& node) {
    return TranslateBinaryOp(node, [&node](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return ConstructNgNode<T>(node.get_name(), ng_lhs, ng_rhs);
    });
}

template OutputVector TranslateBinaryOp<Add>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Equal>(const NodeContext& node);
template OutputVector TranslateBinaryOp<FloorMod>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Greater>(const NodeContext& node);
template OutputVector TranslateBinaryOp<GreaterEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Less>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LessEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LogicalAnd>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LogicalOr>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Maximum>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Minimum>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Multiply>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Mod>(const NodeContext& node);
template OutputVector TranslateBinaryOp<NotEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Power>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Divide>(const NodeContext& node);
template OutputVector TranslateBinaryOp<SquaredDifference>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
