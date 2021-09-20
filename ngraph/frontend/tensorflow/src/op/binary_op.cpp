// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;


// Helper function to translate a binary op
// Parameters:
//
//    TFNodeDecoder* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map - the static input
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
//           auto ng_diff = Output<opset::Subtract>(input1,
//           input2);
//           return Output<opset::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

namespace tensorflow {
namespace ngraph_bridge {
OutputVector TranslateBinaryOp(
        const NodeContext& node,
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
        return ConstructNgNode<opset::Floor>(node.get_name(), ConstructNgNode<opset::Divide>(node.get_name(), x, y));
    };
    return TranslateBinaryOp(node, floordiv_fn);
}
}
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<opset::Add>(op,
//    static_input_map,
//    ng_op_map));
//  }
//

namespace tensorflow {
namespace ngraph_bridge {

template <typename T>
OutputVector TranslateBinaryOp(const NodeContext& node) {
    return TranslateBinaryOp(node, [&node](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return ConstructNgNode<T>(node.get_name(), ng_lhs, ng_rhs);
    });
}
}
}

