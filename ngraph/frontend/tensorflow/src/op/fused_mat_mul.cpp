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

OutputVector TranslateFusedMatMulOp(const NodeContext& node) {
    // auto num_args = node.get_attribute<int>("num_args"); // TODO: it is unused but why?
    auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

    // Transpose arguments if requested.
    auto transpose_a = node.get_attribute<bool>("transpose_a", false);
    auto transpose_b = node.get_attribute<bool>("transpose_b", false);

    auto ng_lhs = node.get_ng_input(0), ng_rhs = node.get_ng_input(1), ng_bias = node.get_ng_input(2);

    Output<Node> ng_matmul = ConstructNgNode<MatMul>(node.get_name(), ng_lhs, ng_rhs, transpose_a, transpose_b);

    auto ng_matmul_shape = ng_matmul.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();

    if (ng_bias_shape.size() != 1) {
        throw errors::InvalidArgument("Bias argument to BiasAdd does not have one dimension");
    }

    auto ng_add = ConstructNgNode<Add>(node.get_name(), ng_matmul, ng_bias);
    if (fused_ops.size() == 1) {  // Only fusing BiasAdd
        return {ng_add};
    } else if (fused_ops.size() == 2) {  // Also has activation
        if (fused_ops[1] == "Relu") {
            return {ConstructNgNode<Relu>(node.get_name(), ng_add)};
        } else if (fused_ops[1] == "Relu6") {
            return {ConstructNgNode<Clamp>(node.get_name(), ng_add, 0, 6)};
        } else {
            throw errors::Internal("Expected activation to be Relu or Relu6 but got " + fused_ops[1]);
        }
    } else {
        // Adding this here to catch future changes in _FusedMatMul
        throw errors::Internal("Unsupported combination");
    }
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
