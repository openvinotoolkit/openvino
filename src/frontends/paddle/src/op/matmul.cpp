// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs matmul(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y = node.get_input("Y");
    auto alpha = node.get_attribute<float>("alpha", 1);
    auto transpose_a = node.get_attribute<bool>("transpose_X", false);
    auto transpose_b = node.get_attribute<bool>("transpose_Y", false);
    auto mm = std::make_shared<ov::opset6::MatMul>(x, y, transpose_a, transpose_b);
    if (alpha == 1) {
        return node.default_single_output_mapping({mm}, {"Out"});
    } else {
        auto alpha_node = ov::opset6::Constant::create(ov::element::f32, {1}, {alpha});
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Multiply>(mm, alpha_node)}, {"Out"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
