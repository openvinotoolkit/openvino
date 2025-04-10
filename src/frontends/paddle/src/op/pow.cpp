// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs pow(const NodeContext& node) {
    auto x = node.get_input("X");
    auto dtype = x.get_element_type();
    Output<Node> factor_node;
    if (node.has_input("FactorTensor")) {
        factor_node = node.get_input("FactorTensor");
        if (factor_node.get_element_type() != dtype)
            factor_node = std::make_shared<ov::opset6::Convert>(factor_node, dtype);
    } else {
        factor_node = ov::opset6::Constant::create(dtype, Shape{1}, {node.get_attribute<float>("factor")});
    }

    return node.default_single_output_mapping({std::make_shared<ov::opset6::Power>(x, factor_node)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
