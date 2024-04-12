// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs rsqrt(const NodeContext& node) {
    auto data = node.get_input("X");

    // Calculate the square root of the input
    auto sqrt = std::make_shared<default_opset::Power>(
        data,
        default_opset::Constant::create(data.get_element_type(), Shape{}, {0.5}));

    // Take the reciprocal of the square root
    auto rsqrt = std::make_shared<default_opset::Power>(
        sqrt,
        default_opset::Constant::create(data.get_element_type(), Shape{}, {-1.0}));

    return node.default_single_output_mapping({rsqrt}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
