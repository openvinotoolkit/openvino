// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector mul(const Node& node) {
    return common::handle_opset6_binary_op<default_opset::Multiply>(node);
}

}  // namespace set_1

namespace set_7 {
inline OutputVector mul(const Node& node) {
    const auto& inputs = node.get_ng_inputs();
    CHECK_VALID_NODE(node, inputs.size() == 2, "Expected number of inputs: 2. Got: ", inputs.size());
    if (inputs[0].get_element_type() == element::boolean && inputs[1].get_element_type() == element::boolean) {
        return {std::make_shared<default_opset::LogicalAnd>(inputs[0], inputs[1])};
    }
    return {std::make_shared<default_opset::Multiply>(inputs[0], inputs[1])};
}

}  // namespace set_7

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
