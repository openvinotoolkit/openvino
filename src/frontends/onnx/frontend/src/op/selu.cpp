// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector selu(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto alpha = node.get_attribute_value<double>("alpha", 1.67326319217681884765625);
    auto gamma = node.get_attribute_value<double>("gamma", 1.05070102214813232421875);

    auto alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {alpha});

    auto gamma_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {gamma});

    return {std::make_shared<v0::Selu>(data, alpha_node, gamma_node)};
}

ONNX_OP("Selu", OPSET_SINCE(1), ai_onnx::opset_1::selu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
