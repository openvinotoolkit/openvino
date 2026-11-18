// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx {
namespace {
ov::OutputVector translate_selu(const ov::frontend::onnx::Node& node, double default_alpha, double default_gamma) {
    auto data = node.get_ov_inputs().at(0);
    auto alpha = node.get_attribute_value<double>("alpha", default_alpha);
    auto gamma = node.get_attribute_value<double>("gamma", default_gamma);

    auto alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {alpha});

    auto gamma_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {gamma});

    return {std::make_shared<v0::Selu>(data, alpha_node, gamma_node)};
}

}  // namespace

namespace opset_1 {
ov::OutputVector selu(const ov::frontend::onnx::Node& node) {
    return translate_selu(node, 1.6732f, 1.0507f);
}

ONNX_OP("Selu", OPSET_RANGE(1, 5), ai_onnx::opset_1::selu);
}  // namespace opset_1

namespace opset_6 {
ov::OutputVector selu(const ov::frontend::onnx::Node& node) {
    return translate_selu(node, 1.67326319217681884765625, 1.05070102214813232421875);
}

ONNX_OP("Selu", OPSET_SINCE(6), ai_onnx::opset_6::selu);
}  // namespace opset_6
}  // namespace ov::frontend::onnx::ai_onnx
