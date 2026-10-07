// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prelu.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector leaky_relu(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 0.01);

    std::shared_ptr<ov::Node> alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{1}, {alpha});
    return {std::make_shared<v0::PRelu>(data, alpha_node)};
}

ONNX_OP("LeakyRelu", OPSET_SINCE(1), ai_onnx::opset_1::leaky_relu);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
