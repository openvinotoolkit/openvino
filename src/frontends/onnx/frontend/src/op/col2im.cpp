// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector col2im(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto output_shape = node.get_ov_inputs().at(1);
    
    std::vector<int64_t> kernel_shape;
    if (node.has_attribute("kernel_shape")) {
        kernel_shape = node.get_attribute_value<std::vector<int64_t>>("kernel_shape");
    } else {
        FRONT_END_GENERAL_CHECK(false, "Col2Im requires 'kernel_shape' attribute.");
    }
    const auto dilations = node.get_attribute_value<std::vector<int64_t>>("dilations", {1, 1});
    const auto pads = node.get_attribute_value<std::vector<int64_t>>("pads", {0, 0, 0, 0});
    const auto strides = node.get_attribute_value<std::vector<int64_t>>("strides", {1, 1});
    
    return std::make_shared<v10::Col2Im>(
        data, 
        output_shape, 
        kernel_shape,
        dilations,
        pads,
        strides)->outputs();
}

ONNX_OP("Col2Im", OPSET_SINCE(1), ai_onnx::opset_1::col2im);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov