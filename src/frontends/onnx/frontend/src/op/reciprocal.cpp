// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector reciprocal(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);

    auto one_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {1});
    return {std::make_shared<v1::Divide>(one_node, data)};
}

ONNX_OP("Reciprocal", OPSET_SINCE(1), ai_onnx::opset_1::reciprocal);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
