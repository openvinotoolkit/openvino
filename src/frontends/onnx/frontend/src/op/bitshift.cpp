// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector bitshift(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> input_x = node.get_ov_inputs().at(0);
    const ov::Output<ov::Node> input_y = node.get_ov_inputs().at(1);

    std::string direction = node.get_attribute_value<std::string>("direction", "");

    CHECK_VALID_NODE(node, !direction.empty(), "Required attribute 'direction' is not specified.");

    CHECK_VALID_NODE(node,
                     direction == "LEFT" || direction == "RIGHT",
                     "Only values 'LEFT' and 'RIGHT' are supported for 'direction' "
                     "attribute. Given: ",
                     direction);

    auto shift =
        std::make_shared<v1::Power>(v0::Constant::create(input_y.get_element_type(), ov::Shape{1}, {2}), input_y);

    if (direction == "RIGHT") {
        return {std::make_shared<v1::Divide>(input_x, shift)};
    } else {
        return {std::make_shared<v1::Multiply>(input_x, shift)};
    }
}

ONNX_OP("BitShift", OPSET_SINCE(1), ai_onnx::opset_1::bitshift);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
