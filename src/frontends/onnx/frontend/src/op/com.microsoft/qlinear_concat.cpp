// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector qlinear_concat(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3);

    auto inputs = node.get_ov_inputs();
    auto Y_scale = inputs[0];
    auto Y_zero_point = inputs[1];

    std::vector<std::shared_ptr<ov::Node>> dequantized_inputs;
    for (size_t i = 2; i < inputs.size(); i += 3) {
        auto X = inputs[i];
        auto X_scale = inputs[i + 1];
        auto X_zero_point = inputs[i + 2];

        auto X_minus_zero_point = std::make_shared<v1::Subtract>(X, X_zero_point);
        auto X_minus_zero_point_float = std::make_shared<v0::Convert>(X_minus_zero_point, X_scale.get_element_type());
        auto dequantized_X = std::make_shared<v1::Multiply>(X_scale, X_minus_zero_point_float);

        dequantized_inputs.push_back(dequantized_X);
    }

    auto axis = node.get_attribute_value<int64_t>("axis");
    auto concatenated =
        std::make_shared<v0::Concat>(ov::OutputVector(dequantized_inputs.begin(), dequantized_inputs.end()), axis);

    auto requantized = std::make_shared<v1::Divide>(concatenated, Y_scale);
    auto Y_zero_point_float = std::make_shared<v0::Convert>(Y_zero_point, Y_scale.get_element_type());
    auto Y_float = std::make_shared<v1::Add>(requantized, Y_zero_point_float);
    auto Y = std::make_shared<v0::Convert>(Y_float, inputs[2].get_element_type());

    return {Y};
}

ONNX_OP("QLinearConcat", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_concat, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
