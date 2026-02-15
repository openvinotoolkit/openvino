// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define _USE_MATH_DEFINES

#include <math.h>

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector hammingwindow(const ov::frontend::onnx::Node& node) {
    const auto size = node.get_ov_inputs().at(0);
    const auto output_datatype = common::get_ov_element_type(node.get_attribute_value<int64_t>("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1) == 1;

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX HammingWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#hammingwindow
    const auto float_size = std::make_shared<v0::Convert>(size, ov::element::f32);
    const auto a_0 = std::make_shared<v1::Divide>(
        std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{25.0f}),
        std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{46.0f}));
    const auto a_1 = std::make_shared<v1::Subtract>(
        std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f}),
        a_0);

    const auto start = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.0f});
    const auto one_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f});
    const auto two_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{2.0f});
    const auto range = std::make_shared<v4::Range>(start, size, one_const, ov::element::f32);
    const auto pi = v0::Constant::create(ov::element::f32, ov::Shape(), std::vector<float>{static_cast<float>(M_PI)});
    std::shared_ptr<ov::Node> factor;
    if (periodic) {
        factor = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, two_const), float_size));
    } else {
        factor = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, two_const),
                                         std::make_shared<v1::Subtract>(float_size, one_const)));
    }

    const auto cos = std::make_shared<v0::Cos>(factor);
    const auto scaled_cos = std::make_shared<v1::Multiply>(cos, a_1);
    const auto y_values = std::make_shared<v1::Subtract>(a_0, scaled_cos);
    if (output_datatype == ov::element::f32) {
        return {y_values};
    } else {
        return {std::make_shared<v0::Convert>(y_values, output_datatype)};
    }
}
ONNX_OP("HammingWindow", OPSET_SINCE(1), ai_onnx::opset_1::hammingwindow);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
