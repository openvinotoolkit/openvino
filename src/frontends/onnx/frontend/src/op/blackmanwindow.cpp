// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define _USE_MATH_DEFINES

#include <math.h>

#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
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
ov::OutputVector blackmanwindow(const ov::frontend::onnx::Node& node) {
    const auto size = node.get_ov_inputs().at(0);
    const auto output_datatype = common::get_ov_element_type(node.get_attribute_value<int64_t>("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1) == 1;

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX BlackmanWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#blackmanwindow
    const auto float_size = std::make_shared<v0::Convert>(size, ov::element::f32);
    const auto a_0 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.42f});
    const auto a_1 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{-0.50f});
    const auto a_2 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.08f});

    const auto start = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.0f});
    const auto one_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f});
    const auto two_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{2.0f});
    const auto four_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{4.0f});
    const auto range = std::make_shared<v4::Range>(start, size, one_const, ov::element::f32);
    const auto pi = v0::Constant::create(ov::element::f32, ov::Shape(), std::vector<float>{static_cast<float>(M_PI)});
    std::shared_ptr<ov::Node> factor_1, factor_2;
    if (periodic) {
        factor_1 = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, two_const), float_size));
        factor_2 = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, four_const), float_size));
    } else {
        factor_1 = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, two_const),
                                         std::make_shared<v1::Subtract>(float_size, one_const)));
        factor_2 = std::make_shared<v1::Multiply>(
            range,
            std::make_shared<v1::Divide>(std::make_shared<v1::Multiply>(pi, four_const),
                                         std::make_shared<v1::Subtract>(float_size, one_const)));
    }

    const auto cos_1 = std::make_shared<v0::Cos>(factor_1);
    const auto cos_2 = std::make_shared<v0::Cos>(factor_2);
    const auto scaled_cos_1 = std::make_shared<v1::Multiply>(cos_1, a_1);
    const auto scaled_cos_2 = std::make_shared<v1::Multiply>(cos_2, a_2);
    const auto y_values = std::make_shared<v1::Add>(std::make_shared<v1::Add>(a_0, scaled_cos_1), scaled_cos_2);

    if (output_datatype == ov::element::f32) {
        return {y_values};
    } else {
        return {std::make_shared<v0::Convert>(y_values, output_datatype)};
    }
}
ONNX_OP("BlackmanWindow", OPSET_SINCE(1), ai_onnx::opset_1::blackmanwindow);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
