// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

template <typename BinaryOp>
ov::OutputVector qlinear_op(const ov::frontend::onnx::Node& node, BinaryOp binary_op) {
    common::default_op_checks(node, 7);

    const auto inputs = node.get_ov_inputs();
    auto A = inputs[0];
    auto A_scale = inputs[1];
    auto A_zero_point =
        (inputs[2].get_shape().empty()) ? v0::Constant::create(A.get_element_type(), {}, {0}) : inputs[2];

    auto B = inputs[3];
    auto B_scale = inputs[4];
    auto B_zero_point =
        (inputs[5].get_shape().empty()) ? v0::Constant::create(A.get_element_type(), {}, {0}) : inputs[5];

    auto C_scale = inputs[6];

    auto C_zero_point = inputs.size() > 7 ? inputs[7] : v0::Constant::create(C_scale.get_element_type(), {}, {0});

    CHECK_VALID_NODE(
        node,
        (A.get_element_type() == element::i8 || A.get_element_type() == element::u8) &&
            (B.get_element_type() == element::i8 || B.get_element_type() == element::u8) &&
            (A_zero_point.get_element_type() == element::i8 || A_zero_point.get_element_type() == element::u8) &&
            (B_zero_point.get_element_type() == element::i8 || B_zero_point.get_element_type() == element::u8),
        "All inputs (A, B, A_zero_point, B_zero_point) must be either int8 or uint8. Got A: ",
        A.get_element_type(),
        ", B: ",
        B.get_element_type(),
        ", A_zero_point: ",
        A_zero_point.get_element_type(),
        ", B_zero_point: ",
        B_zero_point.get_element_type());

    auto A_minus_zero_point = std::make_shared<v1::Subtract>(A, A_zero_point);
    auto B_minus_zero_point = std::make_shared<v1::Subtract>(B, B_zero_point);

    auto A_minus_zero_point_float = std::make_shared<v0::Convert>(A_minus_zero_point, A_scale.get_element_type());
    auto B_minus_zero_point_float = std::make_shared<v0::Convert>(B_minus_zero_point, B_scale.get_element_type());

    auto A_scaled = std::make_shared<v1::Multiply>(A_scale, A_minus_zero_point_float);
    auto B_scaled = std::make_shared<v1::Multiply>(B_scale, B_minus_zero_point_float);

    auto result_scaled = binary_op(A_scaled, B_scaled);

    auto result_divided = std::make_shared<v1::Divide>(result_scaled, C_scale);

    auto C_zero_point_float = std::make_shared<v0::Convert>(C_zero_point, C_scale.get_element_type());

    auto C_float = std::make_shared<v1::Add>(result_divided, C_zero_point_float);
    auto C = std::make_shared<v0::Convert>(C_float, A.get_element_type());

    return ov::OutputVector{C};
}

ov::OutputVector qlinear_add(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqlinearadd
    // C = (A_scale * (A - A_zero_point) + B_scale * (B - B_zero_point))/C_scale + C_zero_point
    return qlinear_op(node, [](auto a, auto b) {
        return std::make_shared<v1::Add>(a, b);
    });
}

ov::OutputVector qlinear_mul(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqlinearmul
    // C = (A_scale * (A - A_zero_point) * B_scale * (B - B_zero_point))/C_scale + C_zero_point
    return qlinear_op(node, [](auto a, auto b) {
        return std::make_shared<v1::Multiply>(a, b);
    });
}

namespace {
ONNX_OP("QLinearAdd", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_add, MICROSOFT_DOMAIN);
}
ONNX_OP("QLinearMul", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_mul, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
