// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector mul_integer(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmulinteger
    // C (int32) = (A - A_zero_point) * (B - B_zero_point) with numpy broadcasting; the zero points are
    // optional scalars of the same 8-bit type as the operand and default to 0.
    common::default_op_checks(node, 3, 4);

    const auto inputs = node.get_ov_inputs();
    const auto dequantize = [&](size_t operand_index, size_t zero_point_index) -> ov::Output<ov::Node> {
        const auto& operand = inputs[operand_index];
        const auto& operand_type = operand.get_element_type();
        CHECK_VALID_NODE(
            node,
            operand_type.is_dynamic() || operand_type == ov::element::u8 || operand_type == ov::element::i8,
            "Unsupported input data type, expected INT8 or UINT8 but got: ",
            operand_type);
        ov::Output<ov::Node> result = std::make_shared<v0::Convert>(operand, ov::element::i32);
        if (common::is_input_valid(inputs, zero_point_index)) {
            const auto zero_point = std::make_shared<v0::Convert>(inputs[zero_point_index], ov::element::i32);
            result = std::make_shared<v1::Subtract>(result, zero_point);
        }
        return result;
    };

    const auto a = dequantize(0, 1);
    const auto b = dequantize(2, 3);
    return {std::make_shared<v1::Multiply>(a, b)};
}

ONNX_OP("MulInteger", OPSET_SINCE(1), com_microsoft::opset_1::mul_integer, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
