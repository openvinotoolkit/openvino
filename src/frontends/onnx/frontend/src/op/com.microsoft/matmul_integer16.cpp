// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector matmul_integer16(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulinteger16
    // Matrix product of two 16-bit integer tensors accumulated in 32 bits: the result is UINT32 when both
    // inputs are UINT16 and INT32 otherwise.
    common::default_op_checks(node, 2, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto& a_type = a.get_element_type();
    const auto& b_type = b.get_element_type();
    CHECK_VALID_NODE(node,
                     (a_type == ov::element::i16 || a_type == ov::element::u16) &&
                         (b_type == ov::element::i16 || b_type == ov::element::u16),
                     "Unsupported input data types, expected INT16 or UINT16 but got: ",
                     a_type,
                     " and ",
                     b_type);

    const auto result_type =
        a_type == ov::element::u16 && b_type == ov::element::u16 ? ov::element::u32 : ov::element::i32;
    const auto converted_a = std::make_shared<v0::Convert>(a, result_type);
    const auto converted_b = std::make_shared<v0::Convert>(b, result_type);
    return {std::make_shared<v0::MatMul>(converted_a, converted_b)};
}

ONNX_OP("MatMulInteger16", OPSET_SINCE(1), com_microsoft::opset_1::matmul_integer16, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
