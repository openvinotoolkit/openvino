// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector greater_or_equal(const ov::frontend::onnx::Node& node) {
    const auto A = node.get_ov_inputs().at(0);
    const auto B = node.get_ov_inputs().at(1);

    FRONT_END_GENERAL_CHECK(A.get_element_type() != ov::element::bf16 && B.get_element_type() != ov::element::bf16,
                            "The input data bfloat16 isn't supported in opset 12");

    const auto C = std::make_shared<v1::GreaterEqual>(A, B);

    return {C};
}
ONNX_OP("GreaterOrEqual", OPSET_RANGE(1, 15), ai_onnx::opset_1::greater_or_equal);
}  // namespace opset_1

namespace opset_16 {
ov::OutputVector greater_or_equal(const ov::frontend::onnx::Node& node) {
    const auto A = node.get_ov_inputs().at(0);
    const auto B = node.get_ov_inputs().at(1);

    const auto C = std::make_shared<v1::GreaterEqual>(A, B);

    return {C};
}
ONNX_OP("GreaterOrEqual", OPSET_SINCE(16), ai_onnx::opset_16::greater_or_equal);
}  // namespace opset_16
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
