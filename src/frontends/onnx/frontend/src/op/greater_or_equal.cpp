// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/greater_or_equal.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector greater_or_equal(const Node& node) {
    const auto A = node.get_ng_inputs().at(0);
    const auto B = node.get_ng_inputs().at(1);

    NGRAPH_CHECK(A.get_element_type() != ov::element::bf16 && B.get_element_type() != ov::element::bf16,
                 "The input data bfloat16 isn't supported in opset 12");

    const auto C = std::make_shared<default_opset::GreaterEqual>(A, B);

    return {C};
}
}  // namespace set_1

namespace set_16 {
OutputVector greater_or_equal(const Node& node) {
    const auto A = node.get_ng_inputs().at(0);
    const auto B = node.get_ng_inputs().at(1);

    const auto C = std::make_shared<default_opset::GreaterEqual>(A, B);

    return {C};
}
}  // namespace set_16
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
