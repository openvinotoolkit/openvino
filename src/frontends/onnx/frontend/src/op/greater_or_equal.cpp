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

namespace ngraph {
namespace onnx_import {
namespace op {

OutputVector relu(const Node& node) {
    const auto opset_version = node.get_opset_version();

    if (opset_version >= 1 && opset_version <= 6) {
        // Handle opset 1-6
        // Provide relevant implementation and error messages
        // Example:
        FRONT_END_GENERAL_CHECK(false, "Relu op not supported in opset 1-6");
    } else if (opset_version > 6 && opset_version <= 13) {
        // Handle opset 6-13
        // Provide relevant implementation and error messages
        // Example:
        FRONT_END_GENERAL_CHECK(false, "Relu op not supported in opset 6-13");
    } else if (opset_version > 13 && opset_version <= 14) {
        // Handle opset 13-14
        // Provide relevant implementation and error messages
        // Example:
        FRONT_END_GENERAL_CHECK(false, "Relu op not supported in opset 13-14");
    } else if (opset_version > 14) {
        // Handle opset 14+
        // Provide relevant implementation and error messages
        // Example:
        FRONT_END_GENERAL_CHECK(false, "Relu op not supported in opset 14+");
    } else {
        // Handle cases where opset version is less than 1 (unsupported)
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported opset version");
    }

    // Default return, modify as needed
    return {std::make_shared<default_opset::Relu>(node.get_ng_inputs().at(0))};
}

}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph


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
