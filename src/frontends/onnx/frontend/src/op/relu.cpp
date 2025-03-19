// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/relu.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector relu(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node,
        !node.has_attribute("consumed_inputs"),
        "consumed_inputs legacy attribute of Relu op is not supported");
    std::vector<ov::element::Type> unsupported_types = {
        ov::element::bf16,
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64
    };
    for (int i = 0; i < unsupported_types.size(); ++i) {
        FRONT_END_GENERAL_CHECK(
            A.get_element_type() != unsupported_types[i] && B.get_element_type() != unsupported_types[i],
            "The input data types bf16, int8, int16, int32, and int64 are not supported in opset 1"
        );
    }
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    return {std::make_shared<ov::op::v0::Relu>(ov_inputs.at(0))};
}

ONNX_OP("Relu", OPSET_RANGE(1, 5), ai_onnx::opset_1::relu);
}  // namespace opset_1

namespace opset_6 {
ov::OutputVector relu(const ov::frontend::onnx::Node& node) {
    std::vector<ov::element::Type> unsupported_types = {
        ov::element::bf16,
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64
    };
    for (int i = 0; i < unsupported_types.size(); ++i) {
        FRONT_END_GENERAL_CHECK(
            A.get_element_type() != unsupported_types[i] && B.get_element_type() != unsupported_types[i],
            "The input data types bf16, int8, int16, int32, and int64 are not supported in opset 6"
        );
    }
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    return {std::make_shared<ov::op::v0::Relu>(ov_inputs.at(0))};
}

ONNX_OP("Relu", OPSET_RANGE(6, 12), ai_onnx::opset_6::relu);
}  // namespace opset_6

namespace opset_13 {
ov::OutputVector relu(const ov::frontend::onnx::Node& node) {
    std::vector<ov::element::Type> unsupported_types = {
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64
    };
    for (int i = 0; i < unsupported_types.size(); ++i) {
        FRONT_END_GENERAL_CHECK(
            A.get_element_type() != unsupported_types[i] && B.get_element_type() != unsupported_types[i],
            "The input data types int8, int16, int32, and int64 are not supported in opset 13"
        );
    }
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    return {std::make_shared<ov::op::v0::Relu>(ov_inputs.at(0))};
}

ONNX_OP("Relu", OPSET_RANGE(13, 13), ai_onnx::opset_13::relu);
}  // namespace opset_13

namespace opset_14 {
ov::OutputVector relu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    return {std::make_shared<ov::op::v0::Relu>(ov_inputs.at(0))};
}

ONNX_OP("Relu", OPSET_SINCE(14), ai_onnx::opset_14::relu);
}  // namespace opset_14
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
