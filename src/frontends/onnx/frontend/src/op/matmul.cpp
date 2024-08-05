// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
ov::OutputVector matmul(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    return {std::make_shared<ov::op::v0::MatMul>(a, b)};
}
}  // namespace detail
namespace opset_1 {
ov::OutputVector matmul(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::MatMul>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}
ONNX_OP("MatMul", OPSET_SINCE(1), ai_onnx::opset_1::matmul);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
