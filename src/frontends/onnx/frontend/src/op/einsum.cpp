// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/einsum.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector einsum(const ov::frontend::onnx::Node& node) {
    const std::string& equation{node.get_attribute_value<std::string>("equation")};

    return {std::make_shared<v7::Einsum>(node.get_ov_inputs(), equation)};
}

ONNX_OP("Einsum", OPSET_SINCE(1), ai_onnx::opset_1::einsum);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
