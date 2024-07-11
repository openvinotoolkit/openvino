// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_xor.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector bitwise_xor(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    OPENVINO_ASSERT(inputs.size() == 2);
    return {std::make_shared<v13::BitwiseXor>(inputs[0], inputs[1])};
}
ONNX_OP("BitwiseXor", OPSET_SINCE(1), ai_onnx::opset_1::bitwise_xor);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
