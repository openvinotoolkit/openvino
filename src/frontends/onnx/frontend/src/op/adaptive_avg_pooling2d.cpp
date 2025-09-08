// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector adaptive_avg_pooling2d(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();

    CHECK_VALID_NODE(node, num_inputs == 2, "adaptive_avg_pooling2d expects 2 input tensors. Got: ", num_inputs);

    return {std::make_shared<v8::AdaptiveAvgPool>(inputs[0], inputs[1])};
}
ONNX_OP("adaptive_avg_pool2d", OPSET_SINCE(1), ai_onnx::opset_1::adaptive_avg_pooling2d, PYTORCH_ATEN_DOMAIN);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
