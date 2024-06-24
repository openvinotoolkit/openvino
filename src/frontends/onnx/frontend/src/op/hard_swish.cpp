// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/hswish.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector hard_swish(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v4::HSwish>(node.get_ov_inputs().at(0))};
}
ONNX_OP("HardSwish", OPSET_SINCE(1), ai_onnx::opset_1::hard_swish);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
