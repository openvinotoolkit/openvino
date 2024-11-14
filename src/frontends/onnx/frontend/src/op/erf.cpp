// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erf.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector erf(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Erf>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Erf", OPSET_SINCE(1), ai_onnx::opset_1::erf);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
