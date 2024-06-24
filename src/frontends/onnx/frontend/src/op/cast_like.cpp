// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/convert_like.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector cast_like(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    return {std::make_shared<v1::ConvertLike>(inputs.at(0), inputs.at(1))};
}

ONNX_OP("CastLike", OPSET_SINCE(1), ai_onnx::opset_1::cast_like);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
