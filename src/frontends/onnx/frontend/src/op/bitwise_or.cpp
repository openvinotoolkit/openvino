// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_or.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector bitwise_or(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    OPENVINO_ASSERT(inputs.size() == 2);
    return {std::make_shared<v13::BitwiseOr>(inputs[0], inputs[1])};
}
static bool registered =
    register_translator("BitwiseOr", VersionRange::single_version_for_all_opsets(), ai_onnx::opset_1::bitwise_or);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
