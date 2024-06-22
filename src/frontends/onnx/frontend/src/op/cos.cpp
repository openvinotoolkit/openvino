// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cos.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector cos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Cos>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Cos", VersionRange::single_version_for_all_opsets(), cos);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
