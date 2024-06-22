// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector abs(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Abs op is not supported");
    return {std::make_shared<ov::op::v0::Abs>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Abs", VersionRange{1, 5}, abs);
}  // namespace opset_1

namespace opset_6 {
static bool registered = register_translator("Abs", VersionRange{6, 12}, ai_onnx::opset_1::abs);
}  // namespace opset_6

namespace opset_13 {
static bool registered = register_translator("Abs", VersionRange::since(13), ai_onnx::opset_1::abs);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
