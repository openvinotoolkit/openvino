// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mish(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    return {std::make_shared<v4::Mish>(data)};
}
static bool registered = register_translator("Mish", VersionRange::single_version_for_all_opsets(), mish);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
