// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "openvino/op/round.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector round(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v5::Round>(node.get_ov_inputs().at(0), v5::Round::RoundMode::HALF_TO_EVEN)};
}
static bool registered = register_translator("Round", VersionRange::single_version_for_all_opsets(), round);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
