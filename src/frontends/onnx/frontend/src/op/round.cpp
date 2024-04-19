// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/round.hpp"

#include "openvino/op/round.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector round(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v5::Round>(node.get_ov_inputs().at(0), v5::Round::RoundMode::HALF_TO_EVEN)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
