// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/operator_set.hpp"
#include "openvino/op/negative.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector neg(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Negative>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Neg", VersionRange::single_version_for_all_opsets(), neg);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
