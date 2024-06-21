// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/acos.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_7 {
ov::OutputVector acos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Acos>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Acos", VersionRange::single_version_for_all_opsets(), acos);
}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
