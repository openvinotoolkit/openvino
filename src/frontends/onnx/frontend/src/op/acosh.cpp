// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/acosh.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_9 {
ov::OutputVector acosh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v3::Acosh>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Acosh", VersionRange::single_version_for_all_opsets(), acosh);
}  // namespace set_9
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
