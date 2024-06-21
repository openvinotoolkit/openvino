// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/operator_set.hpp"
#include "openvino/op/logical_or.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector logical_or(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalOr>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

static bool registered = register_translator("Or", VersionRange::single_version_for_all_opsets(), logical_or);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
