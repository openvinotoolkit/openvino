// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/equal.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector equal(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Equal>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

static bool registered = register_translator("Equal", VersionRange::single_version_for_all_opsets(), equal);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
