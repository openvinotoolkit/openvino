// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/less.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector less(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Less>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

static bool registered = register_translator("Less", VersionRange::single_version_for_all_opsets(), less);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
