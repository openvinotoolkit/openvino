// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector floor(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Floor>(node.get_ov_inputs().at(0))};
}

static bool registered = register_translator("Floor", VersionRange::single_version_for_all_opsets(), floor);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
