// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/asin.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector asin(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Asin>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Asin", VersionRange::single_version_for_all_opsets(), asin);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
