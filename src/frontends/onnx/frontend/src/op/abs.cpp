// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/abs.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector abs(const ov::frontend::onnx::Node& node) {
    CHECK_VALID_NODE(node,
                     !node.has_attribute("consumed_inputs"),
                     "consumed_inputs legacy attribute of Abs op is not supported");
    return {std::make_shared<ov::op::v0::Abs>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Abs", VersionRange{1, 5}, abs);
}  // namespace set_1

namespace set_6 {
static bool registered = register_translator("Abs", VersionRange{6, 12}, set_1::abs);
}  // namespace set_6

namespace set_13 {
static bool registered = register_translator("Abs", VersionRange::since(13), set_1::abs);
}  // namespace set_13
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
