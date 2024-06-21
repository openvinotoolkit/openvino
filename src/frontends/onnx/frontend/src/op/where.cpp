// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/operator_set.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector where(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};

    return {std::make_shared<ov::op::v1::Select>(ov_inputs.at(0), ov_inputs.at(1), ov_inputs.at(2))};
}
static bool registered = register_translator("Where", VersionRange::single_version_for_all_opsets(), where);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
