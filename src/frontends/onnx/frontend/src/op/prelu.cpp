// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    const auto& slope = ov_inputs.at(1);
    return {std::make_shared<v0::PRelu>(data, slope)};
}

static bool registered = register_translator("PRelu", VersionRange::single_version_for_all_opsets(), prelu);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
