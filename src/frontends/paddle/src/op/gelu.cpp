// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs gelu(const NodeContext& node) {
    const auto data = node.get_input("X");
    const auto approximate = node.get_attribute<bool>("approximate", false);
    const auto mode = approximate ? ov::op::GeluApproximationMode::TANH : ov::op::GeluApproximationMode::ERF;

    return node.default_single_output_mapping({std::make_shared<default_opset::Gelu>(data, mode)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
