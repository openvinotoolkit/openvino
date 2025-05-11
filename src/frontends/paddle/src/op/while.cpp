// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/while.hpp"

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

using namespace default_opset;

NamedOutputs while_(const NodeContext& node) {
    const auto data = node.get_ng_inputs("X");
    const auto cond = node.get_input("Condition");
    const auto sub_block = node.get_attribute<int32_t>("sub_block");
    auto outputs_info = node.get_output_port_infos("Out");

    ov::OutputVector inputs = data;
    inputs.push_back(cond);
    NamedOutputs named_outputs;
    named_outputs["Out"] = std::make_shared<ov::op::internal::While>(inputs, sub_block, outputs_info)->outputs();
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
