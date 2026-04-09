// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/op/atan2.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs atan2(const NodeContext& node) {
    auto y = node.get_input("X1");
    auto x = node.get_input("X2");
    auto result = std::make_shared<ov::op::v17::Atan2>(y, x);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {result->output(0)};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
