// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs clip(const NodeContext& node) {
    auto data = node.get_input("X");
    auto min = node.get_attribute<float>("min");
    auto max = node.get_attribute<float>("max");
    PADDLE_OP_CHECK(node, max >= min, "clip: max value must greater than min value!");

    return node.default_single_output_mapping({std::make_shared<ov::opset6::Clamp>(data, min, max)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
