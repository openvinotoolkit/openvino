// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs softmax(const NodeContext& node) {
    auto data = node.get_input("X");
    auto axis = node.get_attribute<int32_t>("axis");
    if (axis < 0) {
        PADDLE_OP_CHECK(node, data.get_partial_shape().rank().is_static(), "Softmax rank must be static");
        auto data_rank = data.get_partial_shape().rank().get_length();
        axis = static_cast<uint32_t>(data_rank + axis);
    }
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Softmax>(data, axis)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
