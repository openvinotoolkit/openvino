// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs hard_swish(const NodeContext& node) {
    auto data = node.get_input("X");
    if (node.has_attribute("threshold")) {
        auto threshold = node.get_attribute<float>("threshold");
        PADDLE_OP_CHECK(node,
                        std::abs(threshold - 6.0) < 0.001,
                        "hard_swish: Only threshold = 6.0 is currently supported");
    }
    if (node.has_attribute("scale")) {
        auto scale = node.get_attribute<float>("scale");
        PADDLE_OP_CHECK(node, std::abs(scale - 6.0) < 0.001, "hard_swish: Only scale = 6.0 is currently supported");
    }
    if (node.has_attribute("offset")) {
        auto offset = node.get_attribute<float>("offset");
        PADDLE_OP_CHECK(node, std::abs(offset - 3.0) < 0.001, "hard_swish: Only offset = 3.0 is currently supported");
    }
    return node.default_single_output_mapping({std::make_shared<ov::opset6::HSwish>(data)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
