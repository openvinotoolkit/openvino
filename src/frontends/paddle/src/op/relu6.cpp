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
NamedOutputs relu6(const NodeContext& node) {
    auto data = node.get_input("X");
    auto threshold = node.get_attribute<float>("threshold", 6.0f);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Clamp>(data, 0.0, threshold)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
