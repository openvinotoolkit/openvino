// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs concat(const NodeContext& node) {
    auto data = node.get_ng_inputs("X");
    auto axis = node.get_attribute<int>("axis");
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Concat>(data, axis)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
