// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs unique(const NodeContext& node) {
    auto x = node.get_input("X");

    std::vector<Output<Node>> outputs;

    auto axis = node.get_attribute<std::vector<int32_t>>("axis");
    auto dtype = node.get_attribute<ov::element::Type>("dtype");

    if (axis.size() != 0) {
        auto axis_node = std::make_shared<default_opset::Constant>(dtype, Shape{}, axis);
        outputs = std::make_shared<ov::opset10::Unique>(x, axis_node, true, dtype, dtype)->outputs();
    } else {
        outputs = std::make_shared<ov::opset10::Unique>(x, true, dtype, dtype)->outputs();
    }

    return NamedOutputs{{"Out", {outputs[0]}},
                        {"Indices", {outputs[1]}},
                        {"Index", {outputs[2]}},
                        {"Counts", {outputs[3]}}};
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
