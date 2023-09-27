// Copyright (C) 2018-2023 Intel Corporation
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
    NamedOutputs named_outputs;

    auto axis = node.get_attribute<std::vector<int32_t>>("axis");

    auto dtype_str = node.get_attribute<std::string>("dtype");
    auto dtype = dtype_str == "int32" ? element::i32 : element::i64;

    if (axis.size() != 0) {
        auto axis_node = std::make_shared<default_opset::Constant>(element::i32, Shape{}, axis);
        outputs = std::make_shared<ov::opset10::Unique>(x, axis_node, true, dtype, dtype)->outputs();
    } else {
        outputs = std::make_shared<ov::opset10::Unique>(x, true, dtype, dtype)->outputs();
    }

    named_outputs["Index"] = {outputs[2]};
    named_outputs["Indices"] = {outputs[1]};
    named_outputs["Counts"] = {outputs[3]};
    named_outputs["Out"] = {outputs[0]};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
