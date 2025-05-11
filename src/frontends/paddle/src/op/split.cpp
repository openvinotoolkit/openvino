// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset7.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs split(const NodeContext& node) {
    using namespace opset7;
    const auto& data = node.get_input("X");
    Output<Node> axis;
    if (node.has_input("AxisTensor")) {
        auto input = node.get_input("AxisTensor");
        auto zero_node = Constant::create(element::i32, {1}, {0});
        axis = std::make_shared<ReduceMin>(input, zero_node, false);
    } else {
        auto dim = -1;
        if (node.has_attribute("axis")) {
            dim = node.get_attribute<int32_t>("axis");
        }
        axis = std::make_shared<Constant>(ov::element::i32, Shape{}, dim);
    }
    auto num_or_sections = node.get_attribute<int32_t>("num");
    NamedOutputs named_outputs;
    std::vector<Output<Node>> split_outputs;
    if (num_or_sections == 0) {
        Output<Node> sections_node;
        if (node.has_input("SectionsTensorList")) {
            auto inputs = node.get_ng_inputs("SectionsTensorList");
            sections_node = std::make_shared<ov::opset7::Concat>(inputs, 0);
        } else {
            PADDLE_OP_CHECK(node, node.has_attribute("sections"), "split: num==0 && no sections is invalid.");
            auto sections = node.get_attribute<std::vector<int32_t>>("sections");
            sections_node = Constant::create(element::i32, {sections.size()}, sections);
        }
        split_outputs = std::make_shared<VariadicSplit>(data, axis, sections_node)->outputs();
    } else {
        split_outputs = std::make_shared<Split>(data, axis, num_or_sections)->outputs();
    }

    auto out_names = node.get_output_names();
    PADDLE_OP_CHECK(node, out_names.size() == 1, "Unexpected number of outputs");

    auto it = std::find(out_names.begin(), out_names.end(), "Out");
    PADDLE_OP_CHECK(node, it != out_names.end(), "Expected output not found");
    for (const auto& split_output : split_outputs) {
        named_outputs[*it].push_back(split_output);
    }
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
