// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs top_k_v2(const NodeContext& node) {
    auto x = node.get_input("X");
    Output<Node> k_expected_node;
    if (node.has_input("K")) {
        auto k_variable = node.get_input("K");
        auto k_var_node = std::make_shared<default_opset::Convert>(k_variable, element::i32);
        k_expected_node = std::make_shared<default_opset::Squeeze>(k_var_node);
    } else {
        const auto k_expected = node.get_attribute<int>("k", 1);
        k_expected_node = default_opset::Constant::create(element::i32, {}, {k_expected});
    }

    auto axis = node.get_attribute<int32_t>("axis", -1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    bool largest = node.get_attribute<bool>("largest", true);

    std::string sort_type = sorted ? "value" : "none";
    std::string mode = largest ? "max" : "min";

    auto node_topk = std::make_shared<default_opset::TopK>(x, k_expected_node, axis, mode, sort_type, element::i64);

    NamedOutputs named_outputs;
    named_outputs["Out"] = OutputVector{node_topk->output(0)};
    named_outputs["Indices"] = OutputVector{node_topk->output(1)};

    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
