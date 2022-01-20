// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs top_k(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> k_expected_node;
    if (node.has_ng_input("K")) {
        auto k_variable = node.get_ng_input("K");
        k_expected_node = std::make_shared<default_opset::Convert>(k_variable, element::i32);
    } else {
        int32_t k_expected;
        if (node.has_attribute<int32_t>("k")) {
            k_expected = node.get_attribute<int32_t>("k");
        } else {
            throw std::runtime_error("top_k: has no k attribute");
        }
        k_expected_node = default_opset::Constant::create(ngraph::element::i32, {}, {k_expected});
    }
    int64_t axis = -1;
    const element::Type& index_element_type = element::i64;
    auto node_topk =
        std::make_shared<default_opset::TopK>(x, k_expected_node, axis, "max", "value", index_element_type);

    NamedOutputs named_outputs;
    named_outputs["Out"] = {node_topk->output(0)};
    named_outputs["Indices"] = {node_topk->output(1)};

    return named_outputs;
}
NamedOutputs top_k_v2(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> k_expected_node;
    if (node.has_ng_input("K")) {
        auto k_variable = node.get_ng_input("K");
        k_expected_node = std::make_shared<default_opset::Convert>(k_variable, element::i32);
    } else {
        int32_t k_expected = node.get_attribute<int32_t>("k", 1);
        k_expected_node = default_opset::Constant::create(ngraph::element::i32, {}, {k_expected});
    }

    int64_t axis = node.get_attribute<int64_t>("axis", -1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    bool largest = node.get_attribute<bool>("largest", true);
    const element::Type& index_element_type = element::i64;

    auto sort_type = sorted ? default_opset::TopK::SortType::SORT_VALUES : default_opset::TopK::SortType::NONE;
    auto mode = largest ? default_opset::TopK::Mode::MAX : default_opset::TopK::Mode::MIN;

    auto node_topk =
        std::make_shared<default_opset::TopK>(x, k_expected_node, axis, mode, sort_type, index_element_type);

    NamedOutputs named_outputs;
    named_outputs["Out"] = {node_topk->output(0)};
    named_outputs["Indices"] = {node_topk->output(1)};

    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
