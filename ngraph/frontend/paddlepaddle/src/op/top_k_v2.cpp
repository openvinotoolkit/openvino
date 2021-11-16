// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs top_k_v2(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> k_expected_node;
    if (node.has_ng_input("K")) {
        auto k_variable = node.get_ng_input("K");
        k_expected_node = std::make_shared<ngraph::opset6::Convert>(k_variable, element::i32);
    }else {
        int32_t k_expected = node.get_attribute<int32_t>("k", 1);
        k_expected_node = ngraph::opset6::Constant::create(ngraph::element::i32, {}, {k_expected});
    }

    int64_t axis = node.get_attribute<int64_t>("axis", -1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    bool largest = node.get_attribute<bool>("largest", true);
    const element::Type& index_element_type = element::i64;
    
    auto sort_type = sorted ? ngraph::opset6::TopK::SortType::SORT_VALUES : ngraph::opset6::TopK::SortType::NONE;
    auto mode = largest ? ngraph::opset6::TopK::Mode::MAX : ngraph::opset6::TopK::Mode::MIN;

    auto node_topk = std::make_shared<ngraph::opset6::TopK>(x, k_expected_node, axis, mode, sort_type, index_element_type);

    NamedOutputs named_outputs;
    named_outputs["Out"] = {node_topk->output(0)};
    named_outputs["Indices"] = {node_topk->output(1)};
    
    return named_outputs;

}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph