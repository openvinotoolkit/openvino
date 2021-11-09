// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs expand(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> times_expected_node;
    if (node.has_ng_input("ExpandTimes")) {
        times_expected_node = node.get_ng_input("ExpandTimes");
    } else if (node.has_ng_input("expand_times_tensor")) {
        auto inputs = node.get_ng_inputs("expand_times_tensor");
        ngraph::NodeVector node_vec;
        for (auto& input : inputs) {
            auto cast = std::make_shared<ngraph::opset6::Convert>(input, element::i32);
            node_vec.push_back(cast);
        }
        times_expected_node = std::make_shared<ngraph::opset6::Concat>(node_vec, 0);
    } else {
        std::vector<int32_t> times_expected;
        if (node.has_attribute<std::vector<int32_t>>("expand_times")) {
            times_expected = node.get_attribute<std::vector<int32_t>>("expand_times");
        } else {
            throw std::runtime_error("expand: has no expand_times attribute");
        }
        times_expected_node =
            ngraph::opset6::Constant::create(ngraph::element::i32, {times_expected.size()}, times_expected);
    }

    return node.default_single_output_mapping(
        {std::make_shared<ngraph::opset6::Tile>(
            x,
            std::make_shared<ngraph::opset6::Convert>(times_expected_node, element::i64))},
        {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph