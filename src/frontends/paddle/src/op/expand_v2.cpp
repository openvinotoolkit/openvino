// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs expand_v2(const NodeContext& node) {
    auto x = node.get_input("X");
    Output<Node> shape_expected_node;
    if (node.has_input("Shape")) {
        shape_expected_node = node.get_input("Shape");
    } else if (node.has_input("expand_shapes_tensor")) {
        auto inputs = node.get_ng_inputs("expand_shapes_tensor");
        ov::NodeVector node_vec;
        for (auto& input : inputs) {
            auto cast = std::make_shared<ov::opset6::Convert>(input, element::i32);
            node_vec.push_back(cast);
        }
        shape_expected_node = std::make_shared<ov::opset6::Concat>(node_vec, 0);
    } else {
        std::vector<int32_t> shape_expected;
        if (node.has_attribute("shape")) {
            shape_expected = node.get_attribute<std::vector<int32_t>>("shape");
        } else {
            throw std::runtime_error("expand: has no shape attribute");
        }
        shape_expected_node = ov::opset6::Constant::create(ov::element::i32, {shape_expected.size()}, shape_expected);
    }
    // if -1 in shape we will copy the orginal value from input
    auto zero_node = ov::opset6::Constant::create(ov::element::i32, {1}, {0});
    auto mask_node = std::make_shared<ov::opset6::Greater>(shape_expected_node, zero_node);
    auto input_shape_node = std::make_shared<ov::opset6::ShapeOf>(x, element::i32);
    auto fixed_shape_node = std::make_shared<ov::opset6::Select>(mask_node, shape_expected_node, input_shape_node);
    auto repeated_node = std::make_shared<ov::opset6::Divide>(fixed_shape_node, input_shape_node, false);

    return node.default_single_output_mapping(
        {std::make_shared<ov::opset6::Tile>(x, std::make_shared<ov::opset6::Convert>(repeated_node, element::i64))},
        {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
