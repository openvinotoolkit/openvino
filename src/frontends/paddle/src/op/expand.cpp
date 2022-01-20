// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>
#include <paddlepaddle_frontend/utility.hpp>
#include "default_opset.hpp"
#include <string>
namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace default_opset;
NamedOutputs expand_ops(const NodeContext& node, std::string input1, std::string input2, std::string attr){
    auto x = node.get_ng_input("X");
    Output<Node> expected_node;
    if (node.has_ng_input(input1)) {
        expected_node = node.get_ng_input(input1);
    } else if (node.has_ng_input(input2)) {
        auto inputs = node.get_ng_inputs(input2);
        ov::NodeVector node_vec;
        for (auto& input : inputs) {
            auto cast = std::make_shared<default_opset::Convert>(input, element::i32);
            node_vec.push_back(cast);
        }
        expected_node = std::make_shared<default_opset::Concat>(node_vec, 0);
    } else {
        std::vector<int32_t> expected;
        if (node.has_attribute<std::vector<int32_t>>(attr)) {
            expected = node.get_attribute<std::vector<int32_t>>(attr);
        } else {
            throw std::runtime_error("expand: has no " + attr + " attribute");
        }
        expected_node =
            default_opset::Constant::create(element::i32, {expected.size()}, expected);
    }
    if (input1=="Shape"){
        auto zero_node = default_opset::Constant::create(element::i32, {1}, {0});
        auto mask_node = std::make_shared<default_opset::Greater>(expected_node, zero_node);
        auto input_shape_node = std::make_shared<default_opset::ShapeOf>(x, element::i32);
        auto fixed_shape_node = std::make_shared<default_opset::Select>(mask_node, expected_node, input_shape_node);
        auto repeated_node = std::make_shared<default_opset::Divide>(fixed_shape_node, input_shape_node, false);
            return node.default_single_output_mapping(
        {std::make_shared<default_opset::Tile>(x,
                                               std::make_shared<default_opset::Convert>(repeated_node, element::i64))},
        {"Out"});
    }
    else {
            return node.default_single_output_mapping(
        {std::make_shared<default_opset::Tile>(x,
                                               std::make_shared<default_opset::Convert>(expected_node, element::i64))},
        {"Out"});
    }
    // if -1 in shape we will copy the orginal value from input
}

NamedOutputs expand(const NodeContext& node_context) {
    return expand_ops(node_context, "ExpandTimes", "expand_times_tensor", "expand_times");
}

NamedOutputs expand_v2(const NodeContext& node_context) {
    return expand_ops(node_context, "Shape", "expand_shapes_tensor", "shape");
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov