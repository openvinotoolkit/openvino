// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs fill_constant(const NodeContext& node) {
    auto shape = node.get_attribute<std::vector<int64_t>>("shape");
    auto dtype = node.get_attribute<ov::element::Type>("dtype");
    Output<Node> value_node;
    Output<Node> shape_node;
    if (node.has_input("ValueTensor")) {
        value_node = node.get_input("ValueTensor");
    } else if (dtype == element::i32) {
        int32_t value = static_cast<int32_t>(node.get_attribute<float>("value"));
        value_node = opset6::Constant::create(dtype, {1}, {value});
    } else if (dtype == element::f32) {
        float value = node.get_attribute<float>("value");
        value_node = opset6::Constant::create(dtype, {1}, {value});
    } else if (dtype == element::i64) {
        int64_t value = static_cast<int64_t>(node.get_attribute<float>("value"));
        value_node = opset6::Constant::create(dtype, {1}, {value});
    } else {
        PADDLE_OP_CHECK(node, false, "fill_constant only supports i32, f32, i64");
    }

    PADDLE_OP_CHECK(node,
                    shape.size() > 0 || node.has_input("ShapeTensor") || node.has_input("ShapeTensorList"),
                    "fill_constant shape not set");

    if (node.has_input("ShapeTensor")) {
        shape_node = node.get_input("ShapeTensor");
    } else if (node.has_input("ShapeTensorList")) {
        auto shape_tensor_list = node.get_ng_inputs("ShapeTensorList");
        for (size_t i = 0; i < shape_tensor_list.size(); i++) {
            auto node = shape_tensor_list[i];
            if (node.get_shape().size() == 0) {
                shape_tensor_list[i] =
                    std::make_shared<opset6::Unsqueeze>(node, opset6::Constant::create(element::i64, {1}, {0}));
            }
        }
        shape_node = Output<Node>{std::make_shared<opset6::Concat>(shape_tensor_list, 0)};
    } else {
        shape_node = opset6::Constant::create(element::i64, {shape.size()}, shape);
    }

    return node.default_single_output_mapping({std::make_shared<ov::opset6::Broadcast>(value_node, shape_node)},
                                              {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
