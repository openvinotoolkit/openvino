// // Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

#define MAX_VALUE(T) std::numeric_limits<T>::max()

void printV(const std::string& name, std::vector<int64_t> input) {
    std::cout << name << ": ";
    for (size_t i = 0; i < input.size(); i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

void printV(const std::string& name, ov::Shape input) {
    std::cout << name << ": ";
    for (size_t i = 0; i < input.size(); i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}
namespace ov {
namespace frontend {
namespace paddle {
namespace op {

std::shared_ptr<default_opset::Constant> get_max_value_by_dtype(ov::element::Type dtype) {
    if (dtype == element::f32)
        return default_opset::Constant::create(dtype, {}, {MAX_VALUE(float)});
    else
        return default_opset::Constant::create(dtype, {}, {MAX_VALUE(int)});
};

Output<Node> handle_minus_index(const OutputVector& node, const Output<Node>& dim) {
    const auto zero = default_opset::Constant::create(node[0].get_element_type(), {1}, {0});
    const auto concat_node = std::make_shared<default_opset::Concat>(node, 0);
    const auto mask = std::make_shared<default_opset::Less>(concat_node, zero);
    const auto res = std::make_shared<default_opset::Add>(concat_node, dim);
    return std::make_shared<default_opset::Select>(mask, res, concat_node);
}

NamedOutputs set_value(const NodeContext& node) {
    const auto input_node = node.get_input("Input");
    auto value_node = node.get_input("ValueTensor");
    const auto input_shape = input_node.get_partial_shape().get_shape();
    const auto dims = static_cast<int64_t>(input_node.get_partial_shape().rank().get_length());
    const auto dtype = input_node.get_element_type();
    const auto axes = node.get_attribute<std::vector<int64_t>>("axes");

    Output<Node> padding_starts_node, padding_ends_node, value_target_shape;
    Shape value_shape(input_shape);

    if (node.has_input("StartsTensorList") && node.has_input("StepsTensorList") && node.has_input("EndsTensorList")) {
        std::vector<int64_t> spec_input_shape;
        for (size_t i =0; i < axes.size(); i++) {
            spec_input_shape.push_back(input_shape[axes[i]]);
        }
        const auto spec_dim_node = default_opset::Constant::create(element::i64, {spec_input_shape.size()}, spec_input_shape);
        auto starts = handle_minus_index(node.get_ng_inputs("StartsTensorList"), spec_dim_node);
        auto ends = handle_minus_index(node.get_ng_inputs("EndsTensorList"), spec_dim_node);
        const auto steps = node.get_ng_inputs("StepsTensorList");
        std::vector<int64_t> needed_input_dim;
        const auto input_shape_node =
            default_opset::Constant::create(element::i64, {spec_input_shape.size()}, spec_input_shape);
        const auto axes_node = default_opset::Constant::create(element::i64, {axes.size(), 1}, axes);

        // get padding starts
        padding_starts_node =
            default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
        padding_starts_node = std::make_shared<default_opset::ScatterNDUpdate>(padding_starts_node, axes_node, starts);

        // get padding ends
        padding_ends_node =
            default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
        const auto ends_update_node = std::make_shared<default_opset::Subtract>(input_shape_node, ends);
        padding_ends_node =
            std::make_shared<default_opset::ScatterNDUpdate>(padding_ends_node, axes_node, ends_update_node);

        // get target value shape
        Output<Node> value_target_shape =
            default_opset::Constant::create(element::i64, {value_shape.size()}, value_shape);
        Output<Node> value_shape_update_node = std::make_shared<default_opset::Add>(ends_update_node, starts);
        value_shape_update_node = std::make_shared<default_opset::Subtract>(input_shape_node, value_shape_update_node);
        value_target_shape =
            std::make_shared<default_opset::ScatterNDUpdate>(value_target_shape, axes_node, value_shape_update_node);
        // broadcast
        value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape);

    } else if (node.has_attribute("starts") && node.has_attribute("steps") && node.has_attribute("ends")) {
        const auto starts = node.get_attribute<std::vector<int64_t>>("starts");
        const auto ends = node.get_attribute<std::vector<int64_t>>("ends");
        const auto steps = node.get_attribute<std::vector<int64_t>>("steps");

        for (size_t i = 0; i < steps.size(); i++)
            PADDLE_OP_CHECK(node, (steps[i] == 1), "Elements of steps must be 1");

        std::vector<int64_t> padding_starts(dims), padding_ends(dims);
        for (size_t i = 0; i < starts.size(); i++) {
            int64_t s = starts[i], e = ends[i], axis = axes[i], dim = input_shape[axis];
            s += s < 0 ? dim : 0;
            e += e < 0 ? dim : 0;
            padding_starts[axis] = s;
            padding_ends[axis] = dim - e;
            value_shape[axis] -= (s + padding_ends[axis]);
        }
        value_target_shape = default_opset::Constant::create(element::i64, {value_shape.size()}, value_shape);
        padding_starts_node = default_opset::Constant::create(element::i64, {value_shape.size()}, padding_starts);
        padding_ends_node = default_opset::Constant::create(element::i64, {value_shape.size()}, padding_ends);

        value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape);
    } else
        PADDLE_OP_CHECK(node, (false), "Invalid arguments!");

    const auto maximum_value = get_max_value_by_dtype(dtype);

    const auto padded_value = std::make_shared<default_opset::Pad>(value_node,
                                                                   padding_starts_node,
                                                                   padding_ends_node,
                                                                   maximum_value,
                                                                   ngraph::op::PadMode::CONSTANT);

    const auto value_mask = std::make_shared<default_opset::Equal>(padded_value, maximum_value);

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Select>(value_mask, input_node, padded_value)},
        {"Out"});
};

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
