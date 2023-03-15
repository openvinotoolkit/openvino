// // Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

#define MAX_VALUE(T) std::numeric_limits<T>::max()

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
    const auto zero = default_opset::Constant::create(element::i64, {1}, {0});
    const auto new_node = std::make_shared<default_opset::Concat>(node, 0);
    const auto mask = std::make_shared<default_opset::Less>(new_node, zero);
    const auto res = std::make_shared<default_opset::Add>(new_node, dim);
    return std::make_shared<default_opset::Select>(mask, res, new_node);
}

Output<Node> handle_minus_index(const std::vector<int64_t>& node, const Output<Node>& dim) {
    const auto zero = default_opset::Constant::create(element::i64, {1}, {0});
    const auto new_node = default_opset::Constant::create(element::i64, {node.size()}, node);
    const auto mask = std::make_shared<default_opset::Less>(new_node, zero);
    const auto res = std::make_shared<default_opset::Add>(new_node, dim);
    return std::make_shared<default_opset::Select>(mask, res, new_node);
}

NamedOutputs set_value(const NodeContext& node) {
    const auto input_node = node.get_input("Input");
    auto value_node = node.get_input("ValueTensor");
    PADDLE_OP_CHECK(node, (input_node.get_partial_shape().rank().is_static()), "rank must be static");
    const auto dims = static_cast<int64_t>(input_node.get_partial_shape().rank().get_length());
    const auto dtype = input_node.get_element_type();
    const auto axes = node.get_attribute<std::vector<int64_t>>("axes");

    auto input_shape = std::make_shared<default_opset::ShapeOf>(input_node);

    Output<Node> padding_starts_node, padding_ends_node, starts, ends, steps;

    // get starts ends and steps
    const auto axes_node = default_opset::Constant::create(element::i64, {axes.size(), 1}, axes);
    const auto spec_dim_node = std::make_shared<default_opset::GatherND>(input_shape, axes_node);
    if (node.has_input("StartsTensorList") && node.has_input("StepsTensorList") && node.has_input("EndsTensorList")) {
        starts = handle_minus_index(node.get_ng_inputs("StartsTensorList"), spec_dim_node);
        ends = handle_minus_index(node.get_ng_inputs("EndsTensorList"), spec_dim_node);
        steps = handle_minus_index(node.get_ng_inputs("StepsTensorList"), spec_dim_node);
    } else if (node.has_attribute("starts") && node.has_attribute("steps") && node.has_attribute("ends")) {
        starts = handle_minus_index(node.get_attribute<std::vector<int64_t>>("starts"), spec_dim_node);
        ends = handle_minus_index(node.get_attribute<std::vector<int64_t>>("ends"), spec_dim_node);
        auto step_vec = node.get_attribute<std::vector<int64_t>>("steps");
        for (size_t i = 0; i < step_vec.size(); i++)
            PADDLE_OP_CHECK(node, (step_vec[i] == 1), "Elements of steps must be 1");
        steps = handle_minus_index(step_vec, spec_dim_node);
    } else
        PADDLE_OP_CHECK(node, (false), "Invalid arguments!");

    // get padding starts
    padding_starts_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    padding_starts_node = std::make_shared<default_opset::ScatterNDUpdate>(padding_starts_node, axes_node, starts);

    // get padding ends
    padding_ends_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    const auto ends_update_node = std::make_shared<default_opset::Subtract>(spec_dim_node, ends);
    padding_ends_node =
        std::make_shared<default_opset::ScatterNDUpdate>(padding_ends_node, axes_node, ends_update_node);

    // get target value shape
    Output<Node> value_shape_update_node = std::make_shared<default_opset::Add>(ends_update_node, starts);
    value_shape_update_node = std::make_shared<default_opset::Subtract>(spec_dim_node, value_shape_update_node);
    const auto value_target_shape =
        std::make_shared<default_opset::ScatterNDUpdate>(input_shape, axes_node, value_shape_update_node);

    // broadcast
    value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape);

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
