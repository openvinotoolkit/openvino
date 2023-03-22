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

    // The following process is:
    // Given:
    // input_data: shape(5, 6, 7, 8, 9)
    // update_value: shape(1, 6, 5, 3)
    // operation: input_data[:, :, 2: 7, -4: -1] = update_value
    // axes = [2, 3]
    // starts = [2, -4]
    // ends = [7, -1]
    // steps = [1, 1]  (now do not support for step > 1)
    // Our process is:
    // 1. Get axes [2, 3], get shape of input [5, 6, 7, 8, 9], select dimension from shape by axes: [7, 8].
    // 2. Get starts [2, -4] and ends [3, -1]. Process minus starts and ends. starts: [2, 4], ends: [7, 7].
    // 3. Calculate padding starts and ends
    //    1. Create `padding_starts` filled with 0. Update `starts` to `padding_starts` according to axes.
    //    padding_starts[axes[i]] = starts[i] for i in axes.size
    //    padding_starts: [0, 0, 0, 0, 0] -> [0, 0, 2, 4, 0].
    //    2. Create `padding_ends` filled with 0. Update `dim - ends` to `padding_ends` according to axes.
    //    padding_ends[axes[i]] = input_shape[axes[i]] - ends[i] for i in axes.size
    //    padding_starts: [0, 0, 0, 0, 0] -> [0, 0, 0, 1, 0].
    // 4. Calculate and broadcast update_value to corresponding shape: [5, 6, 5, 3, 9].
    //    1. Calculate `end - start`: [5, 3].
    //    2. Use `ScatterNDUpdate` to get `target_value_shape` for input_shape: [5, 6, 7, 8, 9] -> [5, 6, 5, 3, 9].
    //    3. Broadcast from [1, 6, 5, 3] to [5, 6, 5, 3, 9]
    // 5. Padding update_value to input_shape according to padding_starts and padding_ends with
    //    the maximum value in corresponding data type(float or int).
    // 6. Use select to update update_value into input_data.

    // get positive starts ends and steps
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

    // 3.1 get padding starts
    padding_starts_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    padding_starts_node = std::make_shared<default_opset::ScatterNDUpdate>(padding_starts_node, axes_node, starts);

    // 3.2 get padding ends
    padding_ends_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    const auto ends_update_node = std::make_shared<default_opset::Subtract>(spec_dim_node, ends);
    padding_ends_node =
        std::make_shared<default_opset::ScatterNDUpdate>(padding_ends_node, axes_node, ends_update_node);

    // 4.get target value shape
    // 4.1 end - start
    const auto value_shape_update_node = std::make_shared<default_opset::Subtract>(ends, starts);
    // 4.2 update
    const auto value_target_shape =
        std::make_shared<default_opset::ScatterNDUpdate>(input_shape, axes_node, value_shape_update_node);

    // 4.3 broadcast
    value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape);

    const auto maximum_value = get_max_value_by_dtype(dtype);

    // 5. pad with maximum_value
    const auto padded_value = std::make_shared<default_opset::Pad>(value_node,
                                                                   padding_starts_node,
                                                                   padding_ends_node,
                                                                   maximum_value,
                                                                   ngraph::op::PadMode::CONSTANT);

    const auto value_mask = std::make_shared<default_opset::Equal>(padded_value, maximum_value);

    // 6. select values
    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Select>(value_mask, input_node, padded_value)},
        {"Out"});
};

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
