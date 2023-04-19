// // Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

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
    auto input_node = node.get_input("Input");
    auto value_node = node.get_input("ValueTensor");
    PADDLE_OP_CHECK(node, (input_node.get_partial_shape().rank().is_static()), "rank must be static");
    const auto dims = static_cast<int64_t>(input_node.get_partial_shape().rank().get_length());
    const auto axes = node.get_attribute<std::vector<int64_t>>("axes");

    auto input_shape = std::make_shared<default_opset::ShapeOf>(input_node);

    Output<Node> starts_node, ends_node, steps_node, starts, ends, steps;

    // The following process is:
    // Given:
    // input_data: shape(5, 6, 7, 8, 9)
    // update_value: shape(1, 6, 3, 3)
    // operation: input_data[:, :, 2: 7: 2, -4: -1] = update_value
    // axes = [2, 3]
    // starts = [2, -4]
    // ends = [7, -1]
    // steps = [2, 1]
    // Our process is:
    // 1. Get axes [2, 3], get shape of input [5, 6, 7, 8, 9], select dimension from shape by axes: [7, 8].
    // 2. Get starts [2, -4] and ends [3, -1]. Process minus starts and ends. starts: [2, 4], ends: [7, 7].
    // 3. Calculate starts_node, ends_node and steps_node
    //    1. Create `starts node` filled with 0. Update `starts` to `starts_node` according to axes.
    //    starts node[axes[i]] = starts[i] for i in axes.size
    //    starts node: [0, 0, 0, 0, 0] -> [0, 0, 2, 4, 0].
    //    2. Create `ends_node` filled with -1. Update `ends` to `ends_node` according to axes.
    //    ends node[axes[i]] = ends[i] for i in axes.size
    //    ends node: [-1, -1, -1, -1, -1] -> [-1, -1, 7, -1, -1].
    //    3. Create `steps_node` filled with 1. Update `steps' to `steps_node` according to axes.
    //    steps node[axes[i]] = steps[i]  for i in axes.size
    //    steps node: [1, 1, 1, 1, 1] -> [1, 1, 2, 1, 1].
    // 4. Calculate and broadcast update_value to corresponding shape: [5, 6, 5, 3, 9].
    //    1. Calculate `end - start`: [5, 3].
    //    2. Calculate `(end - start) / step`: [2.5, 3]
    //    3. Calculate `ceil((end - start) / step)`: [3, 3]
    //    2. Use `ScatterNDUpdate` to get `target_value_shape` for input_shape: [5, 6, 7, 8, 9] -> [5, 6, 3, 3, 9].
    //    3. Broadcast from [1, 6, 3, 3] to [5, 6, 3, 3, 9]
    // 5. Create a range_node filled with number from 0 to numel.
    //    1. Reshape it to input_shape.
    //    2. Use `StridedSlice` to get those indices which are about to be updated.
    // 6. Flatten input, update_value and sliced_range.
    // 7. Use `ScatterUpdate` update update_value into input_data.
    // 8. Reshape input to original input_shape.

    const auto axes_node = default_opset::Constant::create(element::i64, {axes.size(), 1}, axes);
    const auto spec_dim_node = std::make_shared<default_opset::GatherND>(input_shape, axes_node);
    const auto zero_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto one_node = default_opset::Constant::create(element::i64, Shape{}, {1});
    const auto dim_node = default_opset::Constant::create(element::i64, Shape{}, {dims});
    const auto reshape_flatten = default_opset::Constant::create(ov::element::i64, {1}, {-1});

    // get positive starts ends and steps
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

    // 3.1 get starts node
    starts_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    starts_node = std::make_shared<default_opset::ScatterNDUpdate>(starts_node, axes_node, starts);

    // 3.2 get ends node
    ends_node = default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    ends_node = std::make_shared<default_opset::ScatterNDUpdate>(ends_node, axes_node, ends);

    // 3.3 get steps node
    steps_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims, 1));
    steps_node = std::make_shared<default_opset::ScatterNDUpdate>(steps_node, axes_node, steps);

    // 4.get target value shape
    // 4.1 end - start
    Output<Node> value_shape_update_node = std::make_shared<default_opset::Subtract>(ends, starts);
    // 4.2 ( end - start ) / step
    value_shape_update_node = std::make_shared<default_opset::Divide>(value_shape_update_node, steps);
    // 4.3 ceil(( end - start ) / step)
    value_shape_update_node = std::make_shared<default_opset::Ceiling>(value_shape_update_node);
    // 4.4 update
    const auto value_target_shape =
        std::make_shared<default_opset::ScatterNDUpdate>(input_shape, axes_node, value_shape_update_node);

    // 4.5 broadcast
    value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape);

    // get total number of elements
    const auto numel_node = std::make_shared<default_opset::ReduceProd>(input_shape, zero_node);
    // generate indices from 0 to numel - 1
    Output<Node> range_node = std::make_shared<default_opset::Range>(zero_node, numel_node, one_node, element::i64);
    // reshape to input_shape
    range_node = std::make_shared<default_opset::Reshape>(range_node, input_shape, true);
    // slice range node, get the indices thta to be updated
    Output<Node> sliced_range_node = std::make_shared<default_opset::StridedSlice>(range_node,
                                                                                   starts_node,
                                                                                   ends_node,
                                                                                   steps_node,
                                                                                   std::vector<int64_t>(dims),
                                                                                   std::vector<int64_t>(dims));

    // flatten input, upadte_value and sliced_range_node
    input_node = std::make_shared<default_opset::Reshape>(input_node, reshape_flatten, true);
    sliced_range_node = std::make_shared<default_opset::Reshape>(sliced_range_node, reshape_flatten, true);
    value_node = std::make_shared<default_opset::Reshape>(value_node, reshape_flatten, true);

    // update value to input according to sliced_range_node
    input_node = std::make_shared<default_opset::ScatterUpdate>(input_node, sliced_range_node, value_node, zero_node);

    // reshape to original shape
    return node.default_single_output_mapping({std::make_shared<default_opset::Reshape>(input_node, input_shape, true)},
                                              {"Out"});
};

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
