// // Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

std::shared_ptr<Node> get_tensor_list(const OutputVector& node) {
    auto tensor_list = node;
    for (size_t i = 0; i < node.size(); i++) {
        if (node[i].get_shape().size() == 0) {
            tensor_list[i] =
                std::make_shared<default_opset::Unsqueeze>(node[i],
                                                           default_opset::Constant::create(element::i64, {1}, {0}));
        }
    }
    const auto new_node = std::make_shared<default_opset::Concat>(tensor_list, 0);
    return new_node;
}

std::shared_ptr<Node> handle_minus_index(const std::vector<int64_t>& node, const Output<Node>& dim) {
    const auto new_node = default_opset::Constant::create(element::i64, {node.size()}, node);
    return new_node;
}

std::shared_ptr<Node> handle_maximum_index(Output<Node>& node, const Output<Node>& update_node) {
    const auto maximum_node = default_opset::Constant::create(element::i64, {1}, {std::numeric_limits<int32_t>::max()});
    const auto mask = std::make_shared<default_opset::Equal>(node, maximum_node);
    return std::make_shared<default_opset::Select>(mask, update_node, node);
}

bool is_contain_minus(const std::vector<int64_t> vec) {
    for (int64_t i : vec) {
        if (i < 0)
            return true;
    }
    return false;
}

NamedOutputs set_value(const NodeContext& node) {
    auto input_node = node.get_input("Input");
    auto value_node = node.get_input("ValueTensor");

    PADDLE_OP_CHECK(node, (input_node.get_partial_shape().rank().is_static()), "rank must be static");
    const auto dims = static_cast<int64_t>(input_node.get_partial_shape().rank().get_length());
    const auto axes = node.get_attribute<std::vector<int64_t>>("axes");

    // const auto input_shape_ = input_node.get_partial_shape().get_shape();
    // auto input_shape = default_opset::Constant::create(element::i64, {input_shape_.size()}, input_shape_);
    auto input_shape = std::make_shared<default_opset::ShapeOf>(input_node);

    Output<Node> starts_node, ends_node, steps_node, starts, ends, steps;

    // The following process is:
    // Given:
    // input_data: shape(5, 6, 7, 8, 9)
    // update_value: shape(1, 6, 3, 3)
    // operation: input_data[:, :, 2: 7: 2, -4: -1, :] = update_value
    // axes = [2, 3]
    // starts = [2, -4]
    // ends = [7, -1]
    // steps = [2, 1]
    // Our process is:
    // 1. Get axes [2, 3], get shape of input [5, 6, 7, 8, 9], select dimension from shape by axes: [7, 8].
    // 2. Get starts [2, -4] and ends [7, -1]. Process minus starts and ends. starts: [2, 4], ends: [7, 7].
    // 3. Calculate starts_node, ends_node and steps_node
    //    1. Create `starts node` filled with 0. Update `starts` to `starts_node` according to axes.
    //    starts_node[axes[i]] = starts[i] for i in axes.size
    //    starts_node: [0, 0, 0, 0, 0] -> [0, 0, 2, 4, 0].
    //    2. Update `ends` to `input_shape` according to axes.
    //    input_shape[axes[i]] = ends[i] for i in axes.size
    //    ends_node: [5, 6, 7, 8, 9] -> [5, 6, 7, 7, 9].
    //    3. Create `steps_node` filled with 1. Update `steps' to `steps_node` according to axes.
    //    steps_node[axes[i]] = steps[i]  for i in axes.size
    //    steps_node: [1, 1, 1, 1, 1] -> [1, 1, 2, 1, 1].
    // 4. Calculate and broadcast update_value to corresponding shape: [5, 6, 5, 3, 9].
    //    1. Calculate `end - start`: [5, 3].
    //    2. Calculate `(end - start) / abs(step)`: [2.5, 3]
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
    const auto slice_shape = default_opset::Constant::create(ov::element::i64, {1, 1}, {-1});

    // get positive starts ends and steps
    if (node.has_input("StartsTensorList")) {
        starts = get_tensor_list(node.get_ng_inputs("StartsTensorList"));
    } else if (node.has_attribute("starts")) {
        auto start_vec = node.get_attribute<std::vector<int64_t>>("starts");
        if (is_contain_minus(start_vec)) {
            PADDLE_OP_CHECK(node, (false), "Currently not support minus start!");
        }
        starts = handle_minus_index(start_vec, spec_dim_node);
    } else
        PADDLE_OP_CHECK(node, (false), "Invalid arguments!");

    if (node.has_input("EndsTensorList")) {
        ends = get_tensor_list(node.get_ng_inputs("EndsTensorList"));
    } else if (node.has_attribute("ends")) {
        auto ends_vec = node.get_attribute<std::vector<int64_t>>("ends");
        if (is_contain_minus(ends_vec)) {
            PADDLE_OP_CHECK(node, (false), "Currently not support minus ends!");
        }
        ends = handle_minus_index(ends_vec, spec_dim_node);
    } else
        PADDLE_OP_CHECK(node, (false), "Invalid arguments!");

    if (node.has_input("StepsTensorList")) {
        steps = get_tensor_list(node.get_ng_inputs("StepsTensorList"));
    } else if (node.has_attribute("steps")) {
        auto step_vec = node.get_attribute<std::vector<int64_t>>("steps");
        if (is_contain_minus(step_vec)) {
            PADDLE_OP_CHECK(node, (false), "Currently not support minus steps!");
        }
        steps = handle_minus_index(step_vec, spec_dim_node);
    } else
        PADDLE_OP_CHECK(node, (false), "Invalid arguments!");

    // for unsepcified end: x[::2], end will be 2147483647
    ends = handle_maximum_index(ends, spec_dim_node);

    // 3.1 get starts node
    starts_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims));
    starts_node = std::make_shared<default_opset::ScatterNDUpdate>(starts_node, axes_node, starts);

    // 3.2 get ends node
    ends_node = std::make_shared<default_opset::ScatterNDUpdate>(input_shape, axes_node, ends);

    // 3.3 get steps node
    steps_node =
        default_opset::Constant::create(element::i64, {static_cast<size_t>(dims)}, std::vector<int64_t>(dims, 1));
    steps_node = std::make_shared<default_opset::ScatterNDUpdate>(steps_node, axes_node, steps);

    // 4.get target value shape
    // 4.1 end - start
    Output<Node> value_shape_update_node = std::make_shared<default_opset::Subtract>(ends, starts);
    // 4.2 ( end - start ) / step
    value_shape_update_node = std::make_shared<default_opset::Convert>(value_shape_update_node, element::f32);
    // We don't need to process the the minus number in steps
    // if step < 0, end - start < 0, (end - start) / step > 0
    steps = std::make_shared<default_opset::Convert>(steps, element::f32);
    value_shape_update_node = std::make_shared<default_opset::Divide>(value_shape_update_node, steps);
    // 4.3 ceil(( end - start ) / step)
    value_shape_update_node = std::make_shared<default_opset::Ceiling>(value_shape_update_node);
    value_shape_update_node = std::make_shared<default_opset::Convert>(value_shape_update_node, element::i64);
    // 4.4 update
    const auto value_target_shape =
        std::make_shared<default_opset::ScatterNDUpdate>(input_shape, axes_node, value_shape_update_node);

    // 4.5 broadcast
    auto value_shape = std::make_shared<default_opset::ShapeOf>(value_node);
    auto value_rank = std::make_shared<default_opset::ShapeOf>(value_shape);
    auto value_rank_scalar = std::make_shared<default_opset::Squeeze>(value_rank);
    Output<Node> broadcast_axes =
        std::make_shared<default_opset::Range>(zero_node, value_rank_scalar, one_node, element::i64);
    value_node = std::make_shared<default_opset::Broadcast>(value_node, value_target_shape, broadcast_axes);

    // get total number of elements
    const auto numel_node = std::make_shared<default_opset::ReduceProd>(input_shape, zero_node);
    // generate indices from 0 to numel - 1
    Output<Node> range_node = std::make_shared<default_opset::Range>(zero_node, numel_node, one_node, element::i64);
    // reshape to input_shape
    range_node = std::make_shared<default_opset::Reshape>(range_node, input_shape, true);
    // slice range node, get the indices that to be updated
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
    sliced_range_node = std::make_shared<default_opset::Unsqueeze>(sliced_range_node, one_node);
    input_node = std::make_shared<default_opset::ScatterNDUpdate>(input_node, sliced_range_node, value_node);

    // reshape to original shape
    return node.default_single_output_mapping({std::make_shared<default_opset::Reshape>(input_node, input_shape, true)},
                                              {"Out"});
};

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
