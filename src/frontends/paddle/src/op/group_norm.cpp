// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

Output<ov::Node> reshape_channel_shaped_node_to_nchw(const Output<ov::Node>& node,
                                                     const Output<ov::Node>& expected_rank) {
    const auto one_const = default_opset::Constant::create(element::i64, Shape{1}, {1});
    const auto two_const = default_opset::Constant::create(element::i64, Shape{1}, {2});
    const auto tail_shape_rank = std::make_shared<default_opset::Subtract>(expected_rank, two_const);
    const auto tail_shape = std::make_shared<default_opset::Broadcast>(one_const, tail_shape_rank);
    const auto C_dim = std::make_shared<default_opset::ShapeOf>(node);
    const auto new_shape = std::make_shared<default_opset::Concat>(OutputVector{one_const, C_dim, tail_shape}, 0);
    return std::make_shared<default_opset::Reshape>(node, new_shape, false);
}

NamedOutputs group_norm(const NodeContext& node) {
    auto data = node.get_input("X");
    size_t num_groups = static_cast<size_t>(node.get_attribute<int32_t>("groups"));
    auto epsilon = node.get_attribute<float>("epsilon", 1e-5f);
    auto data_layout = node.get_attribute<std::string>("data_layout", "NCHW");

    const auto& pshape = data.get_partial_shape();
    PADDLE_OP_CHECK(node, pshape.rank().is_static());
    size_t rank_size = pshape.rank().get_length();
    PADDLE_OP_CHECK(node, rank_size >= 2, "2-D and above tensors supported only");

    if (data_layout == "NHWC") {
        auto values = std::vector<size_t>{0, rank_size - 1};
        for (size_t i = 1; i < rank_size - 1; i++) {
            values.push_back(i);
        }
        auto perm1 = default_opset::Constant::create(element::i64, Shape{rank_size}, values);
        data = std::make_shared<default_opset::Transpose>(data, perm1);
    }
    // The process below creates a shape to which we need to reshape the input before normalization.
    auto num_groups_const = default_opset::Constant::create(element::i64, Shape{1}, {num_groups});
    auto data_shape_node = std::make_shared<default_opset::ShapeOf>(data);
    auto shape = std::make_shared<default_opset::ShapeOf>(data);
    auto axis_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<default_opset::Split>(shape, axis_node, rank_size);
    auto splits = split->outputs();
    ov::OutputVector new_shape{std::make_shared<default_opset::Multiply>(splits[0], num_groups_const),
                               std::make_shared<default_opset::Divide>(splits[1], num_groups_const)};
    for (size_t i = 2; i < rank_size; i++) {
        new_shape.push_back(splits[i]);
    }
    // The 4D shape: [N * num_groups, C // num_groups, H, W] is created
    // instead of 5D shape: [N, num_groups, C // num_groups, H, W].
    // The reason is the lack of support for 5D MVN input by some plugins.
    auto reshaped_ = std::make_shared<default_opset::Concat>(new_shape, 0);
    auto data_reshaped = std::make_shared<default_opset::Reshape>(data, reshaped_, true);
    const Output<ov::Node> data_reshaped_value = data_reshaped;
    PADDLE_OP_CHECK(node, data_reshaped_value.get_partial_shape().rank().is_static());
    size_t reshape_rank = data_reshaped_value.get_partial_shape().rank().get_length();
    std::vector<size_t> range_value;
    for (size_t i = 1; i < reshape_rank; i++)
        range_value.push_back(i);
    const auto reduction_axes = default_opset::Constant::create(element::i64, {range_value.size()}, range_value);

    auto mvn = std::make_shared<default_opset::MVN>(data_reshaped,
                                                    reduction_axes,
                                                    true,
                                                    epsilon,
                                                    ov::op::MVNEpsMode::INSIDE_SQRT);
    std::shared_ptr<ov::Node> result = std::make_shared<default_opset::Reshape>(mvn, data_shape_node, true);
    // The process below reshape the result that become standrd output after normalization.
    const auto data_rank = std::make_shared<default_opset::ShapeOf>(data_shape_node);
    if (node.has_input("Scale")) {
        auto scale = node.get_input("Scale");
        const auto& scale_shape = scale.get_partial_shape();
        PADDLE_OP_CHECK(node, scale_shape.rank().is_static());
        auto scale_rank = scale_shape.rank().get_length();
        if (scale_rank == 1) {
            result =
                std::make_shared<default_opset::Multiply>(result,
                                                          op::reshape_channel_shaped_node_to_nchw(scale, data_rank));
        } else {
            result = std::make_shared<default_opset::Multiply>(result, scale);
        }
    }

    if (node.has_input("Bias")) {
        auto bias = node.get_input("Bias");
        const auto& bias_shape = bias.get_partial_shape();
        PADDLE_OP_CHECK(node, bias_shape.rank().is_static());
        auto bias_rank = bias_shape.rank().get_length();
        if (bias_rank == 1) {
            result =
                std::make_shared<default_opset::Add>(result, op::reshape_channel_shaped_node_to_nchw(bias, data_rank));
        } else {
            result = std::make_shared<default_opset::Add>(result, bias);
        }
    }

    if (data_layout == "NHWC") {
        auto values = std::vector<size_t>{0};
        for (size_t i = 2; i < rank_size; i++) {
            values.push_back(i);
        }
        values.push_back(1);
        auto perm2 = default_opset::Constant::create(element::i64, Shape{rank_size}, values);
        result = std::make_shared<default_opset::Transpose>(result, perm2);
    }

    return node.default_single_output_mapping({result}, {"Y"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
