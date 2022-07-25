// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
template <typename T>
std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1}) {
    auto value_count = static_cast<std::size_t>(std::floor((end_value - start_value) / step));
    std::vector<T> range(value_count);
    size_t n = start_value - step;
    std::generate(std::begin(range), std::end(range), [&n, &step]() -> T {
        return n += step;
    });
    return range;
}

std::shared_ptr<ov::Node> get_monotonic_range_along_node_rank(const Output<ov::Node>& value,
                                                              int64_t start_value,
                                                              int64_t step) {
    if (value.get_partial_shape().rank().is_static()) {
        const auto range_value =
            op::get_monotonic_range<int64_t>(value.get_partial_shape().rank().get_length(), start_value, step);
        return default_opset::Constant::create(element::i64, {range_value.size()}, range_value);
    }
    const auto value_shape = std::make_shared<default_opset::ShapeOf>(value);
    return std::make_shared<default_opset::Range>(default_opset::Constant::create(element::i64, {}, {start_value}),
                                                  std::make_shared<default_opset::ShapeOf>(value_shape),
                                                  default_opset::Constant::create(element::i64, {}, {step}),
                                                  element::i64);
}

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
    auto epsilon = node.get_attribute<float>("epsilon", 1e-5);
    auto data_layout = node.get_attribute<std::string>("data_layout");

    PADDLE_OP_CHECK(node, (data_layout == "NCHW" || data_layout == "NHWC"), "Not supported input data layout!");
    if (data_layout == "NHWC") {
        auto perm1 = default_opset::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        data = std::make_shared<default_opset::Transpose>(data, perm1);
    }

    auto num_groups_const = default_opset::Constant::create(element::i64, Shape{1}, {num_groups});
    auto data_shape_node = std::make_shared<default_opset::ShapeOf>(data);
    const auto& pshape = data.get_partial_shape();
    size_t rank_size = pshape.rank().get_length();
    auto shape = std::make_shared<default_opset::ShapeOf>(data);
    auto axis_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    auto split = std::make_shared<default_opset::Split>(shape, axis_node, rank_size);
    auto splits = split->outputs();
    ov::OutputVector new_shape{std::make_shared<default_opset::Multiply>(splits[0], num_groups_const),
                               std::make_shared<default_opset::Divide>(splits[1], num_groups_const)};
    for (size_t i = 2; i < rank_size; i++) {
        new_shape.push_back(splits[i]);
    }
    auto reshaped_ = std::make_shared<default_opset::Concat>(new_shape, 0);
    auto data_reshaped = std::make_shared<default_opset::Reshape>(data, reshaped_, true);
    const auto reduction_axes = op::get_monotonic_range_along_node_rank(data_reshaped, 1, 1);

    auto mvn = std::make_shared<default_opset::MVN>(data_reshaped,
                                                    reduction_axes,
                                                    true,
                                                    epsilon,
                                                    ov::op::MVNEpsMode::INSIDE_SQRT);
    std::shared_ptr<ov::Node> result = std::make_shared<default_opset::Reshape>(mvn, data_shape_node, true);

    auto has_scale = node.has_input("Scale");
    auto has_bias = node.has_input("Bias");

    const auto data_rank = std::make_shared<default_opset::ShapeOf>(data_shape_node);
    if (has_scale) {
        auto scale = node.get_input("Scale");
        const auto& scale_shape = scale.get_partial_shape();
        auto scale_rank = scale_shape.rank().get_length();
        if (scale_rank == 1) {
            result =
                std::make_shared<default_opset::Multiply>(result,
                                                          op::reshape_channel_shaped_node_to_nchw(scale, data_rank));
        } else {
            result = std::make_shared<default_opset::Multiply>(result, scale);
        }
    }
    if (has_bias) {
        auto bias = node.get_input("Bias");
        const auto& bias_shape = bias.get_partial_shape();
        auto bias_rank = bias_shape.rank().get_length();
        if (bias_rank == 1) {
            result =
                std::make_shared<default_opset::Add>(result, op::reshape_channel_shaped_node_to_nchw(bias, data_rank));
        } else {
            result = std::make_shared<default_opset::Add>(result, bias);
        }
    }

    if (data_layout == "NHWC") {
        auto perm2 = default_opset::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});
        result = std::make_shared<default_opset::Transpose>(result, perm2);
    }
    return node.default_single_output_mapping({result}, {"Y"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
