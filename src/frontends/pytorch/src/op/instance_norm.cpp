// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_instance_norm(NodeContext& context) {
    auto input = context.get_input(0);
    auto eps = static_cast<float>(context.const_input<double>(7));
    auto input_shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(input));
    auto rank_1d = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(input_shape));
    auto zero = context.mark_node(ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto rank = context.mark_node(std::make_shared<ov::op::v0::Squeeze>(rank_1d));
    auto one = context.mark_node(ov::op::v0::Constant::create(element::i64, {}, {1}));
    auto two = context.mark_node(ov::op::v0::Constant::create(element::i64, {}, {2}));
    auto reduction_axes = context.mark_node(std::make_shared<ov::op::v4::Range>(two, rank, one, element::i64));
    if (context.input_is_none(3) && context.input_is_none(4)) {
        auto norm = context.mark_node(
            std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT));
        if (!context.input_is_none(1)) {
            auto weight = context.get_input(1);
            weight = reshape_conv_bias(context, weight, norm);
            norm = context.mark_node(std::make_shared<ov::op::v1::Multiply>(norm, weight));
        }
        if (!context.input_is_none(2)) {
            auto bias = context.get_input(2);
            bias = reshape_conv_bias(context, bias, norm);
            norm = context.mark_node(std::make_shared<ov::op::v1::Add>(norm, bias));
        }
        return {norm};
    }
    auto batch_dim = context.mark_node(std::make_shared<ov::op::v1::Gather>(input_shape, zero, zero));
    auto channel_dim = context.mark_node(std::make_shared<ov::op::v1::Gather>(input_shape, one, zero));
    auto batch_dim_1d = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(batch_dim, zero));
    auto batch_norm_channels_1d = context.mark_node(std::make_shared<ov::op::v1::Multiply>(batch_dim_1d, channel_dim));
    auto one_1d = context.mark_node(ov::op::v0::Constant::create(element::i64, Shape{1}, {1}));
    auto tail_shape = context.mark_node(std::make_shared<ov::op::v1::Gather>(input_shape, reduction_axes, zero));
    auto reshape_shape = context.mark_node(
        std::make_shared<ov::op::v0::Concat>(OutputVector{one_1d, batch_norm_channels_1d, tail_shape}, 0));
    auto reshaped_input = context.mark_node(std::make_shared<ov::op::v1::Reshape>(input, reshape_shape, false));
    Output<Node> weight;
    Output<Node> bias;
    if (context.input_is_none(1)) {
        weight = context.mark_node(std::make_shared<ov::op::v1::Broadcast>(one, batch_norm_channels_1d));
        weight = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(weight, input));
    } else {
        weight = context.get_input(1);
        weight = context.mark_node(std::make_shared<ov::op::v0::Tile>(weight, batch_dim_1d));
    }
    if (context.input_is_none(2)) {
        bias = context.mark_node(std::make_shared<ov::op::v1::Broadcast>(zero, batch_norm_channels_1d));
        bias = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(bias, input));
    } else {
        bias = context.get_input(2);
        bias = context.mark_node(std::make_shared<ov::op::v0::Tile>(bias, batch_dim_1d));
    }
    auto running_mean = context.get_input(3);
    running_mean = context.mark_node(std::make_shared<ov::op::v0::Tile>(running_mean, batch_dim_1d));
    auto running_var = context.get_input(4);
    running_var = context.mark_node(std::make_shared<ov::op::v0::Tile>(running_var, batch_dim_1d));
    auto batch_norm = context.mark_node(
        std::make_shared<ov::op::v5::BatchNormInference>(reshaped_input, weight, bias, running_mean, running_var, eps));
    return {context.mark_node(std::make_shared<ov::op::v1::Reshape>(batch_norm, input_shape, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov