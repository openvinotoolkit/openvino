// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
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

using namespace ov::op;

namespace {
OutputVector translate_instance_norm_inference(const NodeContext& context,
                                               const Output<Node>& input,
                                               const Output<Node>& reduction_axes,
                                               float eps) {
    auto norm = context.mark_node(std::make_shared<v6::MVN>(input, reduction_axes, true, eps, MVNEpsMode::INSIDE_SQRT));
    if (!context.input_is_none(1)) {
        auto weight = context.get_input(1);
        weight = reshape_channelwise(context, weight, norm);
        norm = context.mark_node(std::make_shared<v1::Multiply>(norm, weight));
    }
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        bias = reshape_channelwise(context, bias, norm);
        norm = context.mark_node(std::make_shared<v1::Add>(norm, bias));
    }
    return {norm};
}

OutputVector translate_instance_norm_train(const NodeContext& context,
                                           const Output<Node>& input,
                                           const Output<Node>& reduction_axes,
                                           float eps) {
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto batch_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, zero, zero));
    auto channel_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, one, zero));
    auto batch_dim_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(batch_dim, zero));
    auto batch_norm_channels_1d = context.mark_node(std::make_shared<v1::Multiply>(batch_dim_1d, channel_dim));
    auto one_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto tail_shape = context.mark_node(std::make_shared<v8::Gather>(input_shape, reduction_axes, zero));
    auto reshape_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{one_1d, batch_norm_channels_1d, tail_shape}, 0));
    auto reshaped_input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_shape, false));
    Output<Node> weight;
    Output<Node> bias;
    if (context.input_is_none(1)) {
        weight = context.mark_node(std::make_shared<v3::Broadcast>(one, batch_norm_channels_1d));
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, input));
    } else {
        weight = context.get_input(1);
        weight = context.mark_node(std::make_shared<v0::Tile>(weight, batch_dim_1d));
    }
    if (context.input_is_none(2)) {
        bias = context.mark_node(std::make_shared<v3::Broadcast>(zero, batch_norm_channels_1d));
        bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, input));
    } else {
        bias = context.get_input(2);
        bias = context.mark_node(std::make_shared<v0::Tile>(bias, batch_dim_1d));
    }
    auto running_mean = context.get_input(3);
    running_mean = context.mark_node(std::make_shared<v0::Tile>(running_mean, batch_dim_1d));
    auto running_var = context.get_input(4);
    running_var = context.mark_node(std::make_shared<v0::Tile>(running_var, batch_dim_1d));
    auto batch_norm = context.mark_node(
        std::make_shared<v5::BatchNormInference>(reshaped_input, weight, bias, running_mean, running_var, eps));
    return {context.mark_node(std::make_shared<v1::Reshape>(batch_norm, input_shape, true))};
}

}  // namespace

OutputVector translate_instance_norm(const NodeContext& context) {
    num_inputs_check(context, 8, 9);
    auto input = context.get_input(0);
    auto eps = context.const_input<float>(7);
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(context, input, true, element::i32);
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    auto reduction_axes = context.mark_node(std::make_shared<v4::Range>(two, rank, one, element::i32));
    if (context.input_is_none(3) && context.input_is_none(4)) {
        return translate_instance_norm_inference(context, input, reduction_axes, eps);
    }
    return translate_instance_norm_train(context, input, reduction_axes, eps);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov