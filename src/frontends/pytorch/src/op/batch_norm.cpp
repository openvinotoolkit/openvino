// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> broadcast_const_to_channel_dim(const NodeContext& context,
                                            const Output<Node>& input,
                                            const Output<Node>& value) {
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto zero_i = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one_i = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto channel_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, one_i, zero_i));
    auto channel_dim_exp = context.mark_node(std::make_shared<v0::Unsqueeze>(channel_dim, zero_i));
    return context.mark_node(std::make_shared<v3::Broadcast>(value, channel_dim_exp));
}
}  // namespace

OutputVector translate_batch_norm_common(const NodeContext& context, bool training) {
    // Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
    // bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor

    //  batch_norm_legit_no_training Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
    //  running_mean, Tensor? running_var, float momentum, float eps) -> Tensor

    auto input = context.get_input(0);
    Output<Node> weight;
    Output<Node> bias;
    Output<Node> running_mean;
    Output<Node> running_var;
    Output<Node> current_mean;
    Output<Node> current_var;
    if (!context.input_is_none(1)) {
        weight = context.get_input(1);
    } else {
        auto one_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
        weight = broadcast_const_to_channel_dim(context, input, one_f);
    }
    if (!context.input_is_none(2)) {
        bias = context.get_input(2);
    } else {
        auto zero_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        bias = broadcast_const_to_channel_dim(context, input, zero_f);
    }
    // index 3 running_mean and index 4 running_var can be none for training case only, check that not training before
    // if training for batch norm activated, but model in eval mode, it uses current statistics instead of running
    if (training) {
        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
        auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
        auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
        auto rank_unsq = context.mark_node(std::make_shared<v3::ShapeOf>(input_shape, element::i32));
        auto rank = context.mark_node(std::make_shared<v0::Squeeze>(rank_unsq, zero));
        auto after_channel_dims = context.mark_node(std::make_shared<v0::Range>(two, rank, one));
        auto axes = context.mark_node(std::make_shared<v0::Concat>(OutputVector{zero_1d, after_channel_dims}, 0));
        current_mean = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, false));
        auto mean = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, true));
        auto sub_v = context.mark_node(std::make_shared<v1::Subtract>(input, mean));
        auto sqr_sub = context.mark_node(std::make_shared<v1::Multiply>(sub_v, sub_v));
        current_var = context.mark_node(std::make_shared<v1::ReduceMean>(sqr_sub, axes, false));
    }
    if (!training) {
        running_mean = context.get_input(3);
    } else {
        running_mean = current_mean;
    }
    if (!training) {
        running_var = context.get_input(4);
    } else {
        running_var = current_var;
    }
    // Input with index 6 is momentum, it is used only for updating running_mean accumulation during training
    // In batch_norm_legit_no_training, momentum is index 5 and epsilon is 6
    float epsilon;
    if (context.get_input_size() == 7) {
        epsilon = context.const_input<float>(6);
    } else {
        epsilon = context.const_input<float>(7);
    }
    // Input with index 8 is flag "cudnn_enabled" we can ignore it
    return {context.mark_node(
        std::make_shared<v5::BatchNormInference>(input, weight, bias, running_mean, running_var, epsilon))};
};

OutputVector translate_batch_norm(const NodeContext& context) {
    num_inputs_check(context, 7, 9);
    auto training = context.const_input<bool>(5);
    return translate_batch_norm_common(context, training);
}

OutputVector translate_batch_norm_legit_fx(const NodeContext& context) {
    num_inputs_check(context, 7, 9);
    auto training = context.const_input<bool>(5);
    auto output = translate_batch_norm_common(context, training);
    return {context.mark_node(make_list_construct(output))};
}

OutputVector translate_batch_norm_legit_no_training_fx(const NodeContext& context) {
    num_inputs_check(context, 7, 9);
    auto output = translate_batch_norm_common(context, false);
    return {context.mark_node(make_list_construct(output))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
