// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/pass/graph_rewrite.hpp"
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
    auto value_ = context.mark_node(std::make_shared<v1::ConvertLike>(value, input));
    return context.mark_node(std::make_shared<v3::Broadcast>(value_, channel_dim_exp));
}

OutputVector make_batch_norm(const NodeContext& context,
                             const Output<Node>& input,
                             const Output<Node>& weight,
                             const Output<Node>& bias,
                             const Output<Node>& running_mean,
                             const Output<Node>& running_var,
                             float epsilon) {
    Output<Node> w = weight;
    Output<Node> b = bias;
    Output<Node> mean = running_mean;
    Output<Node> var = running_var;
    if (!w.get_node_shared_ptr()) {
        auto one_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
        w = broadcast_const_to_channel_dim(context, input, one_f);
    } else {
        w = context.mark_node(std::make_shared<v1::ConvertLike>(w, input));
    }
    if (!b.get_node_shared_ptr()) {
        auto zero_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        b = broadcast_const_to_channel_dim(context, input, zero_f);
    } else {
        b = context.mark_node(std::make_shared<v1::ConvertLike>(b, input));
    }
    auto zero_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    Output<Node> rank = std::get<1>(get_shape_rank(context, input, true));
    auto after_channel_dims = context.mark_node(std::make_shared<v0::Range>(two, rank, one));
    auto axes = context.mark_node(std::make_shared<v0::Concat>(OutputVector{zero_1d, after_channel_dims}, 0));
    if (!mean.get_node_shared_ptr()) {
        mean = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, false));
    } else {
        mean = context.mark_node(std::make_shared<v1::ConvertLike>(mean, input));
    }
    if (!var.get_node_shared_ptr()) {
        auto current_mean = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, true));
        auto sub_v = context.mark_node(std::make_shared<v1::Subtract>(input, current_mean));
        auto sqr_sub = context.mark_node(std::make_shared<v1::Multiply>(sub_v, sub_v));
        var = context.mark_node(std::make_shared<v1::ReduceMean>(sqr_sub, axes, false));
    } else {
        var = context.mark_node(std::make_shared<v1::ConvertLike>(var, input));
    }
    return {context.mark_node(std::make_shared<v5::BatchNormInference>(input, w, b, mean, var, epsilon))};
}
}  // namespace

OutputVector translate_batch_norm(const NodeContext& context) {
    // Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
    // bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
    num_inputs_check(context, 7, 9);
    Output<Node> weight;
    Output<Node> bias;
    Output<Node> running_mean;
    Output<Node> running_var;
    if (!context.input_is_none(1)) {
        weight = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        bias = context.get_input(2);
    }
    // index 3 running_mean and index 4 running_var can be none for training case only, check that not training before
    // if training for batch norm activated, but model in eval mode, it uses current statistics instead of running
    auto training = context.const_input<bool>(5);
    if (!training) {
        running_mean = context.get_input(3);
        running_var = context.get_input(4);
    }
    // Input with index 6 is momentum, it is used only for updating running_mean accumulation during training
    float epsilon = context.const_input<float>(7);
    // Input with index 8 is flag "cudnn_enabled" we can ignore it
    return make_batch_norm(context, context.get_input(0), weight, bias, running_mean, running_var, epsilon);
}

OutputVector translate_batch_norm_legit_fx(const NodeContext& context) {
    auto output = translate_batch_norm(context);
    return {context.mark_node(make_list_construct(output))};
}

OutputVector translate_batch_norm_legit_no_training_fx(const NodeContext& context) {
    num_inputs_check(context, 7, 9);
    Output<Node> weight;
    Output<Node> bias;
    if (!context.input_is_none(1)) {
        weight = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        bias = context.get_input(2);
    }
    auto running_mean = context.get_input(3);
    auto running_var = context.get_input(4);
    float epsilon = context.const_input<float>(6);
    auto output = make_batch_norm(context, context.get_input(0), weight, bias, running_mean, running_var, epsilon);
    return {context.mark_node(make_list_construct(output))};
}

OutputVector translate_batch_norm_legit_no_stats_fx(const NodeContext& context) {
    num_inputs_check(context, 6, 6);
    // torch.ops.aten._native_batch_norm_legit.no_stats(arg2_1, arg0_1, arg1_1, True, 0.1, 5e-05)
    Output<Node> weight;
    if (!context.input_is_none(1)) {
        weight = context.get_input(1);
    }
    Output<Node> bias;
    if (!context.input_is_none(2)) {
        bias = context.get_input(2);
    }
    auto training = context.const_input<bool>(3);
    PYTORCH_OP_CONVERSION_CHECK(training,
                                "aten._native_batch_norm_legit.no_stats can only be used when training=True.");
    // index 4 momentum is used during training only
    auto eps = context.const_input<float>(5);
    auto output = make_batch_norm(context, context.get_input(0), weight, bias, {}, {}, eps);
    return {context.mark_node(make_list_construct(output))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
