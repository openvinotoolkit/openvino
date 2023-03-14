// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
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

OutputVector translate_batch_norm(NodeContext& context) {
    // Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
    // bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
    num_inputs_check(context, 8, 9);
    auto input = context.get_input(0);
    Output<Node> weight;
    Output<Node> bias;
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
    auto training = context.const_input<bool>(5);
    FRONT_END_OP_CONVERSION_CHECK(!training, "Translation for aten::batch_norm do not support training mode.");
    auto running_mean = context.get_input(3);
    auto running_var = context.get_input(4);
    // Input with index 6 is momentum, it is used only in training mode
    auto epsilon = context.const_input<float>(7);
    // Input with index 8 is flag "cudnn_enabled" we can ignore it
    return {context.mark_node(
        std::make_shared<v5::BatchNormInference>(input, weight, bias, running_mean, running_var, epsilon))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov