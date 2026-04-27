// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

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
}  // namespace

OutputVector translate_quantized_batch_norm(const NodeContext& context) {
    // Schema: quantized::batch_norm2d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var,
    //                                  float eps, float output_scale, int output_zero_point) -> Tensor
    num_inputs_check(context, 8, 8);
    const auto input = context.get_input(0);

    Output<Node> weight;
    Output<Node> bias;

    // Handle optional weight (index 1)
    if (!context.input_is_none(1)) {
        weight = context.get_input(1);
        weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, input));
    } else {
        auto one_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
        weight = broadcast_const_to_channel_dim(context, input, one_f);
    }

    // Handle optional bias (index 2)
    if (!context.input_is_none(2)) {
        bias = context.get_input(2);
        bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, input));
    } else {
        auto zero_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        bias = broadcast_const_to_channel_dim(context, input, zero_f);
    }

    // Running mean (index 3)
    auto running_mean = context.get_input(3);
    running_mean = context.mark_node(std::make_shared<v1::ConvertLike>(running_mean, input));

    // Running var (index 4)
    auto running_var = context.get_input(4);
    running_var = context.mark_node(std::make_shared<v1::ConvertLike>(running_var, input));

    // Epsilon (index 5)
    float epsilon = context.const_input<float>(5);

    // Output scale (index 6)
    const auto scale = context.get_input(6);

    // Output zero point (index 7)
    const auto zero_point = context.get_input(7);

    // Create BatchNormInference
    auto batch_norm =
        context.mark_node(std::make_shared<v5::BatchNormInference>(input, weight, bias, running_mean, running_var, epsilon));

    // Quantize the output
    return {quantize(context, batch_norm, scale, zero_point, input)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
