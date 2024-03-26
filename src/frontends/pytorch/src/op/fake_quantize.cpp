// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_fake_quantize_per_tensor_affine(const NodeContext& context) {
    num_inputs_check(context, 5, 5);
    auto input_node = context.get_input(0);
    auto scale = std::make_shared<v0::Convert>(context.get_input(1), element::f32);
    auto zero_point = std::make_shared<v0::Convert>(context.get_input(2), element::f32);
    auto out_low_const = context.const_input<int64_t>(3);
    auto out_high_const = context.const_input<int64_t>(4);
    // Calculate levels value - distance between bounds.
    auto levels = std::abs(out_high_const - out_low_const) + 1;
    auto out_low = std::make_shared<v0::Convert>(context.get_input(3), element::f32);
    auto out_high = std::make_shared<v0::Convert>(context.get_input(4), element::f32);

    // Normalize bounds according to quantization zero point value.
    auto out_low_normalized = std::make_shared<v1::Subtract>(out_low, zero_point);
    auto out_high_normalized = std::make_shared<v1::Subtract>(out_high, zero_point);
    // Rescale bounds according to scale value to calculate limits for input/output maximum/minimum values.
    auto bound_a = std::make_shared<v1::Multiply>(scale, out_low_normalized);
    auto bound_b = std::make_shared<v1::Multiply>(scale, out_high_normalized);
    // In case of negative scale bounds may be inverted, select maximum bound as high and minimal bound as low.
    auto bound_high = std::make_shared<v1::Maximum>(bound_a, bound_b);
    auto bound_low = std::make_shared<v1::Minimum>(bound_a, bound_b);
    return {context.mark_node(
        std::make_shared<v0::FakeQuantize>(input_node, bound_low, bound_high, bound_low, bound_high, levels))};
}

OutputVector translate_fake_quantize_per_channel_affine(const NodeContext& context) {
    num_inputs_check(context, 6, 6);
    auto input_node = context.get_input(0);
    auto scale = std::make_shared<v0::Convert>(context.get_input(1), element::f32);
    auto zero_point = std::make_shared<v0::Convert>(context.get_input(2), element::f32);
    auto axis = get_input_as_i32(context, 3);
    auto out_low_const = context.const_input<int64_t>(4);
    auto out_high_const = context.const_input<int64_t>(5);
    // Calculate levels value - distance between bounds.
    auto levels = std::abs(out_high_const - out_low_const) + 1;
    auto out_low = std::make_shared<v0::Convert>(context.get_input(4), element::f32);
    auto out_high = std::make_shared<v0::Convert>(context.get_input(5), element::f32);

    auto const_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
    auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});

    auto rank = std::get<1>(get_shape_rank(context, input_node));
    auto ones = std::make_shared<v3::Broadcast>(const_1, rank);
    auto normalized_axis = normalize_axis(context, axis, rank);
    // Create vector of length of rank filled with ones, except single -1 value at place selected by axis element.
    auto new_shape = std::make_shared<v3::ScatterElementsUpdate>(ones, normalized_axis, const_neg_1, const_0);
    // Reshape scale and zero point to tensor of the same rank as input, having shape 1 everywhere except dimension
    // selected by axis parameter, allowing for per-channel broadcasting.
    auto scale_bc = std::make_shared<v1::Reshape>(scale, new_shape, false);
    auto zero_point_bc = std::make_shared<v1::Reshape>(zero_point, new_shape, false);

    // Normalize bounds according to per-channel quantization zero point values.
    auto out_low_normalized = std::make_shared<v1::Subtract>(out_low, zero_point_bc);
    auto out_high_normalized = std::make_shared<v1::Subtract>(out_high, zero_point_bc);
    // Rescale bounds according to scale value to calculate limits for input/output maximum/minimum values.
    auto bound_a = std::make_shared<v1::Multiply>(scale_bc, out_low_normalized);
    auto bound_b = std::make_shared<v1::Multiply>(scale_bc, out_high_normalized);
    // In case of negative scale bounds may be inverted, select maximum bound as high and minimal bound as low.
    auto bound_high = std::make_shared<v1::Maximum>(bound_a, bound_b);
    auto bound_low = std::make_shared<v1::Minimum>(bound_a, bound_b);
    return {context.mark_node(
        std::make_shared<v0::FakeQuantize>(input_node, bound_low, bound_high, bound_low, bound_high, levels))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
