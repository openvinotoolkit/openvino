// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_fake_quantize_per_tensor_affine(const NodeContext& context) {
    num_inputs_check(context, 4, 5);
    auto input_node = context.get_input(0);
    auto scale = std::make_shared<v0::Convert>(context.get_input(1), element::f32);
    auto zero_point = std::make_shared<v0::Convert>(context.get_input(2), element::f32);
    auto out_low_input = context.get_input(3);
    auto out_high_input = context.get_input(4);
    auto out_low = std::make_shared<v0::Convert>(out_low_input, element::f32);
    auto out_high = std::make_shared<v0::Convert>(out_high_input, element::f32);
    // Set default levels value to 256. This would be correct for bounds like: 0-255 and -128-127
    int levels = 256;
    if (std::dynamic_pointer_cast<v0::Constant>(out_low_input.get_node_shared_ptr()) &&
        std::dynamic_pointer_cast<v0::Constant>(out_high_input.get_node_shared_ptr())) {
        // If bounds are both constant, calculate levels value depending on values.
        auto out_low_const = context.const_input<int64_t>(3);
        auto out_high_const = context.const_input<int64_t>(4);
        levels = abs(out_high_const - out_low_const) + 1;
    }
    auto out_low_normalized = std::make_shared<v1::Subtract>(out_low, zero_point);
    auto out_high_normalized = std::make_shared<v1::Subtract>(out_high, zero_point);
    auto bound_a = std::make_shared<v1::Multiply>(scale, out_low_normalized);
    auto bound_b = std::make_shared<v1::Multiply>(scale, out_high_normalized);
    auto bound_high = std::make_shared<v1::Maximum>(bound_a, bound_b);
    auto bound_low = std::make_shared<v1::Minimum>(bound_a, bound_b);
    return {context.mark_node(
        std::make_shared<v0::FakeQuantize>(input_node, bound_low, bound_high, bound_low, bound_high, levels))};
}

OutputVector translate_fake_quantize_per_channel_affine(const NodeContext& context) {
    auto input_node = context.get_input(0);
    auto scale = std::make_shared<v0::Convert>(context.get_input(1), element::f32);
    auto zero_point = std::make_shared<v0::Convert>(context.get_input(2), element::f32);
    auto axis = context.get_input(3);
    auto out_low_input = context.get_input(4);
    auto out_high_input = context.get_input(5);
    auto out_low = std::make_shared<v0::Convert>(out_low_input, element::f32);
    auto out_high = std::make_shared<v0::Convert>(out_high_input, element::f32);
    // Set default levels value to 256. This would be correct for bounds like: 0-255 and -128-127
    int levels = 256;
    if (std::dynamic_pointer_cast<v0::Constant>(out_low_input.get_node_shared_ptr()) &&
        std::dynamic_pointer_cast<v0::Constant>(out_high_input.get_node_shared_ptr())) {
        // If bounds are both constant, calculate levels value depending on values.
        auto out_low_const = context.const_input<int64_t>(4);
        auto out_high_const = context.const_input<int64_t>(5);
        levels = abs(out_high_const - out_low_const) + 1;
    }
    auto shape_rank = get_shape_rank(context, input_node);
    auto shape = std::get<0>(shape_rank);
    auto rank = std::get<1>(shape_rank);

    auto const_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
    auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});

    auto ones = std::make_shared<v3::Broadcast>(const_1, rank);
    auto axis_rank = std::make_shared<v1::Add>(axis, rank);
    auto is_less = std::make_shared<v1::Less>(axis_rank, rank);
    auto new_axis = std::make_shared<v1::Select>(is_less, axis_rank, axis);
    auto new_shape = std::make_shared<v3::ScatterElementsUpdate>(ones, new_axis, const_neg_1, const_0);
    auto scale_bc = std::make_shared<v1::Reshape>(scale, new_shape, false);
    auto zero_point_bc = std::make_shared<v1::Reshape>(zero_point, new_shape, false);
    auto out_low_normalized = std::make_shared<v1::Subtract>(out_low, zero_point_bc);
    auto out_high_normalized = std::make_shared<v1::Subtract>(out_high, zero_point_bc);
    auto bound_a = std::make_shared<v1::Multiply>(scale_bc, out_low_normalized);
    auto bound_b = std::make_shared<v1::Multiply>(scale_bc, out_high_normalized);
    auto bound_high = std::make_shared<v1::Maximum>(bound_a, bound_b);
    auto bound_low = std::make_shared<v1::Minimum>(bound_a, bound_b);
    return {context.mark_node(
        std::make_shared<v0::FakeQuantize>(input_node, bound_low, bound_high, bound_low, bound_high, levels))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
