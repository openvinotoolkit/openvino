// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_avg_poolnd(NodeContext& context) {
    num_inputs_check(context, 6, 7);
    auto input = context.get_input(0);
    auto kernel = context.const_input<Shape>(1);
    auto strides = context.const_input<Strides>(2);
    auto pads = context.const_input<Shape>(3);  // pytorch supports only symmetric padding
    auto rounding_type = context.const_input<bool>(4) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;
    auto count_include_pad = context.const_input<bool>(5);
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(6),
                                  "Translation for aten::avg_pool2d do not support divisor_override input.");
    // Although ov::AvgPool provides exclude_pad=false,
    // The corner case of Average Pooling with ceil_mode on
    // PyTorch allows sliding window go off bound, which leads to this accommodation.
    // More detail on https://github.com/pytorch/pytorch/issues/57178
    if (count_include_pad) {
        auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        auto zero_i32 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        Output<Node> rank;
        std::tie(std::ignore, rank) = get_shape_rank(context, input);
        auto pad_values = context.get_input(3);
        auto pads_len = context.mark_node(v0::Constant::create(element::i32, Shape{}, {pads.size()}));
        auto pads_diff = context.mark_node(std::make_shared<v1::Subtract>(rank, pads_len));
        auto pads_remaining = context.mark_node(std::make_shared<v3::Broadcast>(zero_i32, pads_diff));
        auto padding = context.mark_node(
            std::make_shared<v0::Concat>(NodeVector{pads_remaining, pad_values.get_node_shared_ptr()}, 0));
        input = context.mark_node(std::make_shared<v1::Pad>(input, padding, padding, zero, ov::op::PadMode::CONSTANT));
        pads = Shape(pads.size(), 0);
    }

    return {context.mark_node(
        std::make_shared<v1::AvgPool>(input, strides, pads, pads, kernel, !count_include_pad, rounding_type))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov