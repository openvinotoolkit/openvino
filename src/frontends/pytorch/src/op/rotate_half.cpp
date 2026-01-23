// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rotate_half(const NodeContext& context) {
    // Rotates half the hidden dims of the input.
    // Implementation: x1 = x[..., : x.shape[-1] // 2]
    //                 x2 = x[..., x.shape[-1] // 2 :]
    //                 return torch.cat((-x2, x1), dim=-1)
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);

    // Get input shape to determine split point
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));

    // Create constants for slicing
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_0_scalar = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
    auto const_neg_1_scalar = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto int_max = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()}));

    // Get last dimension size: shape[-1]
    auto last_dim = context.mark_node(std::make_shared<v7::Gather>(shape, const_neg_1_scalar, const_0_scalar));

    // Calculate split point: last_dim // 2
    auto split_point = context.mark_node(std::make_shared<v1::Divide>(last_dim, const_2));

    // Create slice for first half: x[..., :split_point]
    // Start: [0], End: [split_point], Step: [1], Axes: [-1]
    auto x1 = context.mark_node(std::make_shared<v8::Slice>(x, const_0, split_point, const_1, const_neg_1));

    // Create slice for second half: x[..., split_point:]
    // Start: [split_point], End: [int_max], Step: [1], Axes: [-1]
    auto x2 = context.mark_node(std::make_shared<v8::Slice>(x, split_point, int_max, const_1, const_neg_1));

    // Negate x2: -x2
    auto neg_1_converted = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1_scalar, x2));
    auto neg_x2 = context.mark_node(std::make_shared<v1::Multiply>(x2, neg_1_converted));

    // Concatenate: cat((-x2, x1), dim=-1)
    auto result = context.mark_node(std::make_shared<v0::Concat>(OutputVector{neg_x2, x1}, -1));

    return {result};
}

OutputVector translate_rotate_half_fx(const NodeContext& context) {
    // FX variant wraps the result in list_construct
    auto result = translate_rotate_half(context);
    return {context.mark_node(make_list_construct(result))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov