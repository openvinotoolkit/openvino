// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// Helper function to perform rotate_half operation
// This is the same logic as translate_rotate_half but returns Output<Node> for reuse
Output<Node> rotate_half_helper(const NodeContext& context, const Output<Node>& x) {
    // Rotates half the hidden dims of the input.
    // Implementation: x1 = x[..., : x.shape[-1] // 2]
    //                 x2 = x[..., x.shape[-1] // 2 :]
    //                 return torch.cat((-x2, x1), dim=-1)

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
    auto const_neg_1_scalar_for_mul = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto neg_1_converted = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1_scalar_for_mul, x2));
    auto neg_x2 = context.mark_node(std::make_shared<v1::Multiply>(x2, neg_1_converted));

    // Concatenate: cat((-x2, x1), dim=-1)
    auto result = context.mark_node(std::make_shared<v0::Concat>(OutputVector{neg_x2, x1}, -1));

    return result;
}

OutputVector translate_apply_rotary_pos_emb(const NodeContext& context) {
    // Applies Rotary Position Embedding to the query and key tensors.
    // Args:
    //     q (torch.Tensor): The query tensor.
    //     k (torch.Tensor): The key tensor.
    //     cos (torch.Tensor): The cosine part of the rotary embedding.
    //     sin (torch.Tensor): The sine part of the rotary embedding.
    //     position_ids (torch.Tensor, optional): Deprecated and unused.
    //     unsqueeze_dim (int, optional, defaults to 1): The dimension along which to unsqueeze cos and sin.
    // Returns:
    //     tuple(torch.Tensor): The query and key tensors rotated using the Rotary Position Embedding.
    
    num_inputs_check(context, 4, 6);
    
    // Get required inputs
    auto q = context.get_input(0);
    auto k = context.get_input(1);
    auto cos = context.get_input(2);
    auto sin = context.get_input(3);
    
    // Get optional unsqueeze_dim (default to 1)
    int64_t unsqueeze_dim = 1;
    if (context.get_input_size() >= 6 && !context.input_is_none(5)) {
        unsqueeze_dim = context.const_input<int64_t>(5);
    }
    
    // Create unsqueeze dimension constant
    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {unsqueeze_dim}));
    
    // Unsqueeze cos and sin for broadcasting
    auto cos_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(cos, unsqueeze_axis));
    auto sin_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(sin, unsqueeze_axis));
    
    // Apply rotation to query tensor
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    auto q_rotated = rotate_half_helper(context, q);
    auto q_cos = context.mark_node(std::make_shared<v1::Multiply>(q, cos_unsqueezed));
    auto q_rot_sin = context.mark_node(std::make_shared<v1::Multiply>(q_rotated, sin_unsqueezed));
    auto q_embed = context.mark_node(std::make_shared<v1::Add>(q_cos, q_rot_sin));
    
    // Apply rotation to key tensor
    // k_embed = (k * cos) + (rotate_half(k) * sin)
    auto k_rotated = rotate_half_helper(context, k);
    auto k_cos = context.mark_node(std::make_shared<v1::Multiply>(k, cos_unsqueezed));
    auto k_rot_sin = context.mark_node(std::make_shared<v1::Multiply>(k_rotated, sin_unsqueezed));
    auto k_embed = context.mark_node(std::make_shared<v1::Add>(k_cos, k_rot_sin));
    
    // Return both transformed tensors (PyTorch returns tuple, OpenVINO represents as multiple outputs)
    return {q_embed, k_embed};
}

OutputVector translate_apply_rotary_pos_emb_fx(const NodeContext& context) {
    // FX variant wraps the result in list_construct
    auto result = translate_apply_rotary_pos_emb(context);
    return {context.mark_node(make_list_construct(result))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov