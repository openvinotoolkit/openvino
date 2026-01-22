// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_repeat_kv(const NodeContext& context) {
    // This function implements the repeat_kv operation for Grouped Query Attention
    // Input 0: hidden_states tensor with shape [batch, num_key_value_heads, seqlen, head_dim]
    // Input 1: n_rep (number of repetitions)
    // Output: expanded tensor with shape [batch, num_attention_heads, seqlen, head_dim]
    //         where num_attention_heads = num_key_value_heads * n_rep
    
    num_inputs_check(context, 2, 2);
    auto hidden_states = context.get_input(0);
    auto n_rep = context.get_input(1);
    
    // Step 3: Handle n_rep == 1 case (early return)
    // If n_rep is 1, no expansion is needed
    if (!context.input_is_none(1)) {
        try {
            auto n_rep_const = context.const_input<int64_t>(1);
            if (n_rep_const == 1) {
                return {hidden_states};
            }
        } catch (...) {
            // If n_rep is not a constant, continue with dynamic implementation
        }
    }
    
    // Step 4: Get shape of hidden_states and extract dimensions
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(hidden_states, element::i64));
    
    // Create constants for indexing
    auto const_0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {2}));
    auto const_3 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {3}));
    
    // Extract dimensions using Gather
    auto batch = context.mark_node(std::make_shared<v8::Gather>(shape, const_0, const_0));
    auto num_kv_heads = context.mark_node(std::make_shared<v8::Gather>(shape, const_1, const_0));
    auto seqlen = context.mark_node(std::make_shared<v8::Gather>(shape, const_2, const_0));
    auto head_dim = context.mark_node(std::make_shared<v8::Gather>(shape, const_3, const_0));
    
    // Step 5: Insert new dimension for repetition
    // Unsqueeze hidden_states at dimension 2: [batch, num_kv_heads, seqlen, head_dim] -> [batch, num_kv_heads, 1, seqlen, head_dim]
    auto unsqueeze_axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {2}));
    auto unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(hidden_states, unsqueeze_axis));
    
    // Step 6: Expand along new dimension
    // Create target shape for broadcast: [batch, num_kv_heads, n_rep, seqlen, head_dim]
    auto target_shape = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{batch, num_kv_heads, n_rep, seqlen, head_dim}, 0));
    
    // Broadcast using numpy mode
    auto expanded = context.mark_node(std::make_shared<v3::Broadcast>(unsqueezed, target_shape, "numpy"));
    
    // Step 7: Reshape to final output shape
    // Compute num_attention_heads = num_kv_heads * n_rep
    auto num_attention_heads = context.mark_node(std::make_shared<v1::Multiply>(num_kv_heads, n_rep));
    
    // Create final shape: [batch, num_attention_heads, seqlen, head_dim]
    auto final_shape = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{batch, num_attention_heads, seqlen, head_dim}, 0));
    
    // Reshape to final output
    auto result = context.mark_node(std::make_shared<v1::Reshape>(expanded, final_shape, false));
    
    // Step 8: Return output
    return {result};
}

OutputVector translate_repeat_kv_fx(const NodeContext& context) {
    // FX variant wraps the result in list_construct
    auto result = translate_repeat_kv(context);
    return {context.mark_node(make_list_construct(result))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov