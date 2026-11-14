// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
ov::OutputVector prepare_inputs_to_sdpa(const NodeContext& context,
                                        const Output<Node>& query,
                                        const Output<Node>& key,
                                        const Output<Node>& value,
                                        const Output<Node>& scale,
                                        const Output<Node>& attn_mask) {
    OutputVector inputs = {query, key, value};

    if (attn_mask.get_node()) {
        inputs.push_back(attn_mask);
    }
    if (scale.get_node()) {
        if (!attn_mask.get_node()) {
            auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
            auto attn_mask_default = context.mark_node(std::make_shared<v1::ConvertLike>(zero, query));
            inputs.push_back(attn_mask_default);
        }
        inputs.push_back(scale);
    }

    return inputs;
}

// Number of heads (axis -3) of an SDPA input [B..., H, L, E], dynamic when unknown at conversion time.
Dimension get_num_heads(const Output<Node>& input) {
    const auto& shape = input.get_partial_shape();
    return shape.rank().is_static() ? shape[-3] : Dimension::dynamic();
}

// Elements [begin, end) of a 1-D shape tensor; negative bounds count from the end.
Output<Node> slice_dims(const NodeContext& context, const Output<Node>& shape, int64_t begin, int64_t end) {
    auto begin_const = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {begin}));
    auto end_const = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {end}));
    auto step = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    return context.mark_node(std::make_shared<v8::Slice>(shape, begin_const, end_const, step));
}

// Repeats the heads of key or value to match the query, like torch repeat_interleave(Hq / Hkv, dim=-3):
//   [B..., Hkv, S, E] -> Unsqueeze -> [B..., Hkv, 1, S, E] -> Broadcast -> [B..., Hkv, G, S, E]
//                     -> Reshape -> [B..., Hq, S, E]
// This is the graph that aten::expand produces for transformers' repeat_kv, which the KV-cache fusions match
// (ov::op::util::match_multi_query_bcst, GPU ExpandBroadcastReshapeSDPAFusion, SDPAToPagedAttention). The broadcast
// target is a Concat of kv's own dims whose second-to-last input is S, as NPUW's find_kv_cache_concat expects. A
// single ShapeOf of kv feeds every shape computation: CPU StatefulSDPAFusion accepts one ShapeOf consumer on the
// KV-cache Concat.
Output<Node> repeat_kv_heads(const NodeContext& context, const Output<Node>& kv, const Output<Node>& query) {
    const auto num_q_heads = get_num_heads(query);
    const auto num_kv_heads = get_num_heads(kv);
    const bool static_heads = num_q_heads.is_static() && num_kv_heads.is_static();
    if (static_heads) {
        PYTORCH_OP_CONVERSION_CHECK(
            num_kv_heads.get_length() > 0 && num_q_heads.get_length() % num_kv_heads.get_length() == 0,
            "Number of heads in key and value must divide the number of heads in query, got ",
            num_kv_heads,
            " and ",
            num_q_heads);
        if (num_q_heads == num_kv_heads) {
            // An all-ones Broadcast would be removed by EliminateNopBroadcast, leaving Unsqueeze -> Reshape,
            // which the KV-cache fusions don't match. SDPA takes the input as is.
            return kv;
        }
    }

    auto kv_shape = context.mark_node(std::make_shared<v3::ShapeOf>(kv, element::i64));
    Output<Node> q_heads, group;
    if (static_heads) {
        q_heads = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {num_q_heads.get_length()}));
        group = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {num_q_heads.get_length() / num_kv_heads.get_length()}));
    } else {
        auto q_shape = context.mark_node(std::make_shared<v3::ShapeOf>(query, element::i64));
        q_heads = slice_dims(context, q_shape, -3, -2);
        group = context.mark_node(std::make_shared<v1::Divide>(q_heads, slice_dims(context, kv_shape, -3, -2)));
    }
    auto seq_len = slice_dims(context, kv_shape, -2, -1);
    auto head_size = slice_dims(context, kv_shape, -1, std::numeric_limits<int64_t>::max());
    auto target_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{slice_dims(context, kv_shape, 0, -2), group, seq_len, head_size}, 0));
    auto output_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{slice_dims(context, kv_shape, 0, -3), q_heads, seq_len, head_size},
                                     0));
    auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-3}));
    auto unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(kv, axis));
    auto broadcasted =
        context.mark_node(std::make_shared<v3::Broadcast>(unsqueezed, target_shape, BroadcastType::BIDIRECTIONAL));
    return context.mark_node(std::make_shared<v1::Reshape>(broadcasted, output_shape, false));
}

// enable_gqa: query [B..., Hq, L, E], key [B..., Hk, S, E], value [B..., Hv, S, Ev], with Hk and Hv dividing Hq
// (they may differ). Key and value heads are repeated to Hq separately, then a regular SDPA of the input rank is
// created, so the KV-cache fusions see the same graph as for transformers' repeat_kv.
std::shared_ptr<Node> decompose_gqa(const NodeContext& context,
                                    const Output<Node>& query,
                                    const Output<Node>& key,
                                    const Output<Node>& value,
                                    const Output<Node>& scale,      // optional
                                    const Output<Node>& attn_mask,  // optional
                                    bool is_causal) {
    for (const auto& input : {query, key, value}) {
        const auto& rank = input.get_partial_shape().rank();
        PYTORCH_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() >= 3,
                                    "scaled_dot_product_attention with enable_gqa=True expects inputs of rank 3 or "
                                    "higher, got rank ",
                                    rank);
    }
    // The mask is broadcastable to [B..., Hq, L, S] and is applied as is.
    auto inputs = prepare_inputs_to_sdpa(context,
                                         query,
                                         repeat_kv_heads(context, key, query),
                                         repeat_kv_heads(context, value, query),
                                         scale,
                                         attn_mask);
    return context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal));
}

}  // namespace

std::shared_ptr<ov::Node> translate_scaled_dot_product_attention_common(const NodeContext& context) {
    auto query = context.get_input(0);
    auto key = context.get_input(1);
    auto value = context.get_input(2);

    auto is_causal = context.input_is_none(5) ? false : context.const_input<bool>(5);

    ov::Output<ov::Node> attn_mask{};
    ov::Output<ov::Node> scale{};
    if (!context.input_is_none(3)) {
        attn_mask = context.get_input(3);
    }
    if (!context.input_is_none(6)) {
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(6), query));
    } else if (context.has_attribute("scale")) {
        scale = context.get_input("scale");
        scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, query));
    }
    if (get_const_input_or_attribute(context, 7, "enable_gqa", false)) {
        return decompose_gqa(context, query, key, value, scale, attn_mask, is_causal);
    }

    auto inputs = prepare_inputs_to_sdpa(context, query, key, value, scale, attn_mask);
    return context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal));
}

OutputVector translate_scaled_dot_product_attention(const NodeContext& context) {
    // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float
    // dropout_p=0., bool is_causal=False, float scale=None, bool enable_gqa=False)
    num_inputs_check(context, 3, 8);
    return {translate_scaled_dot_product_attention_common(context)};
};

OutputVector translate_scaled_dot_product_attention_fx(const NodeContext& context) {
    // torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg1_1, arg2_1, arg3_1, 0.0, True, attn_mask =
    // arg0_1, scale = 5.0)
    // aten._scaled_dot_product_flash_attention.default(arg0_1, arg1_1, arg2_1, 0.0, True, scale = 5.0)
    num_inputs_check(context, 3, 5);
    const auto query = context.get_input(0);
    const auto key = context.get_input(1);
    const auto value = context.get_input(2);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    OutputVector inputs{query, key, value};
    // Index 3 is dropout
    auto causal = false;
    if (context.has_attribute("is_causal")) {
        causal = context.get_attribute<bool>("is_causal");
    } else if (!context.input_is_none(4)) {
        causal = context.const_input<bool>(4);
    }
    if (context.has_attribute("attn_mask")) {
        const auto attn_mask = context.get_input("attn_mask");
        inputs.push_back(attn_mask);
    } else if (context.has_attribute("scale")) {
        // if scale exist but attn_mask no, add zero input to fill in gap
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(zero, query)));
    }
    if (context.has_attribute("scale")) {
        const auto scale = context.get_input("scale");
        inputs.push_back(context.mark_node(std::make_shared<v1::ConvertLike>(scale, query)));
    }
    auto sdpa = context.mark_node(std::make_shared<v13::ScaledDotProductAttention>(inputs, causal));
    return {context.mark_node(make_list_construct({sdpa}))};
};

}  // namespace ov::frontend::pytorch::op
