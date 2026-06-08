// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_attention.hpp"

#include <cmath>
#include <vector>

#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                            size_t num_heads,
                                            size_t head_dim,
                                            const std::string& name) {
    auto shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, -1, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)});

    auto reshape = std::make_shared<ov::opset11::Reshape>(input, shape, true);
    reshape->set_friendly_name(name);

    return reshape->output(0);
}

ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input, const std::string& name) {
    auto order_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto transpose = std::make_shared<ov::opset11::Transpose>(input, order_const);
    transpose->set_friendly_name(name);

    return transpose->output(0);
}

ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                    size_t num_heads,
                                    size_t num_kv_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    const ov::Output<ov::Node>& shared_broadcast_shape) {
    const size_t actual_kv_heads = (num_kv_heads == 0) ? num_heads : num_kv_heads;
    const size_t n_rep = num_heads / actual_kv_heads;

    if (!shared_broadcast_shape.get_node() && n_rep == 1) {
        return kv;
    }

    OPENVINO_ASSERT(num_heads % actual_kv_heads == 0,
                    "num_heads (",
                    num_heads,
                    ") must be divisible by num_kv_heads (",
                    actual_kv_heads,
                    ")");

    ov::Output<ov::Node> broadcast_shape_output;
    if (shared_broadcast_shape.get_node()) {
        broadcast_shape_output = shared_broadcast_shape;
    } else {
        // 4-input Concat {Gather, any, any, any} required by NPUW's AttentionBroadcast
        // pattern (sdpa.cpp); a 3-input form only hits the AttentionBroadcast2 fallback.
        auto shape_of_kv = std::make_shared<ov::opset11::ShapeOf>(kv, ov::element::i64);
        shape_of_kv->set_friendly_name(name + "_shapeof");
        auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto idx_01 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto batch_kv_heads = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_01, gather_axis);
        batch_kv_heads->set_friendly_name(name + "_batch_kv_heads");
        auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});
        auto idx_2 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto seq_dim = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_2, gather_axis);
        seq_dim->set_friendly_name(name + "_seq_dim");
        auto head_dim_const =
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
        auto broadcast_shape = std::make_shared<ov::opset11::Concat>(
            ov::OutputVector{batch_kv_heads, n_rep_const, seq_dim, head_dim_const},
            0);
        broadcast_shape->set_friendly_name(name + "_broadcast_shape");
        broadcast_shape_output = broadcast_shape->output(0);
    }

    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(kv, unsqueeze_axis);
    unsqueezed->set_friendly_name(name + "_unsqueeze");

    auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(unsqueezed,
                                                               broadcast_shape_output,
                                                               ov::op::BroadcastType::BIDIRECTIONAL);
    broadcasted->set_friendly_name(name + "_broadcast");

    auto new_shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)});
    new_shape->set_friendly_name(name + "_shape");

    auto reshaped = std::make_shared<ov::opset11::Reshape>(broadcasted, new_shape, true);
    reshaped->set_friendly_name(name);

    return reshaped->output(0);
}

KVCacheReadState make_kv_cache_read(const ov::Output<ov::Node>& batch_source,
                                    const ov::Output<ov::Node>& beam_idx,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision) {
    auto var_shape = ov::PartialShape{-1, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)};
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(batch_source, ov::element::i64);
    shape_of->set_friendly_name(name + "_shapeof");

    auto zero_idx = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});

    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, zero_idx, gather_axis);
    batch_dim->set_friendly_name(name + "_batch_dim");

    auto num_heads_const = ov::opset11::Constant::create(ov::element::i64,
                                                         ov::Shape{1},
                                                         std::vector<int64_t>{static_cast<int64_t>(num_heads)});
    auto zero_seq = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto head_dim_const = ov::opset11::Constant::create(ov::element::i64,
                                                        ov::Shape{1},
                                                        std::vector<int64_t>{static_cast<int64_t>(head_dim)});

    auto init_shape = std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_dim->output(0),
                                                                              num_heads_const->output(0),
                                                                              zero_seq->output(0),
                                                                              head_dim_const->output(0)},
                                                            0);
    init_shape->set_friendly_name(name + "_init_shape");

    auto zero_scalar = ov::opset11::Constant::create(precision, ov::Shape{}, std::vector<float>{0.0f});

    auto init_value = std::make_shared<ov::opset11::Broadcast>(zero_scalar, init_shape);
    init_value->set_friendly_name(name + "_init");

    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init_value, variable);
    read_value->set_friendly_name(name + "_read");

    auto beam_gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});

    auto beam_gather = std::make_shared<ov::opset11::Gather>(read_value, beam_idx, beam_gather_axis);
    beam_gather->set_friendly_name(name + "_beam_gather");

    return {variable, beam_gather->output(0)};
}

KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision) {
    auto read_state = make_kv_cache_read(batch_source, beam_idx, num_heads, head_dim, name, precision);

    auto concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{read_state.beam_gather, current_kv}, 2);
    concat->set_friendly_name(name + "_concat");

    auto assign = std::make_shared<ov::op::v6::Assign>(concat, read_state.variable);
    assign->set_friendly_name(name + "_assign");

    return {concat->output(0), read_state.beam_gather, assign};
}

KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision) {
    auto var_shape = ov::PartialShape{-1, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)};
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    // No Gather/beam reorder — encoder KV is identical across beams
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(encoder_kv, variable);
    read_value->set_friendly_name(name + "_read");

    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    assign->set_friendly_name(name + "_assign");

    return {read_value->output(0), {}, assign};
}

ov::Output<ov::Node> make_shared_gqa_broadcast(const ov::Output<ov::Node>& shape_source,
                                               size_t kv_heads,
                                               size_t num_heads,
                                               size_t head_dim) {
    const size_t n_rep = num_heads / kv_heads;
    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(shape_source, ov::element::i64);
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto idx0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, idx0, axis0);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto seq_dim = std::make_shared<ov::opset11::Gather>(shape_of, idx1, axis0);
    auto kv_heads_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(kv_heads)});
    auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});
    auto head_dim_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
    auto shared_concat = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim, kv_heads_const, n_rep_const, seq_dim, head_dim_const},
        0);
    shared_concat->set_friendly_name("model.shared_gqa_broadcast_shape");
    return shared_concat->output(0);
}

ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask,
                               size_t head_dim_for_scale) {
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    if (head_dim_for_scale > 0 && attention_mask.get_node()) {
        // 5-input SDPA: Q, K, V, mask, scale (required for embedding model pattern matching)
        auto scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_for_scale));
        auto scale = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {scale_val});
        scale->set_friendly_name(name + ".scale");
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, scale, false);
    } else if (attention_mask.get_node()) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, false);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);
    }
    sdpa->set_friendly_name(name + ".sdpa");

    return sdpa->output(0);
}

ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn) {
    auto attn_trans = make_attention_transpose(sdpa_output, name + "_transpose");

    auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{0, -1, static_cast<int64_t>(hidden_size)});
    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(name + "_reshape");

    return make_linear(attn_reshaped->output(0), hidden_size, hidden_size, name, precision, weight_fn, bias_fn);
}

ov::Output<ov::Node> Attention::operator()(const ov::Output<ov::Node>& q,
                                           const ov::Output<ov::Node>& k,
                                           const ov::Output<ov::Node>& v,
                                           const std::string& prefix,
                                           size_t layer_idx) const {
    auto q_reshaped = make_multihead_reshape(q, num_heads, head_dim, prefix + attn_prefix + "q_reshape");
    auto k_reshaped = make_multihead_reshape(k, num_kv_heads, head_dim, prefix + attn_prefix + "k_reshape");
    auto v_reshaped = make_multihead_reshape(v, num_kv_heads, head_dim, prefix + attn_prefix + "v_reshape");

    ov::Output<ov::Node> q_normed = q_reshaped;
    ov::Output<ov::Node> k_normed = k_reshaped;
    if (qk_norm) {
        q_normed = qk_norm(q_reshaped, prefix + attn_prefix + "q_norm");
        k_normed = qk_norm(k_reshaped, prefix + attn_prefix + "k_norm");
    }

    // [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    auto q_trans = make_attention_transpose(q_normed, prefix + attn_prefix + "q_transpose");
    auto k_trans = make_attention_transpose(k_normed, prefix + attn_prefix + "k_transpose");
    auto v_trans = make_attention_transpose(v_reshaped, prefix + attn_prefix + "v_transpose");

    ov::Output<ov::Node> q_roped = q_trans;
    ov::Output<ov::Node> k_roped = k_trans;
    if (rope_fn) {
        q_roped = rope_fn(q_trans, prefix + "q_rope");
        k_roped = rope_fn(k_trans, prefix + "k_rope");
    }

    ov::Output<ov::Node> k_for_attn = k_roped;
    ov::Output<ov::Node> v_for_attn = v_trans;
    if (kv_cache_fn) {
        std::tie(k_for_attn, v_for_attn) = kv_cache_fn(k_roped, v_trans, layer_idx);
    }

    auto k_expanded =
        make_repeat_kv(k_for_attn, num_heads, num_kv_heads, head_dim, prefix + "k_repeat", shared_broadcast_shape);
    auto v_expanded =
        make_repeat_kv(v_for_attn, num_heads, num_kv_heads, head_dim, prefix + "v_repeat", shared_broadcast_shape);

    // 5-input SDPA with explicit scale needed for embedding model pattern matching
    size_t sdpa_scale_dim = shared_broadcast_shape.get_node() ? head_dim : 0;
    auto attn_output =
        make_sdpa(q_roped, k_expanded, v_expanded, prefix + attn_prefix + "attn", sdpa_mask, sdpa_scale_dim);

    return make_attention_output(attn_output, hidden_size, prefix + o_proj_name, precision, weight_fn, bias_fn);
}

ov::Output<ov::Node> Attention::operator()(const ov::Output<ov::Node>& input,
                                           const ov::Output<ov::Node>& kv_input,
                                           const std::string& prefix,
                                           size_t layer_idx) const {
    auto kv_src = kv_input.get_node() ? kv_input : input;
    size_t kv_dim = num_kv_heads * head_dim;
    auto q =
        make_linear(input, hidden_size, hidden_size, prefix + attn_prefix + "q_proj", precision, weight_fn, bias_fn);
    auto k = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "k_proj", precision, weight_fn, bias_fn);
    auto v = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "v_proj", precision, weight_fn, bias_fn);
    return (*this)(q, k, v, prefix, layer_idx);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
