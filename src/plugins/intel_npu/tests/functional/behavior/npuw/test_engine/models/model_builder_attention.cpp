// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "model_builder.hpp"
#include "model_builder_internal.hpp"
#include "model_builder_norm.hpp"
#include "model_builder_weights.hpp"
#include "openvino/op/loop.hpp"
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
                                           size_t attn_dim,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn) {
    auto attn_trans = make_attention_transpose(sdpa_output, name + "_transpose");

    auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{0, -1, static_cast<int64_t>(attn_dim)});
    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(name + "_reshape");

    return make_linear(attn_reshaped->output(0), attn_dim, hidden_size, name, precision, weight_fn, bias_fn);
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

    return make_attention_output(attn_output,
                                 hidden_size,
                                 num_heads * head_dim,
                                 prefix + o_proj_name,
                                 precision,
                                 weight_fn,
                                 bias_fn);
}

ov::Output<ov::Node> Attention::operator()(const ov::Output<ov::Node>& input,
                                           const ov::Output<ov::Node>& kv_input,
                                           const std::string& prefix,
                                           size_t layer_idx) const {
    auto kv_src = kv_input.get_node() ? kv_input : input;
    size_t q_dim = num_heads * head_dim;
    size_t kv_dim = num_kv_heads * head_dim;
    auto q = make_linear(input, hidden_size, q_dim, prefix + attn_prefix + "q_proj", precision, weight_fn, bias_fn);
    auto k = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "k_proj", precision, weight_fn, bias_fn);
    auto v = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "v_proj", precision, weight_fn, bias_fn);
    return (*this)(q, k, v, prefix, layer_idx);
}

FixedStateResult make_fixed_state(const ov::Output<ov::Node>& batch_source,
                                  const std::vector<int64_t>& state_dims,
                                  const std::string& name,
                                  ov::element::Type precision,
                                  const ov::Output<ov::Node>& beam_idx) {
    std::vector<int64_t> var_shape_vec = {-1};
    var_shape_vec.insert(var_shape_vec.end(), state_dims.begin(), state_dims.end());
    auto var_shape = ov::PartialShape(var_shape_vec);
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(batch_source, ov::element::i64);
    shape_of->set_friendly_name(name + "_shapeof");
    auto zero_idx = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, zero_idx, gather_axis);
    batch_dim->set_friendly_name(name + "_batch_dim");

    auto dims_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{state_dims.size()}, state_dims);
    auto init_shape =
        std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_dim->output(0), dims_const->output(0)}, 0);
    init_shape->set_friendly_name(name + "_init_shape");

    auto zero_scalar = ov::opset11::Constant::create(precision, ov::Shape{}, {0.0f});
    auto init_value = std::make_shared<ov::opset11::Broadcast>(zero_scalar, init_shape);
    init_value->set_friendly_name(name + "_init");

    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init_value, variable);
    read_value->set_friendly_name(name + "_read");

    ov::Output<ov::Node> output = read_value->output(0);
    if (beam_idx.get_node()) {
        auto beam_gather = std::make_shared<ov::opset11::Gather>(
            output, beam_idx, ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        beam_gather->set_friendly_name(name + "_beam_gather");
        output = beam_gather->output(0);
    }

    return {variable, output};
}

CausalConvResult make_causal_conv(const ov::Output<ov::Node>& input,
                                  const ov::Output<ov::Node>& seq_source,
                                  const ov::Output<ov::Node>& beam_idx,
                                  size_t channels,
                                  size_t kernel_size,
                                  const std::string& state_name,
                                  const std::string& prefix,
                                  ov::element::Type prec,
                                  bool apply_silu) {
    // [batch, seq, C] -> [batch, C, seq]
    auto input_t = std::make_shared<ov::opset11::Transpose>(
        input, ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1}));
    input_t->set_friendly_name(prefix + "transpose_in");

    auto state = make_fixed_state(seq_source,
                                  {static_cast<int64_t>(channels), static_cast<int64_t>(kernel_size)},
                                  state_name,
                                  prec,
                                  beam_idx);

    auto cat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{state.read_value, input_t->output(0)}, 2);
    cat->set_friendly_name(prefix + "concat");

    // State update: keep last kernel_size positions.
    auto new_state = std::make_shared<ov::op::v8::Slice>(
        cat,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-static_cast<int64_t>(kernel_size)}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {INT64_MAX}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    new_state->set_friendly_name(prefix + "new_state");
    auto assign = std::make_shared<ov::op::v6::Assign>(new_state, state.variable);
    assign->set_friendly_name(state_name + "_assign");

    // Depthwise GroupConvolution with quantized weights (u8 → Convert → Subtract(zp) → Multiply(scale))
    // mirroring real Qwen3.5 export — a DCOFF-recognized decompression chain partitioning relies on.
    const ov::Shape w_shape{channels, 1, 1, kernel_size};
    const ov::Shape zs_shape{channels, 1, 1, 1};
    const std::string w_name = prefix + "conv_weight";

    uint32_t w_state = seed_from_name(w_name);
    std::vector<uint8_t> w_u8(channels * kernel_size);
    for (size_t i = 0; i < w_u8.size(); ++i) {
        w_u8[i] = static_cast<uint8_t>(128 + (xorshift32(w_state) % 32u));
    }
    auto w_const = ov::opset11::Constant::create(ov::element::u8, w_shape, w_u8);
    w_const->set_friendly_name(w_name);
    auto w_cvt = std::make_shared<ov::opset11::Convert>(w_const, prec);
    w_cvt->set_friendly_name(w_name + "/convert");

    std::vector<uint8_t> zp_u8(channels, static_cast<uint8_t>(128));
    auto zp_const = ov::opset11::Constant::create(ov::element::u8, zs_shape, zp_u8);
    zp_const->set_friendly_name(w_name + "/zero_point");
    auto zp_cvt = std::make_shared<ov::opset11::Convert>(zp_const, prec);
    zp_cvt->set_friendly_name(w_name + "/zero_point/convert");

    auto w_sub = std::make_shared<ov::opset11::Subtract>(w_cvt, zp_cvt);
    w_sub->set_friendly_name(w_name + "/subtract");

    uint32_t s_state = seed_from_name(w_name + "_scale");
    std::vector<float> sc_vals(channels);
    for (size_t i = 0; i < channels; ++i) {
        sc_vals[i] = 0.05f + 0.01f * static_cast<float>(xorshift32(s_state) % 1000u) / 1000.0f;
    }
    auto sc_const = ov::opset11::Constant::create(prec, zs_shape, sc_vals);
    sc_const->set_friendly_name(w_name + "/scale");
    auto weight = std::make_shared<ov::opset11::Multiply>(w_sub, sc_const);
    weight->set_friendly_name(w_name + "/decompress");

    auto conv = std::make_shared<ov::opset11::GroupConvolution>(cat->output(0),
                                                                weight,
                                                                ov::Strides{1},
                                                                ov::CoordinateDiff{0},
                                                                ov::CoordinateDiff{0},
                                                                ov::Strides{1});
    conv->set_friendly_name(prefix + "group_conv");

    // Drop first output element (valid conv artifact).
    auto sliced = std::make_shared<ov::op::v8::Slice>(
        conv,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {INT64_MAX}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    sliced->set_friendly_name(prefix + "sliced");

    ov::Output<ov::Node> conv_out = sliced->output(0);
    if (apply_silu) {
        auto activated = std::make_shared<ov::opset11::Swish>(sliced);
        activated->set_friendly_name(prefix + "silu");
        conv_out = activated->output(0);
    }

    auto output = std::make_shared<ov::opset11::Transpose>(
        conv_out, ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1}));
    output->set_friendly_name(prefix + "transpose_out");

    CausalConvResult result;
    result.output = output->output(0);
    result.assign = assign;
    return result;
}

// GatedDeltaNet delta rule, one timestep per Loop iteration (matched by FuseGDNLoop):
//   gated_state = h_prev * exp(gate);  delta = v - sum(gated_state * k, -2)
//   h_new = gated_state + k * (delta * beta);  output = sum(h_new * q, -2)
static std::shared_ptr<ov::Model> build_ssm_loop_body(ov::element::Type prec,
                                                       int64_t H,
                                                       int64_t Dk,
                                                       int64_t Dv,
                                                       ov::ParameterVector& params,
                                                       ov::ResultVector& results) {
    auto P = [&](ov::element::Type t, ov::PartialShape s) {
        return params.emplace_back(std::make_shared<ov::op::v0::Parameter>(t, s));
    };
    auto iter = P(ov::element::i32, ov::PartialShape{});
    auto q = P(prec, {-1, H, 1, Dk});
    auto k = P(prec, {-1, H, 1, Dk});
    auto v = P(prec, {-1, H, 1, Dv});
    auto gate = P(prec, {-1, H, 1});
    auto beta = P(prec, {-1, H, 1});
    auto h = P(prec, {-1, H, Dk, Dv});
    auto out_buf = P(prec, {-1, H, -1, Dv});

    auto axis_1d = [](int64_t val) {
        return ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {val});
    };

    auto gate_exp = std::make_shared<ov::opset11::Exp>(gate);
    auto gate_bc = std::make_shared<ov::opset11::Unsqueeze>(gate_exp, axis_1d(-1));
    auto gated_state = std::make_shared<ov::opset11::Multiply>(h, gate_bc);

    auto k_sq = std::make_shared<ov::opset11::Squeeze>(k, axis_1d(2));
    auto k_u = std::make_shared<ov::opset11::Unsqueeze>(k_sq, axis_1d(-1));

    auto proj_mul = std::make_shared<ov::opset11::Multiply>(gated_state, k_u);
    auto proj_sum = std::make_shared<ov::opset11::ReduceSum>(proj_mul, axis_1d(-2), false);

    auto v_sq = std::make_shared<ov::opset11::Squeeze>(v, axis_1d(2));
    auto delta = std::make_shared<ov::opset11::Subtract>(v_sq, proj_sum);

    auto scaled_delta = std::make_shared<ov::opset11::Multiply>(delta, beta);
    auto sd_u = std::make_shared<ov::opset11::Unsqueeze>(scaled_delta, axis_1d(-2));
    auto outer_update = std::make_shared<ov::opset11::Multiply>(k_u, sd_u);

    auto h_new = std::make_shared<ov::opset11::Add>(gated_state, outer_update);

    auto q_sq = std::make_shared<ov::opset11::Squeeze>(q, axis_1d(2));
    auto q_u = std::make_shared<ov::opset11::Unsqueeze>(q_sq, axis_1d(-1));
    auto out_mul = std::make_shared<ov::opset11::Multiply>(h_new, q_u);
    auto y = std::make_shared<ov::opset11::ReduceSum>(out_mul, axis_1d(-2), true);

    // Scatter y into out_buf at seq position = iter.
    auto iter_idx = std::make_shared<ov::opset11::Unsqueeze>(iter, axis_1d(0));
    auto scatter_axis = ov::opset11::Constant::create(ov::element::i32, ov::Shape{}, {2});
    auto out_new = std::make_shared<ov::opset11::ScatterUpdate>(out_buf, iter_idx, y, scatter_axis);

    results = {
        std::make_shared<ov::op::v0::Result>(
            ov::opset11::Constant::create(ov::element::boolean, ov::Shape{}, {true})),
        std::make_shared<ov::op::v0::Result>(h_new),
        std::make_shared<ov::op::v0::Result>(out_new),
    };
    return std::make_shared<ov::Model>(results, params);
}

RecurrentStateResult make_recurrent_state(const ov::Output<ov::Node>& query,
                                          const ov::Output<ov::Node>& key,
                                          const ov::Output<ov::Node>& value,
                                          const ov::Output<ov::Node>& gate_input,
                                          const ov::Output<ov::Node>& beta_input,
                                          const ov::Output<ov::Node>& seq_source,
                                          const ov::Output<ov::Node>& beam_idx,
                                          size_t num_heads,
                                          size_t key_head_dim,
                                          size_t value_head_dim,
                                          const std::string& state_name,
                                          const std::string& prefix,
                                          ov::element::Type prec) {
    const auto H = static_cast<int64_t>(num_heads);
    const auto Dk = static_cast<int64_t>(key_head_dim);
    const auto Dv = static_cast<int64_t>(value_head_dim);

    auto state = make_fixed_state(seq_source, {H, Dk, Dv}, state_name, prec, beam_idx);

    auto q_mh = make_attention_transpose(make_multihead_reshape(query, num_heads, key_head_dim, prefix + "q_rs"),
                                         prefix + "q_mh");
    auto k_mh = make_attention_transpose(make_multihead_reshape(key, num_heads, key_head_dim, prefix + "k_rs"),
                                         prefix + "k_mh");
    auto v_mh = make_attention_transpose(make_multihead_reshape(value, num_heads, value_head_dim, prefix + "v_rs"),
                                         prefix + "v_mh");

    // gate/beta: [batch, seq, H] -> [batch, H, seq]
    auto perm021 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto gate_mh = std::make_shared<ov::opset11::Transpose>(gate_input, perm021);
    auto beta_mh = std::make_shared<ov::opset11::Transpose>(beta_input, perm021);

    auto seq_len = std::make_shared<ov::opset11::Gather>(
        std::make_shared<ov::opset11::ShapeOf>(q_mh, ov::element::i64),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));

    ov::ParameterVector bp;
    ov::ResultVector br;
    auto body = build_ssm_loop_body(prec, H, Dk, Dv, bp, br);
    //  bp: [iter, q, k, v, gate, beta, h, out_buf]    br: [cond, h_new, out_new]

    auto loop = std::make_shared<ov::op::v5::Loop>(
        seq_len, ov::opset11::Constant::create(ov::element::boolean, ov::Shape{}, {true}));
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});

    auto v_shape = std::make_shared<ov::opset11::ShapeOf>(v_mh, ov::element::i64);
    auto zero_scalar = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto out_init = std::make_shared<ov::op::v3::Broadcast>(zero_scalar, v_shape);
    out_init->set_friendly_name(prefix + "out_init");

    ov::OutputVector sliced_srcs = {q_mh, k_mh, v_mh, gate_mh, beta_mh};
    for (size_t i = 0; i < sliced_srcs.size(); ++i)
        loop->set_sliced_input(bp[i + 1], sliced_srcs[i], 0, 1, 1, -1, 2);

    loop->set_merged_input(bp[6], state.read_value, br[1]);  // h back-edge
    loop->set_merged_input(bp[7], out_init, br[2]);          // out_buf back-edge

    auto output = loop->get_iter_value(br[2], -1);   // [batch, H, seq, Dv]
    auto final_h = loop->get_iter_value(br[1], -1);  // [batch, H, Dk, Dv]
    loop->set_friendly_name(prefix + "loop");

    auto assign = std::make_shared<ov::op::v6::Assign>(final_h, state.variable);
    assign->set_friendly_name(state_name + "_assign");

    auto out_t = make_attention_transpose(output, prefix + "output");

    RecurrentStateResult result;
    result.output = out_t;
    result.assign = assign;
    return result;
}

std::function<bool(size_t)> make_mamba_schedule(size_t mamba_ratio) {
    if (mamba_ratio == 0)
        return {};
    return [mamba_ratio](size_t layer_idx) {
        return (layer_idx % (mamba_ratio + 1)) < mamba_ratio;
    };
}

std::function<bool(size_t)> make_schedule_with_attention_at(std::vector<size_t> attn_layers) {
    return [attn = std::move(attn_layers)](size_t layer_idx) {
        return std::find(attn.begin(), attn.end(), layer_idx) == attn.end();
    };
}

GatedShortConv::Result GatedShortConv::operator()(const ov::Output<ov::Node>& input,
                                                  const std::string& prefix,
                                                  size_t linear_layer_idx) const {
    auto layer_str = std::to_string(linear_layer_idx);
    auto ap = prefix + "conv.";
    const auto cd = static_cast<int64_t>(conv_dim);

    // in_proj: hidden -> 3*conv_dim, split into B, C, x gates.
    auto in_proj = make_linear(input, hidden_size, 3 * conv_dim, ap + "in_proj", precision, weight_fn);
    auto split_lens = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{cd, cd, cd});
    auto bcx = std::make_shared<ov::opset11::VariadicSplit>(
        in_proj, ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {-1}), split_lens);
    bcx->set_friendly_name(ap + "split");

    // Bx = B * x, then causal conv (no SiLU — gating is external), then C * conv.
    auto bx = std::make_shared<ov::opset11::Multiply>(bcx->output(0), bcx->output(2));
    bx->set_friendly_name(ap + "B_mul_x");

    auto conv = make_causal_conv(bx, seq_source, beam_idx, conv_dim, conv_kernel,
                                 make_cache_params_var_id("conv", layer_str), ap, precision, /*apply_silu=*/false);

    auto gated = std::make_shared<ov::opset11::Multiply>(bcx->output(1), conv.output);
    gated->set_friendly_name(ap + "C_mul_conv");

    auto output = make_linear(gated->output(0), conv_dim, hidden_size, ap + "out_proj", precision, weight_fn);
    return {output, conv.assign};
}

LinearAttention::Result LinearAttention::operator()(const ov::Output<ov::Node>& input,
                                                    const std::string& prefix,
                                                    size_t linear_layer_idx) const {
    const auto kd = key_dim();
    const auto vd = value_dim();
    const auto cd = conv_dim();
    auto layer_str = std::to_string(linear_layer_idx);
    auto ap = prefix + "linear_attn.";

    auto qkv = make_linear(input, hidden_size, cd, ap + "in_proj_qkv", precision, weight_fn);
    auto a_proj = make_linear(input, hidden_size, num_heads, ap + "in_proj_a", precision, weight_fn);
    auto a_bias = FloatWeight{precision}(ap + "in_proj_a.bias", ov::Shape{1, 1, num_heads}, precision);
    auto a_biased = std::make_shared<ov::opset11::Add>(a_proj, a_bias);
    a_biased->set_friendly_name(ap + "in_proj_a_biased");
    auto b_proj = make_linear(input, hidden_size, num_heads, ap + "in_proj_b", precision, weight_fn);
    auto z_proj = make_linear(input, hidden_size, vd, ap + "in_proj_z", precision, weight_fn);

    auto conv = make_causal_conv(qkv,
                                 seq_source,
                                 beam_idx,
                                 cd,
                                 conv_kernel,
                                 make_cache_params_var_id("conv", layer_str),
                                 ap + "conv.",
                                 precision);

    // QKV split: [batch, seq, conv_dim] -> Q[key_dim], K[key_dim], V[value_dim]
    auto split_lens = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{3},
        std::vector<int64_t>{static_cast<int64_t>(kd), static_cast<int64_t>(kd), static_cast<int64_t>(vd)});
    auto qkv_split = std::make_shared<ov::opset11::VariadicSplit>(
        conv.output, ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {-1}), split_lens);
    qkv_split->set_friendly_name(ap + "qkv_split");

    L2Norm l2norm(precision);
    auto q = l2norm(qkv_split->output(0), ap + "q_l2norm");
    auto k = l2norm(qkv_split->output(1), ap + "k_l2norm");

    // Gate: -SoftPlus(a) — state decay (Exp applied per-timestep inside the Loop body).
    auto sp = std::make_shared<ov::opset11::SoftPlus>(a_biased);
    sp->set_friendly_name(ap + "softplus");
    auto gdn_gate =
        std::make_shared<ov::opset11::Multiply>(sp, ov::opset11::Constant::create(precision, ov::Shape{}, {-1.0f}));
    gdn_gate->set_friendly_name(ap + "gate");

    // Beta: Sigmoid(b) — delta scaling.
    auto gdn_beta = std::make_shared<ov::opset11::Sigmoid>(b_proj);
    gdn_beta->set_friendly_name(ap + "beta");

    auto ssm = make_recurrent_state(q,
                                    k,
                                    qkv_split->output(2),
                                    gdn_gate->output(0),
                                    gdn_beta->output(0),
                                    seq_source,
                                    beam_idx,
                                    num_heads,
                                    key_head_dim,
                                    value_head_dim,
                                    make_cache_params_var_id("ssm", layer_str),
                                    ap + "ssm.",
                                    precision);

    // Flatten heads: [batch, seq, num_heads, value_head_dim] -> [batch, seq, value_dim]
    auto flat = std::make_shared<ov::opset11::Reshape>(
        ssm.output,
        ov::opset11::Constant::create(ov::element::i64,
                                      ov::Shape{3},
                                      std::vector<int64_t>{0, 0, static_cast<int64_t>(vd)}),
        true);
    flat->set_friendly_name(ap + "flat");

    // Output gating: Swish(z) * Norm(output)
    auto normed = out_norm(flat->output(0), ap + "norm");
    auto z_gate = std::make_shared<ov::opset11::Swish>(z_proj);
    z_gate->set_friendly_name(ap + "z_gate");
    auto out_gated = std::make_shared<ov::opset11::Multiply>(z_gate, normed);
    out_gated->set_friendly_name(ap + "out_gated");

    auto output = make_linear(out_gated->output(0), vd, hidden_size, ap + "out_proj", precision, weight_fn);

    return {output, conv.assign, ssm.assign};
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
