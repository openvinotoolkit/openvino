// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_flash_attn_ext(const NodeContext& context) {
    num_inputs_check(context, 3, 5);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    // Decoders that compact null source slots expose mask presence explicitly.
    const bool has_mask = context.get_attribute<bool>("has_mask", context.get_input_size() >= 4);
    const size_t sinks_index = has_mask ? 4 : 3;
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() >= sinks_index &&
                                     context.get_input_size() <= sinks_index + 1,
                                 "FLASH_ATTN_EXT inputs do not match has_mask");
    const bool has_sinks = context.get_input_size() > sinks_index;

    float scale = context.get_attribute<float>("scale");
    float kq_soft_cap = context.get_attribute<float>("kq_soft_cap", 0.0f);
    const float max_bias = context.get_attribute<float>("max_bias", 0.0f);
    FRONT_END_CHECK_IMPLEMENTED(max_bias == 0.0f || context.get_input_size() >= 4,
                                "FLASH_ATTN_EXT with nonzero max_bias requires an attention mask");

    // ggml's FLASH_ATTN_EXT operands are [B, n_head, n_tokens, head_size] (the graph PERMUTEs q/k/v first).
    constexpr size_t head_axis = 1;
    constexpr int token_axis = 2;
    // ggml FLASH_ATTN_EXT accumulates in f16 precision by default (GGML_PREC_DEFAULT), and the plugins'
    // SDPA fusions and micro-kernels expect f16 operands; "f32_attention" keeps f32 (encoders). Soft-capped
    // attention is decomposed in f32 below.
    const auto sdpa_type = context.get_attribute<bool>("f32_attention", false) ? ov::element::f32 : ov::element::f16;
    const auto token_major_source = [](const ov::Output<Node>& value) -> ov::Output<Node> {
        const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(value.get_node_shared_ptr());
        const auto order = transpose ? ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1))
                                     : nullptr;
        if (!order || order->cast_vector<int64_t>() != std::vector<int64_t>{0, 2, 1, 3}) {
            return {};
        }
        return transpose->input_value(0);
    };
    // Q is converted to the SDPA type before its head/token transpose, as K/V are below.
    ov::Output<Node> q;
    if (const auto q_source = token_major_source(q_f32); q_source.get_node_shared_ptr()) {
        q = std::make_shared<ov::op::v1::Transpose>(std::make_shared<ov::op::v0::Convert>(q_source, sdpa_type),
                                                    ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } else {
        q = std::make_shared<ov::op::v0::Convert>(q_f32, sdpa_type);
    }
    auto scale_node = std::make_shared<ov::op::v0::Constant>(sdpa_type, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> mask_sliced, res;
    if (has_mask) {
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto token_len = get_dimensions(q, {token_axis});
        mask_sliced = std::make_shared<ov::op::v8::Slice>(context.get_input(3), zero, token_len, one, two);

        if (mask_sliced.get_element_type() != sdpa_type) {
            mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, sdpa_type);
        }
    } else {
        mask_sliced = ov::op::v0::Constant::create(sdpa_type, Shape{}, {0.0f});
    }

    // Expand K/V from n_head_kv to n_head (GQA): insert a repeat axis after the head axis, broadcast it to
    // `factor`, then fold it back in. `kv_head_axis` is 1 for [B, H, L, S] and 2 for token-major [B, L, H, S].
    auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv, size_t kv_head_axis) {
        int64_t factor = num_heads / num_heads_kv;
        if (factor > 1 && num_heads_kv > 1) {
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {(int64_t)kv_head_axis + 1});
            auto kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);
            std::vector<int64_t> bcast(5, 1);
            bcast[kv_head_axis + 1] = factor;
            auto kv_broadcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, bcast);
            // special_zero keeps the leading dims (incl. the dynamic token axis) as-is.
            std::vector<int64_t> new_shape = kv_head_axis == 2 ? std::vector<int64_t>{0, 0, num_heads, head_size}
                                                               : std::vector<int64_t>{0, num_heads, -1, head_size};
            auto new_kv_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, new_shape);
            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed,
                                                         kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    auto q_shape = q.get_partial_shape();
    auto k_shape = k.get_partial_shape();
    auto v_shape = v.get_partial_shape();
    for (const auto& shape : {q_shape, k_shape, v_shape}) {
        FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4 &&
                                         shape[head_axis].is_static() && shape[head_axis].get_length() > 0 &&
                                         shape[3].is_static(),
                                     "FLASH_ATTN_EXT requires rank-4 inputs with static head dimensions");
    }
    FRONT_END_OP_CONVERSION_CHECK(q_shape[head_axis].get_length() % k_shape[head_axis].get_length() == 0 &&
                                     k_shape[head_axis] == v_shape[head_axis],
                                 "FLASH_ATTN_EXT requires matching K/V heads dividing the query heads");
    if (max_bias > 0.0f && has_mask) {
        // ALiBi: ggml scales the additive mask by a per-head slope (ggml_compute_forward_flash_attn_ext).
        const uint32_t n_head = static_cast<uint32_t>(q_shape[head_axis].get_length());
        const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(n_head)));
        const float m0 = std::pow(2.0f, -max_bias / static_cast<float>(n_head_log2));
        const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<float>(n_head_log2));
        std::vector<float> slopes(n_head);
        for (uint32_t h = 0; h < n_head; ++h) {
            slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1))
                                        : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
        }
        ov::Shape slope_shape(4, 1);
        slope_shape[head_axis] = n_head;
        auto slope_node = std::make_shared<ov::op::v0::Constant>(sdpa_type, slope_shape, slopes);
        mask_sliced = std::make_shared<ov::op::v1::Multiply>(mask_sliced, slope_node);
    }
    // K/V can arrive longer than the active attention window (a full-size cache bound for a sliding window
    // layer or a multi-sequence cache). The mask's key axis is the active window, so trim to it. A dynamic
    // K/V (stateful concat, runtime-sized cache) already has the window's length.
    ov::Output<Node> key_len;
    if (has_mask) {
        const auto mask_shape = context.get_input(3).get_partial_shape();
        const bool mask_rank_ok = mask_shape.rank().is_static() && mask_shape.rank().get_length() == 4;
        if (mask_rank_ok && k_shape[token_axis].is_static() &&
            !(mask_shape[3].is_static() && mask_shape[3] == k_shape[token_axis])) {
            key_len = get_dimensions(context.get_input(3), {3});
        }
    }

    // When K/V are the head/token transpose of a token-major tensor ([B, L, H, S], as PERMUTEd from a KV cache),
    // expand and trim that tensor and transpose afterwards: concat -> GQA tile -> single Transpose -> SDPA is
    // the layout OpenVINO's stateful-SDPA fusion, MakeStateful and SDPAToPagedAttention recognise. Each
    // transpose gets its own order constant (the GPU plugin's TransposeSDPAMatcher requires one consumer).
    const auto prepare_kv = [&](const ov::Output<Node>& kv, int64_t kv_heads, int64_t kv_head_size) {
        const auto source = token_major_source(kv);
        const bool token_major = source.get_node_shared_ptr() != nullptr;
        ov::Output<Node> x = token_major ? source : kv;
        x = tile_kv(q_shape[head_axis].get_length(), kv_heads, kv_head_size, x, token_major ? 2 : 1);
        if (key_len.get_node_shared_ptr()) {
            x = std::make_shared<ov::op::v8::Slice>(x,
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                    key_len,
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                    ov::op::v0::Constant::create(ov::element::i64,
                                                                                 {1},
                                                                                 {token_major ? 1 : token_axis}));
        }
        // SDPA requires q/k/v to share an element type; match k/v to q (ConvertConvertLike lowers these).
        x = std::make_shared<ov::op::v1::ConvertLike>(x, q);
        if (token_major) {
            x = std::make_shared<ov::op::v1::Transpose>(x, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        }
        return x;
    };
    // Explicit grouped-query attention, requested by the decoder for stateless GPU graphs. K and V stay the direct
    // [B, n_head_kv, L, S] transposes of the KV cache, with no broadcast to n_head in between: the GPU plugin then
    // fuses the cache read (StatelessKV) and folds the transposes into the Gemm. The query heads that share a KV head
    // are stacked along the row axis (a pure reshape, they are adjacent) and the mask is tiled to match:
    //   Q [B, n_head, L, S] -> [B, n_head_kv, factor * L, S],  scores [B, n_head_kv, factor * L, Lk]
    // The leading 0 with special_zero copies the (possibly dynamic) batch.
    const int64_t n_head = q_shape[head_axis].get_length();
    const int64_t n_head_kv = k_shape[head_axis].get_length();
    const int64_t gqa_factor = n_head / n_head_kv;
    const bool manual_gqa = context.get_attribute<bool>("manual_gqa_attention", false) && gqa_factor > 1 &&
                            n_head_kv > 1 && kq_soft_cap == 0.0f && !has_sinks && k_shape[3] == q_shape[3] &&
                            v_shape[3] == q_shape[3];

    ov::Output<ov::Node> sdpa;
    if (manual_gqa) {
        const int64_t head_size = q_shape[3].get_length();
        const auto prepare_gqa_kv = [&](ov::Output<Node> kv) {
            if (key_len.get_node_shared_ptr()) {
                kv = std::make_shared<ov::op::v8::Slice>(kv,
                                                         ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                         key_len,
                                                         ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                         ov::op::v0::Constant::create(ov::element::i64, {1}, {token_axis}));
            }
            if (kv.get_element_type() != sdpa_type) {
                kv = std::make_shared<ov::op::v0::Convert>(kv, sdpa_type);
            }
            return kv;
        };
        auto q_r = std::make_shared<ov::op::v1::Reshape>(
            q, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, n_head_kv, -1, head_size}), true);
        ov::Output<Node> scores = std::make_shared<ov::op::v0::MatMul>(q_r, prepare_gqa_kv(k), false, true);
        scores = std::make_shared<ov::op::v1::Multiply>(scores, scale_node);
        if (has_mask) {
            // A finite mask value: the fused SDPA kernel works on blocks of keys and a block fully masked with -inf
            // gives NaN.
            auto mask_finite = std::make_shared<ov::op::v1::Maximum>(
                mask_sliced, ov::op::v0::Constant::create(sdpa_type, ov::Shape{}, std::vector<float>{-30000.0f}));
            auto mask_tiled = std::make_shared<ov::op::v0::Tile>(
                mask_finite,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, gqa_factor, 1}));
            scores = std::make_shared<ov::op::v1::Add>(scores, mask_tiled);
        }
        auto weights = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        auto attn = std::make_shared<ov::op::v0::MatMul>(weights, prepare_gqa_kv(v));
        sdpa = std::make_shared<ov::op::v1::Reshape>(
            attn, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, n_head, -1, head_size}), true);
    } else {
        k = prepare_kv(k, k_shape[head_axis].get_length(), q_shape[3].get_length());
        v = prepare_kv(v, v_shape[head_axis].get_length(), v_shape[3].get_length());
    }

    ov::Output<ov::Node> q_t = q, k_t = k, v_t = v;

    // The GPU plugin's fused SDPA kernel works on blocks of keys: a block fully masked with -inf for a row gives
    // exp(-inf - -inf) = NaN, which spreads through the row (garbage output after a few decode steps). A large
    // finite value masks identically (exp underflows to 0) without the NaN.
    if (has_mask && kq_soft_cap == 0.0f) {
        mask_sliced = std::make_shared<ov::op::v1::Maximum>(
            mask_sliced, ov::op::v0::Constant::create(sdpa_type, ov::Shape{}, std::vector<float>{-30000.0f}));
    }

    if (manual_gqa) {
        // Attention is already computed above.
    } else if (kq_soft_cap != 0.0f) {
        // Gemma2 attention soft-cap: tanh(QK^T * scale * (1/cap)) * cap + mask -> softmax -> *V.
        // OV SDPA v13 has no native softcap parameter, so decompose the attention in f32.
        using namespace ov::op;
        auto q_f32_t = std::make_shared<v0::Convert>(q_t, element::f32);
        auto k_f32_t = std::make_shared<v0::Convert>(k_t, element::f32);
        auto v_f32_t = std::make_shared<v0::Convert>(v_t, element::f32);
        auto mask_f32 = mask_sliced.get_element_type() != element::f32
                            ? std::make_shared<v0::Convert>(mask_sliced, element::f32)->output(0)
                            : mask_sliced;

        // QK^T: [B, H, L, S] x [B, H, S, Lk] -> [B, H, L, Lk]
        auto kT =
            std::make_shared<v1::Transpose>(k_f32_t,
                                            v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2}));
        auto qk = std::make_shared<v0::MatMul>(q_f32_t, kT, false, false);

        // Apply scale * (1/softcap), then tanh, then *softcap
        auto pre_cap_scale = v0::Constant::create(element::f32, Shape{}, std::vector<float>{scale / kq_soft_cap});
        auto qk_scaled = std::make_shared<v1::Multiply>(qk, pre_cap_scale);
        auto qk_tanh = std::make_shared<v0::Tanh>(qk_scaled);
        auto post_cap_scale = v0::Constant::create(element::f32, Shape{}, std::vector<float>{kq_soft_cap});
        auto qk_capped = std::make_shared<v1::Multiply>(qk_tanh, post_cap_scale);

        // Add mask (already sliced to [B, 1, L, Lk] or [B, 1, 1, Lk])
        Output<Node> qk_masked = std::make_shared<v1::Add>(qk_capped, mask_f32);
        auto key_len = get_dimensions(qk_masked, {3});
        if (has_sinks) {
            auto sink = std::make_shared<v0::Convert>(context.get_input(static_cast<int>(sinks_index)), element::f32);
            auto sink_shape = v0::Constant::create(element::i64, {4}, {1, -1, 1, 1});
            auto sink_r = std::make_shared<v1::Reshape>(sink, sink_shape, false);
            auto sink_target = std::make_shared<v0::Concat>(
                OutputVector{get_dimensions(qk_masked, {0, 1, 2}), v0::Constant::create(element::i64, {1}, {1})},
                0);
            auto sink_column = std::make_shared<v3::Broadcast>(sink_r, sink_target);
            qk_masked = std::make_shared<v0::Concat>(OutputVector{qk_masked, sink_column}, 3);
        }

        // Softmax over last axis (key dimension)
        Output<Node> attn_weights = std::make_shared<v8::Softmax>(qk_masked, -1);
        if (has_sinks) {
            attn_weights = std::make_shared<v8::Slice>(attn_weights,
                                                      v0::Constant::create(element::i64, {1}, {0}),
                                                      key_len,
                                                      v0::Constant::create(element::i64, {1}, {1}),
                                                      v0::Constant::create(element::i64, {1}, {3}));
        }

        // Weighted sum over values: [B, H, L, Lk] x [B, H, Lk, S] -> [B, H, L, S]
        auto attn_out_caps = std::make_shared<v0::MatMul>(attn_weights, v_f32_t, false, false);

        sdpa = attn_out_caps;
    } else if (!has_sinks) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_t, k_t, v_t, mask_sliced, scale_node, false);
    } else {
        // gpt-oss attention sinks: a learned per-head logit participates in the softmax
        // denominator (so the attention weights do not sum to 1) but contributes no value. OV
        // SDPA's native 6-input form (q, k, v, mask, scale, sink) folds it into the CPU plugin's
        // online-softmax. The sink logit is per head: [n_head] -> [1, n_head, 1, 1] to broadcast
        // over [B, n_head, q, 1] (rank must equal the query rank, last dim 1).
        using namespace ov::op;
        auto sink = context.get_input(static_cast<int>(sinks_index));
        auto sink_converted = sink.get_element_type() != sdpa_type
                                 ? std::make_shared<v0::Convert>(sink, sdpa_type)->output(0)
                                 : sink;
        auto sink_shape = v0::Constant::create(element::i64,
                                               {4},
                                               std::vector<int64_t>{1, (int64_t)q_shape[head_axis].get_length(), 1, 1});
        auto sink_r = std::make_shared<v1::Reshape>(sink_converted, sink_shape, false);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_t,
                                                                        k_t,
                                                                        v_t,
                                                                        mask_sliced,
                                                                        scale_node,
                                                                        sink_r,
                                                                        false);
    }
    // [B, H, L, S] -> [B, L, H, S] (ggml-natural layout expected by caller).
    res = std::make_shared<ov::op::v1::Transpose>(sdpa,
                                                  ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    if (res.get_element_type() != ov::element::f32) {
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    }
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
