// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

namespace {

ov::Output<ov::Node> reshape_flat_kv(const NodeContext& context,
                                     size_t input_index,
                                     const ov::Output<ov::Node>& flat,
                                     const ov::Output<ov::Node>& attention_size) {
    const char* shape_name = input_index == 1 ? "flat_kv_shape_k" : "flat_kv_shape_v";
    const char* offset_name = input_index == 1 ? "flat_kv_offset_k" : "flat_kv_offset_v";
    const auto shape = context.get_attribute<std::vector<int64_t>>(shape_name, {});
    FRONT_END_OP_CONVERSION_CHECK(shape.size() == 4, "Flat FLASH_ATTN_EXT KV view must expose a rank-4 logical shape");
    const int64_t n_head = shape[1];
    const int64_t head_size = shape[3];
    const int64_t row_size = n_head * head_size;
    FRONT_END_OP_CONVERSION_CHECK(row_size > 0, "Flat FLASH_ATTN_EXT KV view has invalid head geometry");

    const int64_t offset = context.get_attribute<int64_t>(offset_name);
    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {offset});
    auto row_size_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {row_size});
    auto extent = std::make_shared<ov::op::v1::Multiply>(attention_size, row_size_node);
    auto end = std::make_shared<ov::op::v1::Add>(begin, extent);
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto sliced = std::make_shared<ov::op::v8::Slice>(flat, begin, end, one, axis);

    auto target = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{one,
                         attention_size,
                         ov::op::v0::Constant::create(ov::element::i64, {1}, {n_head}),
                         ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size})},
        0);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, target, false);
    return std::make_shared<ov::op::v1::Transpose>(reshaped,
                                                   ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
}

}  // namespace

OutputVector translate_flash_attn_ext(const NodeContext& context) {
    num_inputs_check(context, 3, 5);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    const int op_case = context.get_op_case();
    const bool flat_kv = op_case == 1 || op_case == 2;
    const bool has_mask = context.get_input_size() >= 4;
    // gpt-oss: optional 5th input is the per-head attention sink logit [n_head].
    const bool has_sinks = context.get_input_size() == 5;
    FRONT_END_OP_CONVERSION_CHECK(!has_sinks || has_mask, "FLASH_ATTN_EXT sinks require an attention mask");

    if (flat_kv) {
        const char* length_name = op_case == 1 ? "attention_size" : "attention_size_static";
        FRONT_END_OP_CONVERSION_CHECK(context.has_input(length_name), "Flat FLASH_ATTN_EXT KV requires ", length_name);
        const auto attention_size = context.get_input(length_name);
        k = reshape_flat_kv(context, 1, k, attention_size);
        v = reshape_flat_kv(context, 2, v, attention_size);
    }

    float scale = context.get_attribute<float>("scale");
    float kq_soft_cap = context.get_attribute<float>("kq_soft_cap", 0.0f);
    FRONT_END_OP_CONVERSION_CHECK(has_mask || kq_soft_cap == 0.0f,
                                  "Maskless FLASH_ATTN_EXT does not support a non-zero soft cap");

    const auto sdpa_type = ov::element::f16;
    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, sdpa_type);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(sdpa_type, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> mask_sliced, res;
    // Pick the layer flavor's mask. The cgraph decoder answers the "is_swa" attribute directly; the
    // builder identifies it by the mask input's name (self_kq_mask_swa).
    if (has_mask) {
        auto mask = context.get_input(3);
        const bool is_swa = context.get_attribute<bool>("is_swa", false) ||
                            context.get_input_names()[3].find("swa") != std::string::npos;
        const std::string mask_name = is_swa ? "KQ_mask_swa_sliced" : "KQ_mask_sliced";
        if (context.has_input(mask_name)) {
            mask_sliced = context.get_input(mask_name);
        } else {
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
            auto token_len = get_dimensions(q, {2});
            mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len, one, two);
        }

        if (mask_sliced.get_element_type() != sdpa_type) {
            mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, sdpa_type);
        }
    }

    // The two decoders hand q/k/v over in different layouts, so the head axis and the need for a
    // transpose depend on the op_case:
    //   op_case 0   (llama.cpp cgraph decoder): already PERMUTEd to [B, n_head, n_tokens, head_size],
    //               the canonical SDPA layout -- tile K/V on axis 1, feed SDPA directly.
    //   op_case 100 (native .gguf builder): ggml-natural [B, n_tokens, n_head(_kv), head_size] -- tile
    //               K/V on axis 2 first, then transpose all three. That ordering (concat -> GQA tile
    //               -> single Transpose -> SDPA) is what the CPU plugin's stateful_sdpa_fusion
    //               matches, so the attention fuses into ScaledDotProductAttentionWithKVCache.
    FRONT_END_CHECK_IMPLEMENTED(op_case == 0 || flat_kv || op_case == 100, "Unsupported FLASH_ATTN_EXT case");
    const bool ggml_natural = op_case == 100;
    const size_t head_axis = ggml_natural ? 2 : 1;

    auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
        int64_t factor = num_heads / num_heads_kv;
        if (factor > 1 && num_heads_kv > 1) {
            // Insert the repeat axis right after the head axis, broadcast it to `factor`, then fold
            // it back into the head axis: [.., n_head_kv, ..] -> [.., n_head_kv * factor, ..].
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {(int64_t)head_axis + 1});
            auto kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);
            std::vector<int64_t> bcast(5, 1);
            bcast[head_axis + 1] = factor;
            auto kv_broadcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, bcast);
            // special_zero keeps the leading dims (incl. the dynamic token axis) as-is.
            std::vector<int64_t> new_shape = ggml_natural ? std::vector<int64_t>{0, 0, num_heads, head_size}
                                                          : std::vector<int64_t>{0, num_heads, -1, head_size};
            auto new_kv_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, new_shape);
            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed,
                                                         kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    // Use the static ggml input shapes (get_input_shape), not the live OV node shapes: on the
    // stateful KV-cache path the OV node's batch/seq dims are dynamic (K/V are fed by the cache
    // concat), but the head-count / head-size dims are static ggml facts the decoder knows.
    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    k = tile_kv(q_shape[head_axis], k_shape[head_axis], q_shape[3], k);
    v = tile_kv(q_shape[head_axis], k_shape[head_axis], q_shape[3], v);

    // SDPA requires q/k/v to share an element type; match k/v to q (ConvertConvertLike lowers these).
    k = std::make_shared<ov::op::v1::ConvertLike>(k, q);
    v = std::make_shared<ov::op::v1::ConvertLike>(v, q);

    ov::Output<ov::Node> q_t = q, k_t = k, v_t = v;
    if (ggml_natural) {
        // [B, L, H, S] -> [B, H, L, S] (canonical SDPA layout). Each transpose gets its OWN order
        // constant: the GPU plugin's TransposeSDPAMatcher requires consumers_count(1) on it, and a
        // shared one leaves the permutes in the decode path and blocks the broadcast-into-SDPA fusion.
        auto to_bhls = [] {
            return ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
        };
        q_t = std::make_shared<ov::op::v1::Transpose>(q, to_bhls());
        k_t = std::make_shared<ov::op::v1::Transpose>(k, to_bhls());
        v_t = std::make_shared<ov::op::v1::Transpose>(v, to_bhls());
    }

    ov::Output<ov::Node> sdpa;
    if (kq_soft_cap != 0.0f) {
        // Gemma2 attention soft-cap: tanh(QK^T * scale * (1/cap)) * cap + mask -> softmax -> *V.
        // OV SDPA v13 has no native softcap parameter, so decompose the attention manually in f32
        // (q_t/k_t/v_t are f16 from the transpose above).
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
        auto qk_masked = std::make_shared<v1::Add>(qk_capped, mask_f32);

        // Softmax over last axis (key dimension)
        auto attn_weights = std::make_shared<v8::Softmax>(qk_masked, -1);

        // Weighted sum over values: [B, H, L, Lk] x [B, H, Lk, S] -> [B, H, L, S]
        auto attn_out_caps = std::make_shared<v0::MatMul>(attn_weights, v_f32_t, false, false);

        sdpa = attn_out_caps;
    } else if (!has_sinks && has_mask) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_t, k_t, v_t, mask_sliced, scale_node, false);
    } else if (!has_sinks) {
        // The maskless SDPA overload has no explicit scale input and applies 1/sqrt(D). Fold the
        // requested ggml scale into Q so the resulting score multiplier is exactly `scale`.
        const float q_compensation = scale * std::sqrt(static_cast<float>(q_shape[3]));
        auto compensation = ov::op::v0::Constant::create(sdpa_type, ov::Shape{}, {q_compensation});
        auto q_scaled = std::make_shared<ov::op::v1::Multiply>(q_t, compensation);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled, k_t, v_t, false);
    } else {
        // gpt-oss attention sinks: a learned per-head logit participates in the softmax
        // denominator (so the attention weights do not sum to 1) but contributes no value. OV
        // SDPA's native 6-input form (q, k, v, mask, scale, sink) folds it into the CPU plugin's
        // online-softmax. The sink logit is per head: [n_head] -> [1, n_head, 1, 1] to broadcast
        // over [B, n_head, q, 1] (rank must equal the query rank, last dim 1).
        using namespace ov::op;
        auto sink = context.get_input(4);
        auto sink_f16 = sink.get_element_type() != element::f16
                            ? std::make_shared<v0::Convert>(sink, element::f16)->output(0)
                            : sink;
        auto sink_shape =
            v0::Constant::create(element::i64, {4}, std::vector<int64_t>{1, (int64_t)q_shape[head_axis], 1, 1});
        auto sink_r = std::make_shared<v1::Reshape>(sink_f16, sink_shape, false);
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
    // SDPA paths produce f16; the soft-cap path produces f32 directly.
    if (kq_soft_cap == 0.0f) {
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    }
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
