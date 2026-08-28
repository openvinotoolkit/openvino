// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <algorithm>

#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::op::internal {

std::ostream& operator<<(std::ostream& s, const GroupQueryAttentionQuantType& quant_type) {
    return s << as_string(quant_type);
}

}  // namespace ov::op::internal

namespace ov {

template <>
OPENVINO_API EnumNames<op::internal::GroupQueryAttentionQuantType>&
EnumNames<op::internal::GroupQueryAttentionQuantType>::get() {
    static auto enum_names = EnumNames<op::internal::GroupQueryAttentionQuantType>(
        "op::internal::GroupQueryAttentionQuantType",
        {{"NONE", op::internal::GroupQueryAttentionQuantType::NONE},
         {"PER_TENSOR", op::internal::GroupQueryAttentionQuantType::PER_TENSOR},
         {"PER_CHANNEL", op::internal::GroupQueryAttentionQuantType::PER_CHANNEL}});
    return enum_names;
}

AttributeAdapter<op::internal::GroupQueryAttentionQuantType>::~AttributeAdapter() = default;

}  // namespace ov

namespace ov::op::internal {

GroupQueryAttention::GroupQueryAttention(const OutputVector& args,
                                         int64_t num_heads,
                                         int64_t kv_num_heads,
                                         float scale,
                                         bool do_rotary,
                                         bool rotary_interleaved,
                                         int64_t kv_cache_bit_width,
                                         GroupQueryAttentionQuantType k_quant_type,
                                         GroupQueryAttentionQuantType v_quant_type,
                                         int64_t local_window_size,
                                         bool sliding_window_cache,
                                         bool smooth_softmax,
                                         bool causal)
    : Op(args),
      m_num_heads(num_heads),
      m_kv_num_heads(kv_num_heads),
      m_scale(scale),
      m_do_rotary(do_rotary),
      m_rotary_interleaved(rotary_interleaved),
      m_kv_cache_bit_width(kv_cache_bit_width),
      m_k_quant_type(k_quant_type),
      m_v_quant_type(v_quant_type),
      m_local_window_size(local_window_size),
      m_sliding_window_cache(sliding_window_cache),
      m_smooth_softmax(smooth_softmax),
      m_causal(causal) {
    constructor_validate_and_infer_types();
}

void GroupQueryAttention::validate_and_infer_types() {
    OV_OP_SCOPE(GroupQueryAttention_validate_and_infer_types);
    // GroupQueryAttention expects the following inputs:
    // query, key, value, past_key, past_value, seqlens_k, total_sequence_length.
    // In the rotary embedding case, following inputs are also required: cos_cache, sin_cache.
    // All qkv tensors should have the shape [batch, num_heads, seq_len, head_size] ([B, N, S, H]).
    // The operation produces three outputs:
    // 1. Output tensor of shape [B, S, N * H].
    // 2. Present_key tensor of shape [B, N, S, H].
    // 3. Present_value tensor of shape [B, N, S, H].
    // Note: seqlens_k represents the number of 1's in the attention_mask minus 1.

    const auto input_name = [](GroupQueryAttentionInputs input) -> const char* {
        switch (input) {
        case GroupQueryAttentionInputs::QUERY:
            return "query";
        case GroupQueryAttentionInputs::KEY:
            return "key";
        case GroupQueryAttentionInputs::VALUE:
            return "value";
        case GroupQueryAttentionInputs::PAST_KEY:
            return "past_key";
        case GroupQueryAttentionInputs::PAST_VALUE:
            return "past_value";
        case GroupQueryAttentionInputs::SEQLENS_K:
            return "seqlens_k";
        case GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH:
            return "total_sequence_length";
        case GroupQueryAttentionInputs::COS_CACHE:
            return "cos_cache";
        case GroupQueryAttentionInputs::SIN_CACHE:
            return "sin_cache";
        case GroupQueryAttentionInputs::POSITION_IDS:
            return "position_ids";
        case GroupQueryAttentionInputs::ATTENTION_BIAS:
            return "attention_bias";
        case GroupQueryAttentionInputs::HEAD_SINK:
            return "head_sink";
        case GroupQueryAttentionInputs::K_SCALE:
            return "k_scale";
        case GroupQueryAttentionInputs::V_SCALE:
            return "v_scale";
        }
        return "unknown";
    };

    const auto check_input = [&](GroupQueryAttentionInputs input,
                                 std::initializer_list<Rank> allowed_ranks,
                                 const std::vector<element::Type>& allowed_types,
                                 bool required = true) {
        const auto pos = static_cast<size_t>(input);
        const bool present = (pos < get_input_size()) && !ov::util::is_empty_constant_tensor(input_value(pos));
        NODE_VALIDATION_CHECK(this,
                              !required || present,
                              "GroupQueryAttention requires ",
                              input_name(input),
                              " (ONNX input ",
                              pos,
                              ") to be present");

        if (!present) {
            return;
        }

        const auto& rank = get_input_partial_shape(pos).rank();
        const auto& type = get_input_element_type(pos);

        NODE_VALIDATION_CHECK(
            this,
            rank.is_dynamic() || allowed_ranks.size() == 0 || ov::util::is_rank_compatible_any_of(rank, allowed_ranks),
            "Rank of `",
            input_name(input),
            "` input is not compatible with allowed ranks; got ",
            rank,
            ".");

        NODE_VALIDATION_CHECK(this,
                              type.is_dynamic() || allowed_types.empty() ||
                                  std::find(allowed_types.begin(), allowed_types.end(), type) != allowed_types.end(),
                              "Element type of `",
                              input_name(input),
                              "` input is not compatible with allowed element types; got ",
                              type,
                              ".");
    };

    const auto integral_types = []() {
        std::vector<element::Type> types;
        for (const auto& type_info : element::Type::get_known_types()) {
            if (type_info->is_integral_number()) {
                types.push_back(*type_info);
            }
        }
        return types;
    }();

    NODE_VALIDATION_CHECK(this, m_num_heads > 0, "GroupQueryAttention expects num_heads > 0, got: ", m_num_heads);
    NODE_VALIDATION_CHECK(this,
                          m_kv_num_heads > 0,
                          "GroupQueryAttention expects kv_num_heads > 0, got: ",
                          m_kv_num_heads);

    // Base input checks in input_check-style form: required + rank/type whitelist.
    check_input(GroupQueryAttentionInputs::QUERY, {4}, {element::f16, element::f32});
    check_input(GroupQueryAttentionInputs::KEY, {4}, {element::f16, element::f32});
    check_input(GroupQueryAttentionInputs::VALUE, {4}, {element::f16, element::f32});
    check_input(GroupQueryAttentionInputs::PAST_KEY,
                {4},
                {element::f32, element::f16, element::i8, element::u8, element::f8e4m3});
    check_input(GroupQueryAttentionInputs::PAST_VALUE,
                {4},
                {element::f32, element::f16, element::i8, element::u8, element::f8e4m3});
    check_input(GroupQueryAttentionInputs::SEQLENS_K, {1, 2}, integral_types);
    check_input(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH, {0, 1}, integral_types, false);

    if (m_do_rotary) {
        check_input(GroupQueryAttentionInputs::COS_CACHE, {2}, {}, true);
        check_input(GroupQueryAttentionInputs::SIN_CACHE, {2}, {}, true);
        check_input(GroupQueryAttentionInputs::POSITION_IDS, {1, 2}, integral_types, false);
    }
    check_input(GroupQueryAttentionInputs::ATTENTION_BIAS, {4}, {}, false);

    const auto q_shape = get_input_partial_shape(static_cast<size_t>(GroupQueryAttentionInputs::QUERY));
    const auto past_k_shape = get_input_partial_shape(static_cast<size_t>(GroupQueryAttentionInputs::PAST_KEY));

    const auto& q_type = get_input_element_type(static_cast<size_t>(GroupQueryAttentionInputs::QUERY));

    const auto& batch_size = q_shape[0];
    const auto& sequence_len = q_shape[2];
    const auto& head_size = q_shape[3];

    if (m_do_rotary) {
        // Partial RoPE (GPT-NeoX/Phi-style): rotary_dim = 2 * cos_cache.shape[-1] may be smaller than
        // head_size; only the leading rotary_dim channels are rotated (see
        // GroupQueryAttentionDecomposition::rotaryEmbedding). Bound-check it here so an invalid model
        // fails with a diagnostic instead of a raw shape-inference crash inside the decomposition.
        const auto cos_shape = get_input_partial_shape(static_cast<size_t>(GroupQueryAttentionInputs::COS_CACHE));
        const auto& cos_last_dim = cos_shape[1];
        if (head_size.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  cos_last_dim.is_static(),
                                  "GroupQueryAttention: cos_cache last dimension must be statically known when "
                                  "head_size is static, got cos_cache shape ",
                                  cos_shape);
            const auto rotary_dim = 2 * cos_last_dim.get_length();
            NODE_VALIDATION_CHECK(this,
                                  rotary_dim <= head_size.get_length(),
                                  "GroupQueryAttention: rotary_dim (2 * cos_cache.shape[-1] = ",
                                  rotary_dim,
                                  ") must not exceed head_size (",
                                  head_size.get_length(),
                                  ")");
        }
    }

    // The op is reachable directly from a loaded IR, so mirror the ONNX frontend's sliding-window
    // preconditions here as well: local_window_size must be -1 (disabled) or >= 1, and a windowed cache
    // (sliding_window_cache) requires a real window (>= 1). This keeps the op and the frontend in agreement.
    NODE_VALIDATION_CHECK(this,
                          m_local_window_size == -1 || m_local_window_size >= 1,
                          "GroupQueryAttention: local_window_size must be -1 (disabled) or >= 1, got ",
                          m_local_window_size);
    NODE_VALIDATION_CHECK(this,
                          !m_sliding_window_cache || m_local_window_size >= 1,
                          "GroupQueryAttention: sliding_window_cache requires local_window_size >= 1, got ",
                          m_local_window_size);
    // causal=0 (bidirectional) is mutually exclusive with a sliding window, matching the ONNX Runtime
    // precondition (gqa_attention_base.h: causal_ || local_window_size_ == -1).
    NODE_VALIDATION_CHECK(this,
                          m_causal || m_local_window_size == -1,
                          "GroupQueryAttention: local_window_size requires causal=1, got causal=0 and "
                          "local_window_size=",
                          m_local_window_size);
    // Windowed cache: single-token decode (sequence_length == 1) always fits inside the window and is handled
    // by the in-place Gather + ScatterUpdate assembly. Any other step (a multi-token prefill/staging chunk,
    // whether or not it actually crosses a window eviction at runtime) is handled by the staging branch, which
    // seeds a temporary over-sized buffer with the resident survivors, appends the new tokens, runs attention
    // against it, and writes back only the surviving tail - the same math ONNX Runtime's own
    // PlanWindowedKvCache/staging path uses, verified against it. That branch is chosen from whether
    // sequence_length is provably 1, not from whether it is statically known: a *statically* known
    // sequence_length > 1 takes the identical staging branch a dynamic sequence_length resolving to the same
    // runtime value would, so it is not rejected here.

    // The decomposition derives a scalar past length (past_seqlen = total - current) and assumes a single
    // batch entry ("Only consider batch is 1"); with batch_size > 1 the per-batch past lengths differ and the
    // attention mask / cache indexing would be silently wrong. The batch dimension is dynamic in the usual
    // dynamic-shape deployments (CPU/GPU), which cannot be checked here, so reject only a statically known
    // batch_size > 1 rather than the whole dynamic path.
    if (batch_size.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              batch_size.get_length() == 1,
                              "GroupQueryAttention is only supported for batch_size == 1 when the batch dimension is "
                              "statically known, got batch_size = ",
                              batch_size.get_length(),
                              ".");
    }

    // The KV cache (past_key/past_value, input 3/4) may be quantized. present_key/present_value inherit the
    // cache element type so a quantized (i8/u8/f8e4m3) cache round-trips from past to present, matching the ONNX spec.
    const auto& kv_cache_type = get_input_element_type(static_cast<size_t>(GroupQueryAttentionInputs::PAST_KEY));
    const auto& past_value_type = get_input_element_type(static_cast<size_t>(GroupQueryAttentionInputs::PAST_VALUE));
    NODE_VALIDATION_CHECK(this,
                          kv_cache_type.compatible(past_value_type),
                          "GroupQueryAttention expects past_key and past_value element types to match, got ",
                          kv_cache_type,
                          " and ",
                          past_value_type);
    if (is_kv_quantized()) {
        NODE_VALIDATION_CHECK(
            this,
            kv_cache_type == element::i8 || kv_cache_type == element::u8 || kv_cache_type == element::f8e4m3,
            "GroupQueryAttention expects quantized KV cache element type to be one of ",
            "{i8, u8, f8e4m3}, got ",
            kv_cache_type);
        // Quantized KV cache: i8 (8-bit), u8 (4-bit values packed two per byte), or f8e4m3 (8-bit float).
        NODE_VALIDATION_CHECK(this,
                              m_kv_cache_bit_width == 8 || m_kv_cache_bit_width == 4,
                              "GroupQueryAttention supports kv_cache_bit_width of 8 or 4, got: ",
                              m_kv_cache_bit_width);
        NODE_VALIDATION_CHECK(this,
                              m_k_quant_type == m_v_quant_type,
                              "GroupQueryAttention requires matching k_quant_type and v_quant_type, got: ",
                              as_string(m_k_quant_type),
                              " and ",
                              as_string(m_v_quant_type));
        NODE_VALIDATION_CHECK(this,
                              m_k_quant_type == GroupQueryAttentionQuantType::PER_TENSOR ||
                                  m_k_quant_type == GroupQueryAttentionQuantType::PER_CHANNEL,
                              "GroupQueryAttention supports k/v quant types: {PER_TENSOR, PER_CHANNEL}, got: ",
                              as_string(m_k_quant_type));

        check_input(GroupQueryAttentionInputs::K_SCALE, {0, 1, 2, 3, 4}, {element::f32, element::f16}, true);
        check_input(GroupQueryAttentionInputs::V_SCALE, {0, 1, 2, 3, 4}, {element::f32, element::f16}, true);
    }

    // present_key/present_value keep the past KV layout: kv head size may differ from query head size
    // when the cache is quantized (e.g. 4-bit values packed two per byte).
    auto kv_shape = PartialShape{batch_size, m_kv_num_heads, past_k_shape[2], past_k_shape[3]};
    auto& output_kv_len = kv_shape[2];
    // A windowed KV cache keeps the past buffer's own (capacity) sequence dimension: it rolls in place
    // with front eviction instead of growing. Otherwise present = past + current.
    if (!m_sliding_window_cache && (output_kv_len.is_dynamic() || sequence_len.is_dynamic())) {
        output_kv_len += sequence_len;
    }

    set_output_type(0, q_type, PartialShape{batch_size, sequence_len, head_size * m_num_heads});
    for (auto&& port : {1, 2}) {
        set_output_type(port, kv_cache_type, kv_shape);
    }
}

bool GroupQueryAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GroupQueryAttention_visit_attributes);
    visitor.on_attribute("causal", m_causal);
    visitor.on_attribute("do_rotary", m_do_rotary);
    visitor.on_attribute("k_quant_type", m_k_quant_type);
    visitor.on_attribute("kv_cache_bit_width", m_kv_cache_bit_width);
    visitor.on_attribute("kv_num_heads", m_kv_num_heads);
    visitor.on_attribute("local_window_size", m_local_window_size);
    visitor.on_attribute("num_heads", m_num_heads);
    visitor.on_attribute("rotary_interleaved", m_rotary_interleaved);
    visitor.on_attribute("scale", m_scale);
    visitor.on_attribute("sliding_window_cache", m_sliding_window_cache);
    visitor.on_attribute("smooth_softmax", m_smooth_softmax);
    visitor.on_attribute("v_quant_type", m_v_quant_type);
    return true;
}

std::shared_ptr<ov::Node> GroupQueryAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(GroupQueryAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GroupQueryAttention>(new_args,
                                                 m_num_heads,
                                                 m_kv_num_heads,
                                                 m_scale,
                                                 m_do_rotary,
                                                 m_rotary_interleaved,
                                                 m_kv_cache_bit_width,
                                                 m_k_quant_type,
                                                 m_v_quant_type,
                                                 m_local_window_size,
                                                 m_sliding_window_cache,
                                                 m_smooth_softmax,
                                                 m_causal);
}

}  // namespace ov::op::internal
