// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <algorithm>

#include "itt.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/constant.hpp"

namespace ov::op::internal {

GroupQueryAttention::GroupQueryAttention(const OutputVector& args,
                                         int64_t num_heads,
                                         int64_t kv_num_heads,
                                         float scale,
                                         bool do_rotary,
                                         bool rotary_interleaved,
                                         int64_t kv_cache_bit_width,
                                         const std::string& k_quant_type,
                                         const std::string& v_quant_type,
                                         int64_t local_window_size,
                                         bool sliding_window_cache,
                                         bool smooth_softmax)
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
      m_smooth_softmax(smooth_softmax) {
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
        case GroupQueryAttentionInputs::ATTENTION_MASK:
            return "attention_mask";
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
                                 std::initializer_list<int64_t> allowed_ranks,
                                 const std::vector<element::Type>& allowed_types,
                                 bool required = true) {
        const auto pos = static_cast<int64_t>(input);
        const bool present = has_input(pos);
        NODE_VALIDATION_CHECK(this,
                              !required || present,
                              "GroupQueryAttention requires ",
                              input_name(input),
                              " (ONNX input ",
                              pos,
                              ") to be present");
        if (!present) {
            return false;
        }

        const auto& pshape = get_input_partial_shape(static_cast<size_t>(input));
        const auto& rank = pshape.rank();
        const bool rank_ok = rank.is_dynamic() || allowed_ranks.size() == 0 ||
                             std::any_of(allowed_ranks.begin(), allowed_ranks.end(), [&](int64_t allowed_rank) {
                                 return rank.compatible(allowed_rank);
                             });
        NODE_VALIDATION_CHECK(this,
                              rank_ok,
                              "GroupQueryAttention expects ",
                              input_name(input),
                              " rank to be one of the allowed values, got shape ",
                              pshape);

        const auto& type = get_input_element_type(static_cast<size_t>(input));
        const bool type_ok = type.is_dynamic() || allowed_types.size() == 0 ||
                             std::find(allowed_types.begin(), allowed_types.end(), type) != allowed_types.end();
        NODE_VALIDATION_CHECK(this,
                              type_ok,
                              "GroupQueryAttention expects ",
                              input_name(input),
                              " element type to be one of the allowed values, got ",
                              type);
        return true;
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
    check_input(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH, {0, 1}, integral_types);

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
    // Windowed cache: single-token decode (sequence_length == 1) is always correct, and a multi-token step
    // that fits inside the window (past + current <= capacity) also decomposes correctly. The unmodeled case
    // is a multi-token step that crosses a window eviction (the staging regime ONNX Runtime runs against a
    // temporary larger buffer). Whether a step crosses depends on the past length, which is a runtime value
    // (derived from seqlens_k), so it cannot be decided from shapes alone. A dynamic sequence_length is
    // therefore left enabled (CPU/GPU): at runtime it is typically decode or fitting prefill, and rejecting it
    // would disable those. A *statically* known sequence_length > 1 is a genuine multi-token graph whose
    // correctness we cannot guarantee (it may cross an eviction at runtime), so reject it up front; only
    // sequence_length == 1 is provably safe statically.
    if (m_sliding_window_cache && sequence_len.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              sequence_len.get_length() == 1,
                              "GroupQueryAttention: sliding_window_cache with a statically known sequence length is "
                              "only supported for single-token decode (sequence_length == 1), got ",
                              sequence_len.get_length(),
                              " (the multi-token staging regime is not yet modelled).");
    }

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
    if (is_kv_quantized()) {
        // Quantized KV cache: i8 (8-bit), u8 (4-bit values packed two per byte), or f8e4m3 (8-bit float).
        NODE_VALIDATION_CHECK(this,
                              m_kv_cache_bit_width == 8 || m_kv_cache_bit_width == 4,
                              "GroupQueryAttention supports kv_cache_bit_width of 8 or 4, got: ",
                              m_kv_cache_bit_width);
        NODE_VALIDATION_CHECK(this,
                              m_k_quant_type == m_v_quant_type,
                              "GroupQueryAttention requires matching k_quant_type and v_quant_type, got: ",
                              m_k_quant_type,
                              " and ",
                              m_v_quant_type);
        NODE_VALIDATION_CHECK(this,
                              kv_cache_type.bitwidth() == static_cast<size_t>(m_kv_cache_bit_width),
                              "GroupQueryAttention with quantized KV cache requires a past/present KV type "
                              "with bitwidth matching kv_cache_bit_width (",
                              m_kv_cache_bit_width,
                              "), got: ",
                              kv_cache_type);
        NODE_VALIDATION_CHECK(this,
                              m_k_quant_type == "PER_TENSOR" || m_k_quant_type == "PER_CHANNEL",
                              "GroupQueryAttention supports k/v quant types: {PER_TENSOR, PER_CHANNEL}, got: ",
                              m_k_quant_type);

        check_input(GroupQueryAttentionInputs::K_SCALE, {0, 1}, {element::f32, element::f16}, true);
        check_input(GroupQueryAttentionInputs::V_SCALE, {0, 1}, {element::f32, element::f16}, true);
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

bool GroupQueryAttention::has_input(int64_t input_position) const {
    if (input_position < 0 || input_position >= static_cast<int64_t>(get_input_size())) {
        return false;
    }
    const auto input_node = input_value(static_cast<size_t>(input_position)).get_node_shared_ptr();
    const auto constant = ov::as_type_ptr<v0::Constant>(input_node);
    return !(constant && ov::shape_size(constant->get_shape()) == 0);
}

bool GroupQueryAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(GroupQueryAttention_visit_attributes);
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
                                                 m_smooth_softmax);
}

}  // namespace ov::op::internal
