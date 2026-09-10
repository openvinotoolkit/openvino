// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"

namespace ov::op::internal {

enum class GroupQueryAttentionInputs : size_t {
    QUERY = 0,                  // Q (mandatory)
    KEY = 1,                    // K (mandatory)
    VALUE = 2,                  // V (mandatory)
    PAST_KEY = 3,               // KV cache key (mandatory)
    PAST_VALUE = 4,             // KV cache value (mandatory)
    SEQLENS_K = 5,              // Sequence lengths (mandatory)
    TOTAL_SEQUENCE_LENGTH = 6,  // Total sequence length (mandatory)
    COS_CACHE = 7,              // RoPE cos cache (optional, required if do_rotary=1)
    SIN_CACHE = 8,              // RoPE sin cache (optional, required if do_rotary=1)
    POSITION_IDS = 9,           // Position IDs (optional)
    ATTENTION_BIAS = 10,        // Attention bias (optional)
    HEAD_SINK = 11,             // Head sink (optional, required if smooth_softmax=1)
    K_SCALE = 12,               // Quantization scale for K (optional, required if kv_cache_bit_width != 0)
    V_SCALE = 13,               // Quantization scale for V (optional, required if kv_cache_bit_width != 0)
    // Positions 14-15 are reserved (QK-Norm, not supported)
};

enum class GroupQueryAttentionQuantType {
    NONE,
    PER_TENSOR,
    PER_CHANNEL,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const GroupQueryAttentionQuantType& quant_type);

}  // namespace ov::op::internal

namespace ov {

template <>
class OPENVINO_API AttributeAdapter<op::internal::GroupQueryAttentionQuantType>
    : public EnumAttributeAdapterBase<op::internal::GroupQueryAttentionQuantType> {
public:
    AttributeAdapter(op::internal::GroupQueryAttentionQuantType& value)
        : EnumAttributeAdapterBase<op::internal::GroupQueryAttentionQuantType>(value) {}

    ~AttributeAdapter() override;

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::GroupQueryAttentionQuantType>");
};

}  // namespace ov

namespace ov::op::internal {

// This is an experimental operation that is implemented in the plugins.
class OPENVINO_API GroupQueryAttention : public Op {
public:
    OPENVINO_OP("GroupQueryAttention");

    GroupQueryAttention() = default;
    GroupQueryAttention(const ov::OutputVector& args,
                        int64_t num_heads,
                        int64_t kv_num_heads,
                        float scale,
                        bool do_rotary,
                        bool rotary_interleaved,
                        int64_t kv_cache_bit_width = 0,
                        GroupQueryAttentionQuantType k_quant_type = GroupQueryAttentionQuantType::NONE,
                        GroupQueryAttentionQuantType v_quant_type = GroupQueryAttentionQuantType::NONE,
                        int64_t local_window_size = -1,
                        bool sliding_window_cache = false,
                        bool smooth_softmax = false,
                        bool causal = true);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_num_heads() const {
        return m_num_heads;
    }
    int64_t get_kv_num_heads() const {
        return m_kv_num_heads;
    }
    float get_scale() const {
        return m_scale;
    }
    bool get_do_rotary() const {
        return m_do_rotary;
    }
    bool get_rotary_interleaved() const {
        return m_rotary_interleaved;
    }
    int64_t get_kv_cache_bit_width() const {
        return m_kv_cache_bit_width;
    }
    GroupQueryAttentionQuantType get_k_quant_type() const {
        return m_k_quant_type;
    }
    GroupQueryAttentionQuantType get_v_quant_type() const {
        return m_v_quant_type;
    }
    // Mistral-style local (sliding-window) attention. -1 disables the window (pure causal); a value
    // >= 1 limits each query at absolute position q to keys k in [q - local_window_size + 1, q].
    int64_t get_local_window_size() const {
        return m_local_window_size;
    }
    // When true, the past/present KV buffers are window-sized (capacity C) and rolled with front
    // eviction, keeping only the most recent min(total, C) tokens in cache-relative coordinates.
    bool get_sliding_window_cache() const {
        return m_sliding_window_cache;
    }
    // When true, softmax gains an extra zero logit in its denominator (or a per-head logit when a
    // head_sink input is present), matching the ONNX Runtime smooth-softmax / head-sink behavior.
    bool get_smooth_softmax() const {
        return m_smooth_softmax;
    }
    // KV cache is quantized when a bit width is set and a K quantization scheme is selected.
    bool is_kv_quantized() const {
        return m_kv_cache_bit_width != 0 && m_k_quant_type != GroupQueryAttentionQuantType::NONE;
    }
    // When true (the ONNX Runtime default), each query only attends to keys at or before its own position
    // (plus the optional sliding window). When false, attention is bidirectional: every query attends to all
    // valid keys up to total_sequence_length (only the unused cache tail beyond it is masked).
    bool get_causal() const {
        return m_causal;
    }

private:
    int64_t m_num_heads = 0;
    int64_t m_kv_num_heads = 0;
    float m_scale = 0;
    bool m_do_rotary = false;
    bool m_rotary_interleaved = false;
    int64_t m_kv_cache_bit_width = 0;
    GroupQueryAttentionQuantType m_k_quant_type = GroupQueryAttentionQuantType::NONE;
    GroupQueryAttentionQuantType m_v_quant_type = GroupQueryAttentionQuantType::NONE;
    int64_t m_local_window_size = -1;
    bool m_sliding_window_cache = false;
    bool m_smooth_softmax = false;
    bool m_causal = true;
};

}  // namespace ov::op::internal
