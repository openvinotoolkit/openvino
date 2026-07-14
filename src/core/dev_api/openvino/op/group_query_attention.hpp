// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"

namespace ov::op::internal {

// ONNX GroupQueryAttention input position enumeration
// Matches com.microsoft.GroupQueryAttention spec
enum class GroupQueryAttentionInputs : int64_t {
    QUERY = 0,                  // Q or packed QKV
    KEY = 1,                    // K (optional if Q is packed QKV)
    VALUE = 2,                  // V (optional if Q is packed QKV)
    PAST_KEY = 3,               // KV cache key (optional)
    PAST_VALUE = 4,             // KV cache value (optional)
    SEQLENS_K = 5,              // Sequence lengths (mandatory)
    TOTAL_SEQUENCE_LENGTH = 6,  // Total sequence length (mandatory)
    COS_CACHE = 7,              // RoPE cos cache (optional, required if do_rotary=1)
    SIN_CACHE = 8,              // RoPE sin cache (optional, required if do_rotary=1)
    POSITION_IDS = 9,           // Position IDs (optional)
    ATTENTION_BIAS = 10,        // Attention bias (optional)
    HEAD_SINK = 11,             // Head sink (optional, required if smooth_softmax=1)
    K_SCALE = 12,  // Quantization scale for K (optional, required if kv_cache_bit_width != 0)
    V_SCALE = 13,  // Quantization scale for V (optional, required if kv_cache_bit_width != 0)
    // Positions 14-15 are reserved (QK-Norm, not supported)
};

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
                        const std::string& k_quant_type = "NONE",
                        const std::string& v_quant_type = "NONE",
                        int64_t local_window_size = -1,
                        bool sliding_window_cache = false,
                        bool smooth_softmax = false,
                        const std::vector<int64_t>& null_input_positions = {});
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
    const std::string& get_k_quant_type() const {
        return m_k_quant_type;
    }
    const std::string& get_v_quant_type() const {
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
        return m_kv_cache_bit_width != 0 && m_k_quant_type != "NONE";
    }

    // Get original input positions that are null/missing and therefore omitted from OV inputs.
    const std::vector<int64_t>& get_null_input_positions() const {
        return m_null_input_positions;
    }

    int64_t get_original_input_count() const {
        return static_cast<int64_t>(get_input_size() + m_null_input_positions.size());
    }

    // Check if a specific original input position exists and was not filtered as NullNode.
    bool has_input(int64_t input_position) const {
        if (input_position < 0 || input_position >= get_original_input_count()) {
            return false;
        }
        return std::find(m_null_input_positions.begin(), m_null_input_positions.end(), input_position) ==
               m_null_input_positions.end();
    }

    // Calculate the actual input index in the OV graph for a given original input position.
    // Accounts for optional inputs that were filtered out (NullNode).
    // For example, if original inputs 3,4 are null and we want position 12,
    // we count null inputs < 12, and return 12 - 2 = 10.
    int64_t get_input_index(int64_t input_position) const {
        int64_t null_before_count = std::count_if(m_null_input_positions.begin(),
                                                  m_null_input_positions.end(),
                                                  [input_position](int64_t null_pos) {
                                                      return null_pos < input_position;
                                                  });
        return input_position - null_before_count;
    }

private:
    int64_t m_num_heads = 0;
    int64_t m_kv_num_heads = 0;
    float m_scale = 0;
    bool m_do_rotary = false;
    bool m_rotary_interleaved = false;
    int64_t m_kv_cache_bit_width = 0;
    std::string m_k_quant_type = "NONE";
    std::string m_v_quant_type = "NONE";
    int64_t m_local_window_size = -1;
    bool m_sliding_window_cache = false;
    bool m_smooth_softmax = false;
    std::vector<int64_t> m_null_input_positions = {};
};

}  // namespace ov::op::internal
