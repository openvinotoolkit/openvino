// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "openvino/op/op.hpp"

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
                        const std::string& k_quant_type = "NONE",
                        const std::string& v_quant_type = "NONE");
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
    // KV cache is quantized when a bit width is set and a K quantization scheme is selected.
    bool is_kv_quantized() const {
        return m_kv_cache_bit_width != 0 && m_k_quant_type != "NONE";
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
};

}  // namespace ov::op::internal
