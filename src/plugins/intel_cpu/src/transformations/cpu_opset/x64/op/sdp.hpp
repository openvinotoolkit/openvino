// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {
/// \brief Scaled dot product attention from PyTorch, fused with RoPE & Read/AssignValue
///
/// \ingroup ov_ops_cpp_api

class OPENVINO_API ScaledDotProductAttentionNode : public ov::op::Op {
public:
    OPENVINO_OP("ScaledDotProductAttentionNode", "cpu_plugin_opset");

    ScaledDotProductAttentionNode() = default;

    struct Config {
        bool qkv_merged = false;        // qkv is merged in [B, L, H, Sq + Sk + Sv]
                                        // if merged, qkv will be in a single
        bool input_trans0213 = false;   // transpose input before *cos/*sin
        bool cos_is_raw3d = false;      // cos input is [B,L,ndims/2]
        bool sin_is_raw3d = false;      // sin input is [B,L,ndims/2]
        bool output_BLHxS = false;      // true implies that output is [B,L,H*S]
        size_t rope_ndims = 0;          // how many dimensions used for RoPE
        int gather_position_arg_id = 0;

        bool fuse_causal_attn = false;  // fuse causal mask and attn mask into attn_mask
        bool is_causal = false;         // apply causal mask internally
        bool fuse_big_pattern = false;  // fuse big pattern

        int num_heads;                  // number of heads
        int num_states_per_head;        // states per head

        // for accessing State Variable
        std::shared_ptr<ov::op::util::Variable> past_key_var;    // [B, H, past_seq_len, S]
        std::shared_ptr<ov::op::util::Variable> past_value_var;  // [B, H, past_seq_len, S]
    };

    ScaledDotProductAttentionNode(const OutputVector& args, const Config& cfg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const Config& get_config() const {
        return m_config;
    }

    Config& get_config() {
        return m_config;
    }

private:
    Config m_config;
};

}  // namespace intel_cpu
}  // namespace ov
