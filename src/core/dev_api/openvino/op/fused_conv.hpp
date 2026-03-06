// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
//
// FusedConv fuses Gather(beam_idx) + Concat + GroupConv + SiLU + Slice for depthwise causal conv
// used in Qwen3.5 linear attention layers.
//
// Inputs:
//   0: input         [B, conv_dim, S]           - mixed_qkv tensor
//   1: conv_weight   [conv_dim, kernel_size]    - per-channel 1D conv weights
//   2: beam_idx      [B]                        - beam search reorder indices (i32/i64)
//   3: initial_state [B, conv_dim, kernel_size] - fallback state when variable is not set yet
//
// Outputs:
//   0: conv_output   [B, conv_dim, S]           - GroupConv + SiLU result
//   1: updated_state [B, conv_dim, kernel_size] - conv state (typically mapped to variable memory)
class OPENVINO_API FusedConv : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("FusedConv");

    FusedConv() = default;

    FusedConv(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace op
}  // namespace ov
