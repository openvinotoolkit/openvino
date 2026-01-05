// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_npu::op {

/// \brief Operation for FlashAttention implementation of SDPA
class FlashAttentionTile : public ::ov::op::Op {
public:
    OPENVINO_OP("FlashAttentionTile", "intel_npu", Op);

    struct Config {
        bool is_head = false;
        bool is_tail = false;
    };

    FlashAttentionTile() = default;

    FlashAttentionTile(const OutputVector& inputs, Config config);

    FlashAttentionTile(const Output<Node>& query,
                       const Output<Node>& key,
                       const Output<Node>& value,
                       const Output<Node>& running_output,
                       const Output<Node>& running_max,
                       const Output<Node>& running_sum,
                       const Output<Node>& attn_mask,
                       const Output<Node>& scale,
                       Config config);

    FlashAttentionTile(const Output<Node>& query,
                       const Output<Node>& key,
                       const Output<Node>& value,
                       const Output<Node>& running_output,
                       const Output<Node>& running_max,
                       const Output<Node>& running_sum,
                       const Output<Node>& attn_mask,
                       Config config);

    FlashAttentionTile(const Output<Node>& query,
                       const Output<Node>& key,
                       const Output<Node>& value,
                       const Output<Node>& running_output,
                       const Output<Node>& running_max,
                       const Output<Node>& running_sum,
                       Config config);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    bool has_evaluate() const override;
    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const override;

    const Config& get_config() const;
    void set_config(Config config);

private:
    Config m_config{};
};

}  // namespace ov::intel_npu::op
