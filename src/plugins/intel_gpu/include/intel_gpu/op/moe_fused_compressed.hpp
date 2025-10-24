// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief MOEFusedCompressed that support compressed and fused MOE for GEMM3_SWIGLU.
class MOEFusedCompressed : public ov::op::Op {
public:
    OPENVINO_OP("MOEFusedCompressed", "gpu_opset");

    MOEFusedCompressed() = default;

    struct Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t num_expert = 0;
        size_t top_k = 0;
        size_t group_size = 0;
        ov::element::Type out_type = ov::element::dynamic;  // fp16
    };

    /// \brief Constructs a MOEFusedCompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    ///   2: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   3: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   4: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   6: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   7: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, group_num, group_size]
    ///   9: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///   10: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    /// \param config Configuration for the MOE operation
    MOEFusedCompressed(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config;
};

}  // namespace ov::intel_gpu::op
