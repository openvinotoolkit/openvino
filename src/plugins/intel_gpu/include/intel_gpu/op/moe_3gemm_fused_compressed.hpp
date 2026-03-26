// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {

/// \brief MOE3GemmFusedCompressed that support compressed and fused MOE for GEMM3_SWIGLU.
class MOE3GemmFusedCompressed : public ov::op::internal::MOECompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressed", "gpu_opset", ov::op::internal::MOECompressed);

    MOE3GemmFusedCompressed() = delete;

    /// \brief Constructs a MOE3GemmFusedCompressed operation with config only
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
    ///   11: routing_bias (optional, SIGMOID_BIAS only) - [1, num_experts] routing bias for sigmoid routing
    ///   12: routing_eps (optional, SIGMOID_BIAS only) - scalar epsilon for normalization (read at kernel compile time)
    /// \param config Configuration for the MOE 3GEMM SWIGLU fused operation
    MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
