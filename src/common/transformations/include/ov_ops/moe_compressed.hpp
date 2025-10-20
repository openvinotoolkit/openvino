// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
// #include "openvino/op/moe.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief MOECompressed experts that support compressed weights for GEMM3_SWIGLU MOE.
class TRANSFORMATIONS_API MOECompressed : public ov::op::Op {
public:
    OPENVINO_OP("MOECompressed", "ie_internal_opset");

    MOECompressed() = default;

    struct Config {
        ov::element::Type out_type = ov::element::dynamic;  // fp16
        size_t group_size = 0;
    };

    /// \brief Constructs a MOECompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_experts, ...] normalized weights for selected experts
    ///      (input to final multiplication)
    ///   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   4: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   6: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, hidden_size]
    ///   7: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   9: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, inter_size]
    ///   10: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   11: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    /// \param config Configuration for the MOE operation
    MOECompressed(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
