// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/moe.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {

/// \brief MOECompressed experts that support compressed weights for GEMM3_SWIGLU MOE.
class TRANSFORMATIONS_API MOECompressed : public ov::op::internal::MOE {
public:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE);

    enum class RoutingType { SOFTMAX, SIGMOID_BIAS };

    struct Config : public MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t num_expert = 0;
        size_t num_shared_expert = 0;
        size_t top_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group).
        // other non-zero value means group compression with this given group_size.
        size_t group_size = 0;
        // In CB, intermediate shapes are expanded to {SeqLen, 1, HiddenSize}
        // In Non-CB, intermediate shapes are expanded to {Batch, SeqLen, HiddenSize}
        size_t has_batch_dim = 0;
        bool has_zp = false;
        ov::element::Type out_type = ov::element::dynamic;
        RoutingType routing_type = RoutingType::SOFTMAX;
    };

    /// \brief Constructs a MOECompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - normalized routing weights for selected experts.
    ///      Supports both:
    ///        * legacy "scattered" layout: [num_experts, ...] (one slice per expert)
    ///        * compact post-GatherMatmul layout, consistent with ov::op::MOE
    ///          routing_weights as documented in openvino/op/moe.hpp.
    ///      In all cases, the layout must be compatible with router_topk_output_indices
    ///      and the MOE configuration (top_k, num_expert, etc.).
    ///   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   4: w0_scale - expert scale for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   5: w0_zp - expert zp for first projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   6: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, group_num, group_size]
    ///   7: w1_scale - expert scale for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   8: w1_zp - expert zp for second projection for compressed experts,
    ///   shape [num_experts, inter_size, group_num, 1]
    ///   9: w2_weight - expert weights for final projection,
    ///   shape [num_experts, hidden_size, group_num, group_size]
    ///   10: w2_scale - expert scale for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    ///   11: w2_zp - expert zp for final projection for compressed experts,
    ///   shape [num_experts, hidden_size, group_num, 1]
    /// \param config Configuration for the MOE operation
    MOECompressed(const OutputVector& args, const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    Config m_config;
};

TRANSFORMATIONS_API std::ostream& operator<<(std::ostream& s, const MOECompressed::RoutingType& type);

}  // namespace ov::op::internal

namespace ov {
template <>
class AttributeAdapter<ov::op::internal::MOECompressed::RoutingType>
    : public EnumAttributeAdapterBase<ov::op::internal::MOECompressed::RoutingType> {
public:
    AttributeAdapter(ov::op::internal::MOECompressed::RoutingType& value)
        : EnumAttributeAdapterBase<ov::op::internal::MOECompressed::RoutingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::MOECompressed::RoutingType>");
    ~AttributeAdapter() override = default;
};

}  // namespace ov
