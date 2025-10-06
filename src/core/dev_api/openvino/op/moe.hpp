// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {
///
/// \brief MOE experts
class OPENVINO_API MOE : public ov::op::Op {
public:
    OPENVINO_OP("MOE");

    MOE() = default;

    enum class Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    struct Config {
        Expert_type expert_type{Expert_type::GEMM2_BIAS_SWIGLU_CLAMP};
        float expert_alpha{0.0f};  // Expert attribute for clamp bounds
        float expert_beta{1.0f};   // Expert attribute for swish beta
    };

    /// \brief Constructs a MOE operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_experts, ...] normalized weights for selected experts
    ///      (input to final multiplication)
    ///   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection, shape [num_experts, inter_size, hidden_size] or
    ///   [num_experts, hidden_size, 2 * inter_size] if fused
    ///   4: w0_bias (optional) - expert bias for first projection,
    ///   shape [num_experts, ...] or empty tensor if not needed
    ///   5: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, hidden_size]
    ///   6: w1_bias (optional) - expert bias for second projection, shape
    ///   [num_experts, ...] or empty tensor if not needed
    ///   7: w2_weight - expert weights for final projection, shape
    ///   [num_experts, hidden_size, inter_size]
    ///   8: w2_bias (optional) - expert bias for final projection
    /// \param config Configuration for the MOE operation
    MOE(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config;
};

}  // namespace ov::op::internal

namespace ov {
std::ostream& operator<<(std::ostream& s, const ov::op::internal::MOE::Expert_type& type);

template <>
class AttributeAdapter<ov::op::internal::MOE::Expert_type>
    : public EnumAttributeAdapterBase<ov::op::internal::MOE::Expert_type> {
public:
    AttributeAdapter(ov::op::internal::MOE::Expert_type& value)
        : EnumAttributeAdapterBase<ov::op::internal::MOE::Expert_type>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::MOE::Expert_type>");
    ~AttributeAdapter() override = default;
};
}  // namespace ov
