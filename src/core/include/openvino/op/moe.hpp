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

namespace ov::op::v16 {
///
/// \brief MOE experts
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MOE : public ov::op::Op {
public:
    OPENVINO_OP("MOE", "opset16");

    MOE() = default;

    enum class Expert_type { GEMM3_SWIGLU, GEMM2_BIAS_SWIGLU_CLAMP };

    struct Config {
        Expert_type expert_type{Expert_type::GEMM2_BIAS_SWIGLU_CLAMP};
        float expert_alpha{1.0f};  // Expert attribute, e.g. sigmoid alpha
        float expert_beta{0.0f};   // Expert attribute, e.g. clamp limit
    };

    /// \brief Constructs a MOE operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: router_topk_output_weights - normalized weights for selected experts (input to final multiplication)
    ///   2: router_topk_output_indices - indices of selected top-k experts
    ///   3: w0_weight - expert weights for first projection, shape [num_experts, inter_size, hidden_size] or
    ///   [num_experts, hidden_size, 2 * inter_size] if fused 4: w0_bias (optional) - expert bias for first projection,
    ///   shape [num_experts, ...] or empty tensor if not needed 5: w1_weight - expert weights for second projection,
    ///   shape [num_experts, inter_size, hidden_size] 6: w1_bias (optional) - expert bias for second projection, shape
    ///   [num_experts, ...] or empty tensor if not needed 7: w2_weight - expert weights for final projection, shape
    ///   [num_experts, hidden_size, inter_size] 8: w2_bias (optional/redundant) - expert bias for final projection,
    ///   usually not required
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

}  // namespace ov::op::v16

namespace ov {
OPENVINO_API
std::ostream& operator<<(std::ostream& s, const ov::op::v16::MOE::Expert_type& type);

template <>
class OPENVINO_API
    AttributeAdapter<ov::op::v16::MOE::Expert_type> : public EnumAttributeAdapterBase<ov::op::v16::MOE::Expert_type> {
public:
    AttributeAdapter(ov::op::v16::MOE::Expert_type& value)
        : EnumAttributeAdapterBase<ov::op::v16::MOE::Expert_type>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::v16::MOE::Expert_type>");
    ~AttributeAdapter() override = default;
};
}  // namespace ov
