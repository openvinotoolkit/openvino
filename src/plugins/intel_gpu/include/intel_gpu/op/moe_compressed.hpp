// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief MOECompressed experts that support compressed weights for GEMM3_SWIGLU MOE.
class MOECompressed : public ov::op::internal::MOE {
public:
    OPENVINO_OP("MOECompressed", "gpu_opset", ov::op::internal::MOE);

    MOECompressed() = default;
    MOECompressed(const OutputVector& args) : MOE(args) {}

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
        Config() = default;
        Config(const MOE::Config& moe_config) : MOE::Config(moe_config) {}
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

    const Config& get_config() const;
    void set_config(const Config& config);
    void set_scale_factor(float scale_factor) {
        m_config.scale_factor = scale_factor;
        MOE::set_scale_factor(scale_factor);
    }
    float get_scale_factor() const {
        return m_config.scale_factor;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    Config m_config;
};

std::ostream& operator<<(std::ostream& s, const MOECompressed::RoutingType& type);

}  // namespace ov::intel_gpu::op

namespace ov {
template <>
class AttributeAdapter<ov::intel_gpu::op::MOECompressed::RoutingType> : public EnumAttributeAdapterBase<ov::intel_gpu::op::MOECompressed::RoutingType> {
public:
    AttributeAdapter(ov::intel_gpu::op::MOECompressed::RoutingType& value) : EnumAttributeAdapterBase<ov::intel_gpu::op::MOECompressed::RoutingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::intel_gpu::op::MOECompressed::RoutingType>");
    ~AttributeAdapter() override = default;
};
}  // namespace ov
