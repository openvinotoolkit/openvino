// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/op.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {

/// \brief MoERouterFused — GPU-internal op that computes expert selection and
/// routing-weight normalization for Mixture-of-Experts blocks.
///
/// Performs softmax/sigmoid + top-k + normalization on raw router logits,
/// producing the normalized routing weights and selected expert indices
/// consumed by MOECompressed.
///
/// Inputs:
///   0: router_logits  [num_tokens, num_experts]  — output of the router MatMul
///   1: routing_bias   (optional, SIGMOID_BIAS only) [1, num_experts]
///   2: routing_eps    (optional, SIGMOID_BIAS only) scalar constant
///
/// Outputs:
///   0: topk_weights   [num_tokens, top_k]  — normalized routing weights (f16/f32)
///   1: topk_indices   [num_tokens, top_k]  — selected expert indices (i32)
///
class MoERouterFused : public ov::op::Op {
public:
    OPENVINO_OP("MoERouterFused", "gpu_opset");

    enum class RoutingType { SOFTMAX, SIGMOID_BIAS };

    struct Config {
        size_t num_expert = 0;
        size_t top_k = 0;
        RoutingType routing_type = RoutingType::SOFTMAX;
    };

    MoERouterFused() = default;
    MoERouterFused(const ov::OutputVector& args, const Config& config);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    const Config& get_config() const { return m_config; }

private:
    Config m_config;
};

std::ostream& operator<<(std::ostream& s, const MoERouterFused::RoutingType& type);

}  // namespace ov::intel_gpu::op

namespace ov {
template <>
class AttributeAdapter<ov::intel_gpu::op::MoERouterFused::RoutingType>
    : public EnumAttributeAdapterBase<ov::intel_gpu::op::MoERouterFused::RoutingType> {
public:
    AttributeAdapter(ov::intel_gpu::op::MoERouterFused::RoutingType& value)
        : EnumAttributeAdapterBase<ov::intel_gpu::op::MoERouterFused::RoutingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::intel_gpu::op::MoERouterFused::RoutingType>");
    ~AttributeAdapter() override = default;
};

}  // namespace ov
