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

    enum class Expert_type {
        GEMM3_SWIGLU,
        GEMM2_BIAS_SWIGLU_CLAMP
    };

    struct Config {
        size_t topk{};
        size_t expert_num{};
        size_t hidden_size{};
        size_t intermediate_size{};
        size_t group_size{};              // quantized group size, 0 for no group size. same for gate/up/down
        ov::element::Type weight_type{};  // same for gate/up/down
        ov::element::Type scale_type{};   // same for gate/up/down
        ov::element::Type zp_type{};      // same for gate/up/down

        Expert_type expert_type{Expert_type::GEMM2_BIAS_SWIGLU_CLAMP};
        float expert_alpha{1.0f};  // Expert attribute, e.g. sigmoid alpha (gpt-oss: 1.702)
        float expert_beta{0.0f};   // Expert attribute, e.g. clamp limit (gpt-oss: 7.0)

        bool operator==(const Config& rhs) const {
            return std::tie(topk,
                            expert_num,
                            hidden_size,
                            intermediate_size,
                            group_size,
                            weight_type,
                            scale_type,
                            zp_type) == std::tie(rhs.topk,
                                                 rhs.expert_num,
                                                 rhs.hidden_size,
                                                 rhs.intermediate_size,
                                                 rhs.group_size,
                                                 rhs.weight_type,
                                                 rhs.scale_type,
                                                 rhs.zp_type);
        }
    };

    /// \brief Constructs a MOE operation with config only
    /// \param args The input tensors: [hidden_states, router_logits] followed by expert weights/scales/zps
    /// \param config Configuration for the MOE operation
    MOE(const OutputVector& args, const Config& config);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief Get expert weight/scale/zp constant for a specific expert and weight type
    /// \param expert_idx Index of the expert (0 to expert_num-1)
    /// \param weight_type 0=gate, 1=up, 2=down
    /// \param const_type 0=weight, 1=scale, 2=zp
    /// \return Constant node or nullptr if not present
    std::shared_ptr<ov::op::v0::Constant> get_expert_const(size_t expert_idx,
                                                           size_t weight_type,
                                                           size_t const_type) const;

private:
    Config m_config;
};

}  // namespace ov::op::v16
