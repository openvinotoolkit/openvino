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
        Expert_type expert_type{Expert_type::GEMM2_BIAS_SWIGLU_CLAMP};
        float expert_alpha{1.0f};  // Expert attribute, e.g. sigmoid alpha
        float expert_beta{0.0f};   // Expert attribute, e.g. clamp limit
    };

    /// \brief Constructs a MOE operation with config only
    /// \param args The input tensors
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
