// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class ScaledDotProductAttention : public ::ov::op::Op {
public:
    OPENVINO_OP("ScaledDotProductAttention", "gpu_opset");

    ScaledDotProductAttention() = default;

    /// \brief    Constructs Operation for Scaled Dot-Product Attention
    ///
    /// \param input_q         Matrix Q.
    /// \param input_k         Matrix K.
    /// \param input_v         Matrix V.
    /// \param scale           Scalar scale value.
    /// \param attn_mask       Matrix attention mask.
    ScaledDotProductAttention(const Output<Node>& input_q,
                              const Output<Node>& input_k,
                              const Output<Node>& input_v,
                              const Output<Node>& scale,
                              const Output<Node>& attn_mask);

/// \brief    Constructs Operation for Scaled Dot-Product Attention
    ///
    /// \param input_q         Matrix Q.
    /// \param input_k         Matrix K.
    /// \param input_v         Matrix V.
    /// \param scale           Scalar scale value.
    ScaledDotProductAttention(const Output<Node>& input_q,
                              const Output<Node>& input_k,
                              const Output<Node>& input_v,
                              const Output<Node>& scale);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
