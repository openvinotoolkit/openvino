// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note HyperConnectionPost op class is under development and subject to change
///
/// \brief mHC post step (DeepSeek-V4 / GLM-5.3): writes the sublayer output back into the n residual streams.
///   out[..., d, :] = y[..., :] * post[..., d] + sum_s comb[..., s, d] * residual[..., s, :]
/// Inputs: y [..., E], residual [..., n, E], post [..., n], comb [..., n, n] ([src][dst]). Output [..., n, E].
/// Reference: the mHC formulation of DeepSeek-V4 / GLM-5.3-Flash.
class OPENVINO_API HyperConnectionPost : public ov::op::Op {
public:
    OPENVINO_OP("HyperConnectionPost");

    HyperConnectionPost() = default;
    HyperConnectionPost(const Output<Node>& y,
                        const Output<Node>& residual,
                        const Output<Node>& post,
                        const Output<Node>& comb);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace ov::op::internal
