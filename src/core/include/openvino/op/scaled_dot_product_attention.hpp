// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v12 {
/// \brief Scaled dot product attention from PyTorch
///
/// \ingroup ov_ops_cpp_api

class OPENVINO_API ScaledDotProductAttention : public Op {
public:
    OPENVINO_OP("ScaledDotProductAttention", "opset12", op::Op);

    /// \brief Constructs a round operation.
    ScaledDotProductAttention() = default;

    ScaledDotProductAttention(const Output<Node>& query,
            const Output<Node>& key,
            const Output<Node>& value,
            bool is_causal,
            const Output<Node>& attn_mask = Output<Node>());

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    // Replace itself by decomposition
    OutputVector decompose();

private:

    bool m_is_causal;
};

}  // namespace v0
}  // namespace op
}  // namespace ov
