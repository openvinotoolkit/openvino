// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov::op::internal {

// Explicit rounding uses F32 arithmetic/output, rounding Q before QK and softmax probabilities before PV.
class OPENVINO_API ScaledDotProductAttentionExtension : public v13::ScaledDotProductAttention {
public:
    OPENVINO_OP("ScaledDotProductAttentionExtension", "internal", v13::ScaledDotProductAttention);

    ScaledDotProductAttentionExtension() = default;
    ScaledDotProductAttentionExtension(const OutputVector& inputs,
                                       bool causal,
                                       element::Type query_precision = element::dynamic,
                                       element::Type probability_precision = element::dynamic);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    bool has_evaluate() const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

    element::Type get_query_precision() const {
        return m_query_precision;
    }
    element::Type get_probability_precision() const {
        return m_probability_precision;
    }

private:
    element::Type m_query_precision = element::dynamic;
    element::Type m_probability_precision = element::dynamic;
};

}  // namespace ov::op::internal
