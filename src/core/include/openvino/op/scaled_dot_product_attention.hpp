// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Scaled dot product attention operation from PyTorch
///
/// \ingroup ov_ops_cpp_api

class OPENVINO_API ScaledDotProductAttention : public Op {
public:
    OPENVINO_OP("ScaledDotProductAttention", "opset13", op::Op);

    /// \brief Constructs a ScaledDotProductAttention operation.
    ScaledDotProductAttention() = default;

    ScaledDotProductAttention(const OutputVector& inputs, bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              const Output<Node>& attn_mask,
                              const Output<Node>& scale,
                              const Output<Node>& sink,
                              bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              const Output<Node>& attn_mask,
                              const Output<Node>& scale,
                              bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              const Output<Node>& attn_mask,
                              bool causal);

    ScaledDotProductAttention(const Output<Node>& query,
                              const Output<Node>& key,
                              const Output<Node>& value,
                              bool causal);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    /// \brief Tells whether an element type is accepted on the key and value operands as
    ///        quantization codes rather than values.
    ///
    /// The mapping from codes to values is not carried by this operation, so a consumer that
    /// cannot define one is expected to reject such an operand rather than read the codes as
    /// magnitudes. Deliberately not element::Type::is_quantized(): that predicate is true for
    /// i32, which this operation still rejects, and false for u4, which it accepts.
    static bool is_quantized_kv_type(const element::Type& type);

    /// \brief Tells whether the key or value operand of \p node holds quantization codes.
    static bool has_quantized_kv(const ScaledDotProductAttention& node);

    bool get_causal() const {
        return m_causal;
    }

    void set_causal(bool causal) {
        m_causal = causal;
    }

private:
    bool m_causal = false;
};

}  // namespace v13
}  // namespace op
}  // namespace ov
