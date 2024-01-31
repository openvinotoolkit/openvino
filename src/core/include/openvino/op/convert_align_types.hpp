// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise type alignment conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvertAlignTypes : public Op {
public:
    OPENVINO_OP("ConvertAlignTypes", "", op::Op);

    /// \brief Constructs a type alignment and conversion operation.
    ConvertAlignTypes() = default;
    /// \brief Constructs a type alignment and conversion operation.
    /// \param lhs  Node with datatype to be aligned.
    /// \param rhs  Node with datatype to be aligned.
    /// \param promote_unsafe  Bool attribute wether to allow for promotions that might result in bit-widening,
    /// precision loss and undefined behaviors.
    /// \param pytorch_scalar_align  Bool attribute wether to align scalars using  PyTorch-like rules.
    /// \param u64_integer_promotion_target  Element type attribute to select alignment target for u64 and signed
    /// integers.
    ConvertAlignTypes(const Output<Node>& lhs,
                      const Output<Node>& rhs,
                      const bool promote_unsafe = false,
                      const bool pytorch_scalar_align = false,
                      const element::Type& u64_integer_promotion_target = element::f32);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_pytorch_scalar_align() const {
        return m_pytorch_scalar_align;
    }

    void set_pytorch_scalar_align(bool pytorch_scalar_align) {
        m_pytorch_scalar_align = pytorch_scalar_align;
    }

    bool get_promote_unsafe() const {
        return m_promote_unsafe;
    }

    void set_promote_unsafe(bool promote_unsafe) {
        m_promote_unsafe = promote_unsafe;
    }

    element::Type get_u64_integer_promotion_target() const {
        return m_u64_integer_promotion_target;
    }

    void set_u64_integer_promotion_target(element::Type u64_integer_promotion_target) {
        m_u64_integer_promotion_target = u64_integer_promotion_target;
    }

private:
    bool m_promote_unsafe = false;
    bool m_pytorch_scalar_align = false;
    element::Type m_u64_integer_promotion_target = element::f32;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
