// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {
/// \brief Elementwise type alignment conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ConvertPromoteTypes : public Op {
public:
    OPENVINO_OP("ConvertPromoteTypes", "opset14", op::Op);

    /// \brief Constructs a type alignment and conversion operation.
    ConvertPromoteTypes() = default;
    /// \brief Constructs a type alignment and conversion operation.
    /// \param input_0  Node with datatype to be aligned.
    /// \param input_1  Node with datatype to be aligned.
    /// \param promote_unsafe  Bool attribute whether to allow promotions that might result in bit-widening,
    /// precision loss and undefined behaviors.
    /// \param pytorch_scalar_align  Bool attribute whether to align scalars using PyTorch-like rules.
    /// \param u64_integer_promotion_target  Element type attribute to select alignment target for u64 and signed
    /// integers.
    ConvertPromoteTypes(const Output<Node>& input_0,
                      const Output<Node>& input_1,
                      const bool promote_unsafe = false,
                      const bool pytorch_scalar_align = false,
                      const element::Type& u64_integer_promotion_target = element::f32);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_pytorch_scalar_align() const;

    void set_pytorch_scalar_align(bool pytorch_scalar_align);

    bool get_promote_unsafe() const;

    void set_promote_unsafe(bool promote_unsafe);

    const element::Type& get_u64_integer_promotion_target() const;

    void set_u64_integer_promotion_target(const element::Type& u64_integer_promotion_target);

private:
    bool m_promote_unsafe = false;
    bool m_pytorch_scalar_align = false;
    element::Type m_u64_integer_promotion_target = element::f32;
};
}  // namespace v14
}  // namespace op
}  // namespace ov
