// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace gna {
/// \brief A Self Regularized Non-Monotonic Neural Activation Function
/// f(x) =  x/(1.0 + |x|)
///
class OPENVINO_API SoftSign : public Op {
public:
    OPENVINO_OP("SoftSign", "opset8", op::Op, 8);
    BWDCMP_RTTI_DECLARATION;

    SoftSign() = default;
    /// \brief Constructs an SoftSign operation.
    ///
    /// \param data Input tensor
    SoftSign(const Output<Node>& arg);
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace gna
}  // namespace op
}  // namespace ov
