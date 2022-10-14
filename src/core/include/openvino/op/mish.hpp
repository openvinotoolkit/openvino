// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v4 {
/// \brief A Self Regularized Non-Monotonic Neural Activation Function
/// f(x) =  x * tanh(log(exp(x) + 1.))
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Mish : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Mish", "opset4", util::UnaryElementwiseArithmetic, 4);
    BWDCMP_RTTI_DECLARATION;

    Mish() = default;
    /// \brief Constructs an Mish operation.
    ///
    /// \param data Input tensor
    Mish(const Output<Node>& arg);
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v4
}  // namespace op
}  // namespace ov
