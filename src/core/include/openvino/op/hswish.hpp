// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v4 {
/// \brief A HSwish Activation Function
/// f(x) =  x * min(max(x + 3, 0), 6) / 6 or
/// f(x) = x * min(ReLU(x + 3), 6) / 6
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API HSwish : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("HSwish", "opset4", op::util::UnaryElementwiseArithmetic, 4);
    BWDCMP_RTTI_DECLARATION;
    HSwish() = default;

    /// \brief Constructs a HSwish (hard version of Swish) operation.
    ///
    /// \param data Input tensor
    HSwish(const Output<Node>& arg);

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v4
}  // namespace op
}  // namespace ov
