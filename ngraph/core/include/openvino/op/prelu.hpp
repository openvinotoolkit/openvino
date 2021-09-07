// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Parametrized Relu
/// x <  0 => f(x) = x * slope
/// x >= 0 => f(x) = x
///
class OPENVINO_API PRelu : public Op {
public:
    OPENVINO_RTTI_DECLARATION;
    PRelu();
    /// \brief Constructs a PRelu operation.
    ///
    /// \param data Input tensor
    /// \param slope Multipliers for negative values
    PRelu(const Output<Node>& data, const Output<Node>& slope);

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
