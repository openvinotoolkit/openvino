// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief PWL activation function
class OPENVINO_API Pwl : public Op {
public:
    OPENVINO_OP("Pwl", "opset1");
    BWDCMP_RTTI_DECLARATION;

    Pwl() = default;
    /// \brief Constructs a Pwl node.
    ///
    /// \param data  - The input data tensor.
    /// \param m     - The list of the slopes of segment.
    /// \param b     - The list of the y-intercepts of segment.
    /// \param knots - The list of x-coordinates of segment endpoints (segments number + 1).
    Pwl(const Output<Node>& data, const Output<Node>& m, const Output<Node>& b, const Output<Node>& knots);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
