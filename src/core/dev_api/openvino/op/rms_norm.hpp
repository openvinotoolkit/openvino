// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
/// \note RMSNorm op class is under development and subject to change
///
/// \brief Operator performing Root Mean Square Normalization
/// \ingroup ov_ops_cpp_api
class OPENVINO_API RMSNorm : public ov::op::Op {
public:
    OPENVINO_OP("RMSNorm");

    RMSNorm() = default;
    /// \brief Constructs an RMSNorm operation without scaling.
    ///
    /// \param data Input tensor with data
    /// \param axes Axes for reduce mean calculation
    /// \param eps Epsilon for not dividing by zero while normalizing the value
    /// \param compute_type Precision for the internal computation, if undefined then type of input data will be used
    RMSNorm(const Output<Node>& data,
            const Output<Node>& axes,
            double epsilson,
            const ov::element::Type& compute_type = ov::element::undefined);

    /// \brief Constructs an RMSNorm operation with scaling.
    ///
    /// \param data Input tensor with data
    /// \param axes Axes for reduce mean calculation
    /// \param scale Scale values for weight
    /// \param eps Epsilon for not dividing by zero while normalizing the value
    /// \param compute_type Precision for the internal computation, if undefined then type of input data will be used
    RMSNorm(const Output<Node>& data,
            const Output<Node>& axes,
            const Output<Node>& scale,
            double epsilson,
            const ov::element::Type& compute_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    double get_epsilon() const;
    const ov::element::Type& get_compute_type() const;

private:
    double m_epsilon{0};
    ov::element::Type m_compute_type{ov::element::undefined};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
