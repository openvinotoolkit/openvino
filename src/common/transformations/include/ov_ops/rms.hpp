// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
/// \brief Operator performing Root Mean Square Normalization
///
/// \note Performs re-scaling invariance and regularizes the summed input according to RMS statistics
class TRANSFORMATIONS_API RMS : public ov::op::Op {
public:
    OPENVINO_OP("RMS", "ie_internal_opset");

    RMS() = default;
    /// \brief Constructs an RMS operation.
    ///
    /// \param data Input tensor with data
    /// \param gamma Gamma values for weight
    /// \param eps Epsilon for not dividing by zero while normalizing the value
    /// \param output_type Output element type
    RMS(const Output<Node>& data,
        const Output<Node>& gamma,
        double epsilon,
        const ov::element::Type output_type = ov::element::dynamic);

    /// @brief Constructs an RMS operation without gamma.
    ///
    /// @param data Input tensor with data
    /// @param eps Epsilon for not dividing by zero while normalizing the value
    /// @param output_type Output element type
    RMS(const Output<Node>& data, double epsilon, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    double get_epsilon() const {
        return m_epsilon;
    }

    void set_epsilon(double epsilon) {
        m_epsilon = epsilon;
    }

    void set_output_type_attr(const element::Type& output_type) {
        m_output_type = output_type;
    }

    bool get_elementwise_affine() const {
        return m_elementwise_affine;
    }

    void set_elementwise_affine(bool elementwise_affine) {
        m_elementwise_affine = elementwise_affine;
    }

private:
    double m_epsilon{0};
    ov::element::Type m_output_type;
    bool m_elementwise_affine{true};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
