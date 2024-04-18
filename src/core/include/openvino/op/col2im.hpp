// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Operator combining sliding blocks into an image tensor
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Col2Im : public ov::op::Op {
public:
    OPENVINO_OP("Col2Im", "opset15", ov::op::Op);

    Col2Im() = default;
    /// \brief Constructs an RMSNorm operation without scaling.
    ///
    /// \param data Input tensor with data
    /// \param axes Axes for reduce mean calculation
    /// \param eps Epsilon for not dividing by zero while normalizing the value
    /// \param compute_type Precision for the internal computation, if undefined it's the same as the input type
    Col2Im(const Output<Node>& data,
            const ov::Shape& output_size,
            const ov::Shape& kernel_size,
            const Strides& strides = Strides{1, 1},
            const Strides& dilations = Strides{1, 1},
            const Shape& pads_begin = Shape{0, 0},
            const Shape& pads_end = Shape{0, 0});

    /// \brief Constructs an RMSNorm operation with scaling.
    ///
    /// \param data Input tensor with data
    /// \param axes Axes for reduce mean calculation
    /// \param scale Scale values for weight
    /// \param eps Epsilon for not dividing by zero while normalizing the value
    /// \param compute_type Precision for the internal computation, if undefined it's the same as the input type
    Col2Im(const Output<Node>& data,
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

}  // namespace v15
}  // namespace op
}  // namespace ov
