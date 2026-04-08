// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v17 {
/// \brief Elementwise four-quadrant arctangent operation.
///
/// Computes atan2(y, x) element-wise: the angle (in radians) between the
/// positive x-axis and the point (x, y), in the range [-π, π].
/// Supports numpy-style auto-broadcasting. Accepts floating-point inputs only.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Atan2 : public util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("Atan2", "opset17", util::BinaryElementwiseArithmetic);

    /// \brief Constructs an uninitialized Atan2 operation.
    Atan2() : util::BinaryElementwiseArithmetic(AutoBroadcastType::NUMPY) {}

    /// \brief Constructs an Atan2 operation.
    ///
    /// \param y             Output that produces the y (ordinate) input tensor.
    /// \param x             Output that produces the x (abscissa) input tensor.
    /// \param auto_broadcast Auto broadcast specification.
    Atan2(const Output<Node>& y,
          const Output<Node>& x,
          const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
}  // namespace v17
}  // namespace op
}  // namespace ov
