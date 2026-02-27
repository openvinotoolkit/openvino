// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace custom {

/// \brief PercentFormat operation
/// Formats a scalar float into percentage representation (logical op)
///
/// Inputs:
///   value     : scalar float
///   precision : scalar int
///
/// Output:
///   scalar float (same as input value, formatting handled at frontend/runtime)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PercentFormat : public Op {
public:
    OPENVINO_OP("PercentFormat", "custom_opset", op::Op);

    PercentFormat() = default;

    /// \brief Constructs PercentFormat operation
    PercentFormat(const Output<Node>& value,
                  const Output<Node>& precision);

    /// \brief Validates inputs and infers output types
    void validate_and_infer_types() override;

    /// \brief Required for graph transformations
    std::shared_ptr<Node> clone_with_new_inputs(
        const OutputVector& new_args) const override;

    /// \brief Optional evaluation implementation
    bool evaluate(
        TensorVector& outputs,
        const TensorVector& inputs) const override;
};

}  // namespace custom
}  // namespace op
}  // namespace ov