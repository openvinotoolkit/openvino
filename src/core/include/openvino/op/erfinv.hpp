// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v17 {
/// \brief Elementwise inverse error function (erfinv) operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ErfInv : public util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("ErfInv", "opset17", util::UnaryElementwiseArithmetic);

    ErfInv() = default;

    /// \brief Constructs an ErfInv operation.
    ///
    /// \param arg Node that produces the input tensor of floating-point type.
    ErfInv(const Output<Node>& arg);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v17
}  // namespace op
}  // namespace ov
