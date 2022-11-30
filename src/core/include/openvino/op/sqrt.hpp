// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace opset1 {
// clang-format off
/// \brief Elementwise square root operation.
///
/// ## Inputs
///
/// |       | Type                              | Description                                     |
/// | ----- | --------------------------------- | ----------------------------------------------- |
/// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
///
/// ## Output
///
/// | Type                   | Description                                                                           |
/// | ---------------------- | ------------------------------------------------------------------------------------- |
/// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \sqrt{\texttt{arg}[i_1,\dots,i_n]}\f$ |
/// \ingroup ov_ops_cpp_api
// clang-format on
class OPENVINO_API Sqrt : public op::util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("Sqrt", "opset1", op::util::UnaryElementwiseArithmetic);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a square operation.
    ///
    /// \param arg Node that produces the input tensor.
    Sqrt(const Output<Node>& arg);
    Sqrt() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace opset1
namespace op {
namespace v0 {
using ::ov::opset1::Sqrt;
}  // namespace v0
}  // namespace op
}  // namespace ov

#define OPERATION_DEFINED_Sqrt 1
#include "openvino/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_Sqrt
