// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Elementwise subtraction operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Subtract : public util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("Subtract", "opset1", util::BinaryElementwiseArithmetic, 1);
    BWDCMP_RTTI_DECLARATION;

    Subtract() : util::BinaryElementwiseArithmetic(AutoBroadcastType::NUMPY) {}

    /// \brief Constructs a subtraction operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification
    Subtract(const Output<Node>& arg0,
             const Output<Node>& arg1,
             const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
