// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace opset1 {
/// \brief Elementwise maximum operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Maximum : public op::util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("Maximum", "opset1", op::util::BinaryElementwiseArithmetic, 1);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a maximum operation.
    Maximum() : op::util::BinaryElementwiseArithmetic(op::AutoBroadcastType::NUMPY) {}

    /// \brief Constructs a maximum operation.
    ///
    /// \param arg0 Node that produces the first input tensor.
    /// \param arg1 Node that produces the second input tensor.
    /// \param auto_broadcast Auto broadcast specification
    Maximum(const Output<Node>& arg0,
            const Output<Node>& arg1,
            const op::AutoBroadcastSpec& auto_broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace opset1
namespace op {
namespace v1 {
using ::ov::opset1::Maximum;
}  // namespace v1
}  // namespace op
}  // namespace ov

#define OPERATION_DEFINED_Maximum 1
#include "openvino/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_Maximum
