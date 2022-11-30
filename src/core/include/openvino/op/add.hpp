// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace opset1 {

/// \brief Elementwise addition operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Add : public op::util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("Add", "opset1", op::util::BinaryElementwiseArithmetic, 1);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs an uninitialized addition operation
    Add() : op::util::BinaryElementwiseArithmetic(op::AutoBroadcastType::NUMPY) {}

    /// \brief Constructs an addition operation.
    ///
    /// \param arg0 Output that produces the first input tensor.<br>
    /// `[d0, ...]`
    /// \param arg1 Output that produces the second input tensor.<br>
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
    ///                       implicit broadcasting.
    ///
    /// Output `[d0, ...]`
    ///
    Add(const Output<Node>& arg0,
        const Output<Node>& arg1,
        const op::AutoBroadcastSpec& auto_broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};

}  // namespace opset1
namespace op {
namespace v1 {
using ::ov::opset1::Add;
}  // namespace v1
}  // namespace op
}  // namespace ov

#define OPERATION_DEFINED_Add 1
#include "openvino/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_Add
