// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/op/op.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v13 {

// BinaryElementwiseLogical => BinaryBitwiseLogical
class OPENVINO_API BitwiseAnd : public util::BinaryElementwiseArithmetic {
public:
    OPENVINO_OP("BitwiseAnd", "opset13", util::BinaryElementwiseArithmetic);
    BitwiseAnd() : util::BinaryElementwiseArithmetic(AutoBroadcastType::NUMPY) {}
    BitwiseAnd(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
