// Copyright (C) 2018-2024 Intel CBitwiseRightShiftpBitwiseRightShiftation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Elementwise bitwise BitwiseRightShift operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BitwiseRightShift : public util::BinaryElementwiseBitwise {
public:
    OPENVINO_OP("BitwiseRightShift", "opset15", util::BinaryElementwiseBitwise);
    /// \brief Constructs a bitwise BitwiseRightShift operation.
    BitwiseRightShift() = default;
    /// \brief Constructs a bitwise BitwiseRightShift operation.
    ///
    BitwiseRightShift(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v15
}  // namespace op
}  // namespace ov
