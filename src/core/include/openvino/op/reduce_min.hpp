// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief ReduceMin operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReduceMin : public util::ArithmeticReductionKeepDims {
public:
    OPENVINO_OP("ReduceMin", "opset1", util::ArithmeticReductionKeepDims, 1);
    BWDCMP_RTTI_DECLARATION;
    /// \brief Constructs a summation operation.
    ReduceMin() = default;
    /// \brief Constructs a summation operation.
    ///
    /// \param arg The tensor to be summed.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to 1 it holds axes that are used for reduction.
    ReduceMin(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate_lower(const HostTensorVector& outputs) const override;
    bool evaluate_upper(const HostTensorVector& outputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
};
}  // namespace v1
}  // namespace op
}  // namespace ov
