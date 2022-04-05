// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Product reduction operation.
///
/// Reduces the tensor, eliminating the specified reduction axes by taking the product.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ReduceProd : public util::ArithmeticReductionKeepDims {
public:
    OPENVINO_OP("ReduceProd", "opset1", util::ArithmeticReductionKeepDims, 1);
    BWDCMP_RTTI_DECLARATION;
    /// \brief Constructs a product reduction operation.
    ReduceProd() = default;
    /// \brief Constructs a product reduction operation.
    ///
    /// \param arg The tensor to be reduced.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    /// \param keep_dims If set to true it holds axes that are used for reduction.
    ReduceProd(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims = false);

    /// \return The default value for Product.
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<Node> get_default_value() const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

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
