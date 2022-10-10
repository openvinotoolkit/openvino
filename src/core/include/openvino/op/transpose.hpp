// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {
/// \brief Tensor transpose operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Transpose : public Op {
public:
    OPENVINO_OP("Transpose", "opset1", op::Op, 1);
    BWDCMP_RTTI_DECLARATION;

    Transpose() = default;
    ///
    /// \brief      Constructs a transpose operation.
    ///
    /// \param      arg          Node producing the tensor to be transposed.
    /// \param      input_order  Node producing the permutation to apply to the axes
    ///                          of the input shape. Must be a vector with shape [n],
    ///                          where n is the rank of arg. The tensor's value must
    ///                          contain every integer in the range [0, n-1].
    ///
    Transpose(const Output<Node>& arg, const Output<Node>& input_order);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool evaluate_upper(const HostTensorVector& output_values) const override;
    bool evaluate_lower(const HostTensorVector& output_values) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    bool has_evaluate() const override;
    bool evaluate_label(TensorLabelVector& output_labels) const override;

    /// \brief Inputs indexes and count.
    enum Ins : size_t { ARG, ORDER, IN_COUNT };
    /// \brief Outputs indexes and count.
    enum Outs : size_t { ARG_T, OUT_COUNT };
};
}  // namespace v1
}  // namespace op
}  // namespace ov
