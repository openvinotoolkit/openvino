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

    /// \brief Generates default axes order at end of input vector.
    ///
    /// Default axes order is decreasing sequence numbers which start from `length - 1`.
    ///
    /// \param axes_order  Vector where default order will be generated.
    /// \param length      Sequence length of axes order.
    ///
    static void generate_default_order(std::vector<int64_t>& axes_order, const size_t length);

    /// \brief Check if vector of axes order has got valid values.
    ///
    /// Axes order has to be unique numbers in range of [0, size)
    ///
    /// \param axes_order  Vector with axes order to check.
    /// \param size        Input for transpose rank size.
    ///
    /// \return true if axes order is valid otherwise false.
    ///
    static bool is_valid_order(const std::vector<int64_t>& axes_order, const size_t size);
};
}  // namespace v1
}  // namespace op
}  // namespace ov
