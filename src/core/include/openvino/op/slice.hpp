// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief Slice operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Slice : public Op {
public:
    OPENVINO_OP("Slice", "opset8");

    BWDCMP_RTTI_DECLARATION;

    Slice() = default;

    /// \brief    Constructs Slice operation (default axes).
    ///
    /// \param data             The tensor to be sliced.
    /// \param start            1D tensor with start indices of the slice.
    /// \param stop             1D tensor with end indices of the slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    Slice(const Output<Node>& data, const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step);

    /// \brief    Constructs Slice operation.
    ///
    /// \param data             The tensor to be sliced.
    /// \param start            1D tensor with start indices of the slice.
    /// \param stop             1D tensor with end indices of the slice.
    /// \param step             1D tensor specifies the increment to use in slicing along corresponding axes.
    /// \param axes             1D tensor indicating which dimensions the values in the `start` and `stop` apply to.
    Slice(const Output<Node>& data,
          const Output<Node>& start,
          const Output<Node>& stop,
          const Output<Node>& step,
          const Output<Node>& axes);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool has_evaluate() const override;
    // TODO: Update to use new evaluate with TensorVector
    bool evaluate(const HostTensorVector&, const HostTensorVector&) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate_lower(const HostTensorVector& outputs) const override;
    bool evaluate_upper(const HostTensorVector& outputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool evaluate_label(TensorLabelVector& output_labels) const override;

    std::shared_ptr<ngraph::op::v0::Constant> get_default_const_axes(const Output<Node>& start) const;
    PartialShape calculate_output_shape(const std::vector<int64_t>& starts,
                                        const std::vector<int64_t>& stops,
                                        const std::vector<int64_t>& steps,
                                        const std::vector<int64_t>& axes,
                                        const PartialShape& data_shape) const;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
