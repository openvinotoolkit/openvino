// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Result operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Result : public Op {
public:
    OPENVINO_OP("Result", "opset1");

    /// \brief Allows a value to be used as a function result.
    Result() = default;
    /// \brief Allows a value to be used as a function result.
    ///
    /// \param arg Node that produces the input tensor.
    Result(const Output<Node>& arg);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

    /// \brief Returns current layout, or empty Layout if it is not set
    Layout get_layout() const;

    /// \brief Sets layout runtime information to tensor.
    ///
    /// \param layout Layout to set. If empty (default constructed), layout runtime information is erased.
    void set_layout(const Layout& layout);
};
}  // namespace v0
}  // namespace op
using ResultVector = std::vector<std::shared_ptr<op::v0::Result>>;

template <>
class OPENVINO_API AttributeAdapter<ResultVector> : public VisitorAdapter {
public:
    AttributeAdapter(ResultVector& ref);

    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<ResultVector>");

protected:
    ResultVector& m_ref;
};

}  // namespace ov
