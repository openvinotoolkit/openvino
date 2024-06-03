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
///
/// The Result operator output is special output which share tensor with node connected to this node.
/// The Result's output names are visible as model outputs names.
/// To set these use
/// - `Result::output(0)::set_names/add_names` to set/add this names on Result's output.
///
/// Using `Result::get_output_tensor(0)::set_names/add_names` will set/add names on tensor without modify
/// Result's output names.
/// The Result's output names are appended to connected tensor or transferred to new tensor when Result is connected
/// with new node.
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
