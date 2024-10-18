// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <utility>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Squeeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Squeeze : public Op {
public:
    OPENVINO_OP("Squeeze", "opset1");

    Squeeze();
    Squeeze(const Output<Node>& data, const Output<Node>& axes);
    Squeeze(const Output<Node>& data);

    void validate_and_infer_types() override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool is_dynamic() const override;

private:
    Output<Node> get_default_axes_input() const;
};
}  // namespace v0

namespace v15 {
class OPENVINO_API Squeeze : public Op {
public:
    OPENVINO_OP("Squeeze", "opset15");

    Squeeze();
    Squeeze(const Output<Node>& data);

    /// \brief Constructs a squeeze operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The axis along which to squeeze the input tensor.
    /// \param allow_axis_skip Shape inference result dynamic rank if selected axis has 1 in range of its dynamic
    Squeeze(const Output<Node>& data, const Output<Node>& axes, const bool allow_axis_skip = false);

    void validate_and_infer_types() override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;
    bool evaluate_symbol(TensorSymbolVector& output_symbols) const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool is_dynamic() const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool get_allow_axis_skip() const;

private:
    Output<Node> get_default_axes_input() const;
    bool m_allow_axis_skip{};
};
}  // namespace v15
}  // namespace op
}  // namespace ov
