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
class OPENVINO_API Squeeze : public v0::Squeeze {
public:
    OPENVINO_OP("Squeeze", "opset15");

    using v0::Squeeze::Squeeze;

    /// \brief Constructs a squeeze operation.
    ///
    /// \param data Input tensor with data
    /// \param axis The axis along which to squeeze the input tensor.
    /// \param allow_axis_skip Shape inference result dynamic rank if selected axis has 1 in range of its dynamic
    Squeeze(const Output<Node>& data, const Output<Node>& axes, const bool allow_axis_skip = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool get_allow_axis_skip() const;

    std::pair<bool, std::reference_wrapper<const ov::PartialShape>> get_deduced_output_shape() const;
    void set_deduced_output_shape(const ov::PartialShape& output_shapes);

private:
    bool m_allow_axis_skip{};
    ov::PartialShape deduced_output_shape{};
    bool is_deduced_output_shape{};
};
}  // namespace v15
}  // namespace op
}  // namespace ov
