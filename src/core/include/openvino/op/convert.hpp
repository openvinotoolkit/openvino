// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Elementwise type conversion operation.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Convert : public Op {
public:
    OPENVINO_OP("Convert", "opset1");
    BWDCMP_RTTI_DECLARATION;

    /// \brief Constructs a conversion operation.
    Convert() = default;
    /// \brief Constructs a conversion operation.
    ///
    /// \param arg          Node that produces the input tensor.
    /// \param destination_type  Element type for the output tensor.
    Convert(const Output<Node>& arg, const ov::element::Type& destination_type);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    const element::Type& get_destination_type() const {
        return m_destination_type;
    }
    void set_destination_type(const element::Type& destination_type) {
        m_destination_type = destination_type;
    }
    const element::Type& get_convert_element_type() const {
        return m_destination_type;
    }
    void set_convert_element_type(const element::Type& destination_type) {
        m_destination_type = destination_type;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate_lower(const HostTensorVector& outputs) const override;
    bool evaluate_upper(const HostTensorVector& outputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool evaluate_label(TensorLabelVector& output_labels) const override;

protected:
    ov::element::Type m_destination_type;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
