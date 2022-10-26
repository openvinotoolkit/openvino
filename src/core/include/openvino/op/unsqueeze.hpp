// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Unsqueeze operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Unsqueeze : public Op {
public:
    OPENVINO_OP("Unsqueeze", "opset1");
    BWDCMP_RTTI_DECLARATION;

    Unsqueeze() = default;
    Unsqueeze(const Output<Node>& data, const Output<Node>& axes);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate_lower(const HostTensorVector& output_values) const override;
    bool evaluate_upper(const HostTensorVector& output_values) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool evaluate_label(TensorLabelVector& output_labels) const override;

    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
