// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v1 {

class OPENVINO_API SparseConv : public Op {
public:
    OPENVINO_OP("SparseConv", "opset1", op::Op, 1);

    BWDCMP_RTTI_DECLARATION;
    SparseConv() = default;

    SparseConv(const Output<ngraph::Node>& features,
               const Output<ngraph::Node>& inp_pos,
               const Output<ngraph::Node>& out_pos,
               const Output<ngraph::Node>& kernel,
               const Output<ngraph::Node>& offset);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    bool has_evaluate() const override;
};
}  // namespace v1
}  // namespace op
}  // namespace ov
