// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

//! [op:header]
namespace TemplateExtension {

class Operation : public ngraph::op::Op {
public:
    OPENVINO_OP("Template", "custom_opset");

    Operation() = default;
    Operation(const ngraph::Output<ngraph::Node>& arg, int64_t add);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    int64_t getAddAttr() const {
        return add;
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

private:
    int64_t add;
};
//! [op:header]

}  // namespace TemplateExtension
