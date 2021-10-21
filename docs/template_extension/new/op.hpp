// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/op.hpp>

//! [op:header]
namespace TemplateExtension {

class Operation : public ov::op::Op {
public:
    OPENVINO_OP("Operation");

    Operation() = default;
    Operation(const ov::Output<ov::Node>& arg, int64_t add);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::runtime::TensorVector& outputs, const ov::runtime::TensorVector& inputs) const override;
    bool has_evaluate() const override;

    int64_t getAddAttr() const {
        return add;
    }
private:
    int64_t add;
};
//! [op:header]

}  // namespace TemplateExtension
