// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph { namespace vpu { namespace op {

class OutShapeOfReshape : public ngraph::op::Op {
public:
    OPENVINO_OP("OutShapeOfReshape", "VPUOpset");

    OutShapeOfReshape(
            const Output<Node>& inDataShape,
            const Output<Node>& outShapeDescriptor,
            bool specialZero);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool getSpecialZero() const { return m_specialZero; }
    void setSpecialZero(bool special_zero) { m_specialZero = special_zero; }

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    void set_output_type(const ngraph::element::Type& output_type);
    using Node::set_output_type;

private:
    bool m_specialZero;
    element::Type m_output_type = element::i64;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
