// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ngraph { namespace vpu { namespace op {

class OutShapeOfReshape : public ngraph::op::Op {
public:
    static constexpr NodeTypeInfo type_info{"OutShapeOfReshape", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    OutShapeOfReshape(
            const Output<Node>& inDataShape,
            const Output<Node>& outShapeDescriptor,
            bool specialZero);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool getSpecialZero() const { return m_specialZero; }
    void setSpecialZero(bool special_zero) { m_specialZero = special_zero; }

private:
    bool m_specialZero;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
