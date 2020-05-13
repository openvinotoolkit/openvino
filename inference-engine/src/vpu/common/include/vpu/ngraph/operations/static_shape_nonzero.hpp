// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeNonZero : public ngraph::op::Op {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeNonZero", 0};

    const NodeTypeInfo& get_type_info() const override { return type_info; }

    explicit StaticShapeNonZero(const Output<ngraph::Node>& input, const element::Type& output_type = element::i64);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool evaluate(const HostTensorVector& output_values,
                  const HostTensorVector& input_values) override;

    element::Type get_output_type() const { return m_output_type; }
    void set_output_type(element::Type output_type) { m_output_type = output_type; }
    // Overload collision with method on Node
    using Node::set_output_type;

protected:
    element::Type m_output_type;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
