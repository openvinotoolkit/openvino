// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include <snippets/snippets_isa.hpp>
#include <snippets/lowered/expression.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface SerializationNode
 * @brief Fake node needed to serialize lowered::Expression sessionIR
 * @ingroup snippets
 */
class SerializationNode : public ngraph::op::Op {
public:
    OPENVINO_OP("SerializationNode", "SnippetsOpset");

    SerializationNode() = default;
    SerializationNode(const Output<Node> &arg, const std::shared_ptr<lowered::Expression>& expr);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &new_args) const override;
    bool visit_attributes(AttributeVisitor &visitor) override;

private:
    std::shared_ptr<lowered::Expression> m_expr;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
