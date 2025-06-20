// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <snippets/lowered/expression.hpp>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface SerializationNode
 * @brief Fake node needed to serialize lowered::Expression sessionIR
 * @ingroup snippets
 */
class SerializationNode : public ov::op::Op {
public:
    OPENVINO_OP("SerializationNode", "SnippetsOpset");

    enum SerializationMode { DATA_FLOW, CONTROL_FLOW };
    SerializationNode() = default;
    SerializationNode(const ov::OutputVector& args,
                      const std::shared_ptr<lowered::Expression>& expr,
                      SerializationMode mode = SerializationMode::CONTROL_FLOW);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    std::shared_ptr<lowered::Expression> m_expr;
    SerializationMode m_mode;
};

}  // namespace ov::snippets::op
