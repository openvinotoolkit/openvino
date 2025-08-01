// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface VectorBuffer
 * @brief The operation is for intermediate data storage in vector register
 * @ingroup snippets
 */
class VectorBuffer : public ov::op::Op {
public:
    OPENVINO_OP("VectorBuffer", "SnippetsOpset");

    explicit VectorBuffer(ov::element::Type element_type = ov::element::f32);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::element::Type m_element_type;
};

}  // namespace ov::snippets::op
