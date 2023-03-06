// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface VectorBuffer
 * @brief The operation is for intermediate data storage in vector register
 * @ingroup snippets
 */
class VectorBuffer : public ngraph::op::Op {
public:
    OPENVINO_OP("VectorBuffer", "SnippetsOpset");

    VectorBuffer(const ov::element::Type element_type = ov::element::f32);

    bool visit_attributes(AttributeVisitor& visitor) override { return true;}
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::element::Type m_element_type;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
