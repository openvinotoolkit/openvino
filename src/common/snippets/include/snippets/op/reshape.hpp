// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Reshape
 * @brief Reshape input tensor to reqiured target shape
 * @ingroup snippets
 */
class Reshape : public ov::op::Op {
public:
    OPENVINO_OP("Reshape", "SnippetsOpset");
    Reshape(const Output<Node>& x, ov::PartialShape target_shape);
    Reshape() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    const ov::PartialShape& get_target_shape() const;
    void set_target_shape(ov::PartialShape shape);

private:
    ov::PartialShape m_target_shape = {};
};

} // namespace op
} // namespace snippets
} // namespace ov
