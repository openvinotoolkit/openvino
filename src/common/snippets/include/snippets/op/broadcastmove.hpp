// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface BroadcastMove
 * @brief Added to a subgraph if explicit broadcast instruction should be generated
 * @ingroup snippets
 */
class BroadcastMove : public ngraph::op::Op {
public:
    OPENVINO_OP("BroadcastMove", "SnippetsOpset");

    BroadcastMove(const Output<Node>& x, ov::PartialShape output_shape);
    BroadcastMove() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;


protected:
    ov::PartialShape output_shape;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
