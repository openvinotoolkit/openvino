// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface BroadcastMove
 * @brief Added to a subgraph if explicit broadcast instruction should be generated
 * @ingroup snippets
 */
class BroadcastMove : public ov::op::Op {
public:
    OPENVINO_OP("BroadcastMove", "SnippetsOpset");

    BroadcastMove(const Output<Node>& x, ov::Dimension bcast_dimension);
    BroadcastMove() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;
    const ov::Dimension& get_bcast_dimension() {return bcast_dimension;}
    void set_bcast_dimension(ov::Dimension new_dim) {bcast_dimension = std::move(new_dim);}
    // Note:BroadcastMove and BroadcastLoad are implemented as separate classes,
    // but have identical shapeInfer semantics. In order to avoid code duplication,
    // we created dummy ShapeInfer classes that are essentially instantiations
    // of a common ShapeInfer class template;
    struct ShapeInfer : public BroadcastShapeInfer<BroadcastMove> {
        explicit ShapeInfer(const std::shared_ptr<Node>& n) : BroadcastShapeInfer<BroadcastMove>(n) {}
    };

protected:
    ov::Dimension bcast_dimension;
};

} // namespace op
} // namespace snippets
} // namespace ov
