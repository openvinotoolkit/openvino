// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/shape_inference/shape_infer_instances.hpp"
#include <snippets/op/memory_access.hpp>
#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface BroadcastLoad
 * @brief Is generated for broadcasting by least varying dimension for non-blocked cases and the second varying dimension for blocked
 * @ingroup snippets
 */
class BroadcastLoad : public modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("BroadcastLoad", "SnippetsOpset");

    BroadcastLoad(const Output<Node>& x, ov::Dimension bcast_dimension, size_t offset = 0lu);
    BroadcastLoad() = default;

    size_t get_offset() const { return get_input_offset(0); }

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    const ov::Dimension& get_bcast_dimension() {return bcast_dimension;}
    void set_bcast_dimension(ov::Dimension new_dim) {bcast_dimension = std::move(new_dim);}

    // Note:BroadcastMove and BroadcastLoad are implemented as separate classes,
    // but have identical shapeInfer semantics. In order to avoid code duplication,
    // we created dummy ShapeInfer classes that are essentially instantiations
    // of a common ShapeInfer class template;
    struct ShapeInfer : public BroadcastShapeInfer<BroadcastLoad> {
        explicit ShapeInfer(const std::shared_ptr<Node>& n) : BroadcastShapeInfer<BroadcastLoad>(n) {}
    };
private:
    ov::Dimension bcast_dimension;
};

} // namespace op
} // namespace snippets
} // namespace ov
