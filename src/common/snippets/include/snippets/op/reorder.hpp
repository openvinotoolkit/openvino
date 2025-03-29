// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_infer_op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace op {
/**
 * @interface Reorder
 * @brief Reorder reshapes input tensor shape by reqiured target order.
 *        The tensor data is not updated.
 *        Note: Order is stored in input PortDescriptor
 * @ingroup snippets
 */
class Reorder : public ShapeInferOp {
public:
    OPENVINO_OP("Reorder", "SnippetsOpset", ShapeInferOp);
    Reorder() = default;
    Reorder(const Output<Node>& x, std::vector<size_t> order);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    class ShapeInfer : public IShapeInferSnippets {
        std::vector<size_t> m_target_order {};
    public:
        explicit ShapeInfer(const std::shared_ptr<Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

private:
    void custom_constructor_validate_and_infer_types(std::vector<size_t> order);
};

} // namespace op
} // namespace snippets
} // namespace ov
