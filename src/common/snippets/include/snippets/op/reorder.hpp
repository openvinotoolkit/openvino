// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "shape_infer_op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {
/**
 * @interface Reorder
 * @brief Reorder reshapes input tensor shape by reqiured target order.
 *        The tensor data is not updated.
 *        Note: Order is stored in input PortDescriptor
 * @ingroup snippets
 */
class SNIPPETS_API Reorder : public ShapeInferOp {
public:
    OPENVINO_OP("Reorder", "SnippetsOpset", ShapeInferOp);
    Reorder() = default;
    Reorder(const Output<Node>& arg, const std::vector<size_t>& order);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    class SNIPPETS_API ShapeInfer : public IShapeInferSnippets {
        std::vector<size_t> m_target_order;

    public:
        explicit ShapeInfer(const std::shared_ptr<Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& order);
};

}  // namespace ov::snippets::op
