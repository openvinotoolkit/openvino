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
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "shape_infer_op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov::snippets::op {

/**
 * @interface Reshape
 * @brief Reshape input tensor to reqiured target shape
 * @ingroup snippets
 */
class Reshape : public ShapeInferOp {
public:
    OPENVINO_OP("Reshape", "SnippetsOpset", ShapeInferOp);
    Reshape(const Output<Node>& arg, ov::PartialShape target_shape);
    Reshape() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    const ov::PartialShape& get_target_shape() const;
    void set_target_shape(ov::PartialShape shape);

    class ShapeInfer : public IShapeInferSnippets {
        VectorDims target_shape;
        size_t target_shape_volume = 0;

    public:
        explicit ShapeInfer(const std::shared_ptr<Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

private:
    ov::PartialShape m_target_shape;
};

}  // namespace ov::snippets::op
