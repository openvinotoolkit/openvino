// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 * @ingroup snippets
 */
class Buffer : public ov::op::Op {
    enum class Type {
        NewMemory,
        IntermediateMemory
    };

public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const ov::Output<ov::Node>& arg);
    Buffer(const OutputVector& arguments);
    Buffer(const ov::Shape& shape, ov::element::Type element_type = ov::element::u8);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_allocation_size() const;

    class ShapeInfer : public IShapeInferSnippets {
        ov::Shape m_shape;
        Type m_type;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

protected:
    const Type m_type = Type::NewMemory;
    const ov::Shape m_output_shape {};
    const ov::element::Type m_element_type = ov::element::u8;  // u8 - default 1 byte
};


} // namespace op
} // namespace snippets
} // namespace ov
