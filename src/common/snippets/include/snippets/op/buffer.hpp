// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 *        Notes:
 *               - All buffers with the same ID in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @param m_shape - output allocation shape for Buffer with type NewMemory
 * @param m_offset - offset in common Buffer scratchpad
 * @param m_id - Buffer ID in common Buffer system
 * @ingroup snippets
 */
class Buffer : public ov::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const OutputVector& arguments, const ov::Shape& shape, size_t id, ov::element::Type element_type = ov::element::u8);

    bool visit_attributes(AttributeVisitor& visitor) override;

    size_t get_id() const { return m_id; }
    int64_t get_offset() const { return m_offset; }
    void set_id(size_t id) { m_id = id; }
    const ov::Shape& get_allocation_shape() const { return m_shape; }
    void set_allocation_shape(const ov::Shape& allocation_shape) { m_shape = allocation_shape; }
    void set_offset(int64_t offset) { m_offset = offset; }
    size_t get_byte_size() const;

protected:
    ov::Shape m_shape = {};
    size_t m_id = 0;  // Default ID - 0. All Buffers are from the same set
    ov::element::Type m_element_type = ov::element::u8;  // u8 - default 1 byte
    int64_t m_offset = 0;
};

/**
 * @interface IntermediateMemoryBuffer
 * @brief Represents an intermediate memory storage operation. It always has a parent.
 * @ingroup snippets
 *
 */
class IntermediateMemoryBuffer : public Buffer {
public:
    OPENVINO_OP("IntermediateMemoryBuffer", "SnippetsOpset", Buffer);
    IntermediateMemoryBuffer() = default;
    IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, const ov::Shape& shape, size_t id = 0);
    IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, int32_t allocation_rank = -1, size_t id = 0);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    ov::Shape compute_shape_from_allocation_rank(const ov::Output<ov::Node>& arg, int32_t allocation_rank);
};

/**
 * @interface NewMemoryBuffer
 * @brief Represents a new empty memory for allocation with specified shape. It has no parent operations.
 * @ingroup snippets
 *
 */
class NewMemoryBuffer : public Buffer {
public:
    OPENVINO_OP("NewMemoryBuffer", "SnippetsOpset", Buffer);
    NewMemoryBuffer() = default;
    NewMemoryBuffer(const ov::Shape& shape, size_t id = 0, ov::element::Type element_type = ov::element::u8);

    void validate_and_infer_types() override;
    void set_element_type(ov::element::Type element_type);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    class ShapeInfer : public IShapeInferSnippets {
        ov::Shape m_shape;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };
};

} // namespace op
} // namespace snippets
} // namespace ov
