// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 *        Notes:
 *               - All buffers in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @ingroup snippets
 */
class Buffer : public ngraph::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");

    size_t get_byte_size() const;
    virtual ov::PartialShape get_allocation_shape() const = 0;

protected:
    Buffer() = default;
};

/**
 * @interface AllocationBuffer
 * @brief The operation is for allocation of new empty memory. The operation has one parent that is equal to allocation shape
 *        - m_element_type - element type of memory
 * @ingroup snippets
 */
class AllocationBuffer : public Buffer {
public:
    OPENVINO_OP("AllocationBuffer", "SnippetsOpset", Buffer);

    AllocationBuffer() = default;
    AllocationBuffer(const ov::Output<ov::Node>& shape, const ov::element::Type element_type);

    ov::PartialShape get_allocation_shape() const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

protected:
    ov::element::Type m_element_type;
};

/**
 * @interface IntermediateBuffer
 * @brief The operation is for intermediate data storage.
 *        If Buffer has only one parent, the Buffer will allocate a full memory with input shape of Buffer.
 *        If Buffer has second parent as well, the Buffer will allocate memory with shape that is equal to values from second input but
 *        saves the input shape for shape inference and input element type.
 *        For example,
 *              Parameter [5, 3, 128]    Constant [2] (with values {3, 128})
 *                     \                 /
 *                  Buffer with allocated memory 3x128 size
 *                              |
 *                       Result [5, 3, 128]
 * @ingroup snippets
 */
class IntermediateBuffer : public Buffer {
public:
    OPENVINO_OP("IntermediateBuffer", "SnippetsOpset", Buffer);

    IntermediateBuffer() = default;
    IntermediateBuffer(const ov::Output<ov::Node>& x);
    IntermediateBuffer(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& shape);

    ov::PartialShape get_allocation_shape() const override;

    bool visit_attributes(AttributeVisitor& visitor) override { return true; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    static std::shared_ptr<ov::Node> create_shape_constant(const ov::PartialShape& shape, size_t allocation_rank);
    static std::shared_ptr<ov::Node> create_shape_constant(const ov::PartialShape& shape);
};

} // namespace op
} // namespace snippets
} // namespace ngraph
