// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a base class for memory storage.
 *        Notes:
 *               - All buffers with the same reg_group in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @param m_allocation_size - memory size for allocation in u8 data type. Dynamic value means undefined size.
 * @param m_offset - offset in common Buffer scratchpad
 * @param m_reg_group - number of register group. The Buffers from the same register group will have the same GPR
 * @param m_cluster_id - number of cluster. The Buffers from the same cluster shares memory between them and will have the same offset.
 * @ingroup snippets
 */
class Buffer : public ov::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const OutputVector& arguments, size_t allocation_size = utils::get_dynamic_value<size_t>(), size_t reg_group = 0, size_t cluster_id = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;

    size_t get_reg_group() const { return m_reg_group; }
    size_t get_cluster_id() const { return m_cluster_id; }
    size_t get_offset() const { return m_offset; }
    size_t get_allocation_size() const { return m_allocation_size; }
    size_t get_byte_size() const;

    void set_reg_group(size_t reg_group) { m_reg_group = reg_group; }
    void set_cluster_id(size_t cluster) { m_cluster_id = cluster; }
    void set_allocation_size(size_t allocation_size) { m_allocation_size = allocation_size; }
    void set_offset(size_t offset) { m_offset = offset; }

    // Returns True, if allocation size is known. Otherwise returns False - allocation size is undefined
    bool is_defined() const;

protected:
    size_t m_allocation_size = utils::get_dynamic_value<size_t>();
    size_t m_reg_group = 0;
    size_t m_cluster_id = 0;
    size_t m_offset = utils::get_dynamic_value<size_t>();
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
    IntermediateMemoryBuffer(const ov::Output<ov::Node>& arg, size_t allocation_size = utils::get_dynamic_value<size_t>(),
                             size_t reg_group = 0, size_t cluster_id = 0);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
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
    NewMemoryBuffer(const ov::Shape& shape, size_t reg_group = 0, size_t cluster_id = 0, ov::element::Type element_type = ov::element::u8);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void set_element_type(ov::element::Type element_type);

    class ShapeInfer : public IShapeInferSnippets {
        ov::Shape m_shape;
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

private:
    ov::Shape m_output_shape;
    ov::element::Type m_element_type = ov::element::u8;  // u8 - default 1 byte
};

} // namespace op
} // namespace snippets
} // namespace ov
