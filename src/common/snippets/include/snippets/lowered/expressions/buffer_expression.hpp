// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/expression.hpp"

#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

// To avoid cycle-dependancy of includes, we forward-declare LoopManager
class LoopManager;
/**
 * @interface BufferExpression
 * @brief This is a base class for memory storage.
 *        Note that Buffer should be a single consumer for operation output port
 * @param m_allocation_size - memory size for allocation in bytes. Dynamic value means undefined size.
 * @param m_offset - offset in common Buffer scratchpad
 * @param m_reg_group - number of register group. The Buffers from the same register group will have the same GPR
 * @param m_cluster_id - number of cluster. The Buffers from the same cluster shares memory between them and will have the same offset.
 * @ingroup snippets
 */
class BufferExpression : public Expression {
    friend class ExpressionFactory;
public:
    OPENVINO_RTTI("BufferExpression", "0", Expression)
    BufferExpression() = default;

    bool visit_attributes(AttributeVisitor &visitor) override;

    size_t get_reg_group() const { return m_reg_group; }
    size_t get_cluster_id() const { return m_cluster_id; }
    size_t get_offset() const { return m_offset; }
    size_t get_allocation_size() const { return m_allocation_size; }
    size_t get_byte_size() const;
    ov::element::Type get_data_type() const;

    void set_reg_group(size_t reg_group) { m_reg_group = reg_group; }
    void set_cluster_id(size_t cluster) { m_cluster_id = cluster; }
    void set_allocation_size(size_t size) { m_allocation_size = size; }
    void set_offset(size_t offset) { m_offset = offset; }

    virtual void init_allocation_size(const std::shared_ptr<LoopManager>& loop_manager, size_t allocation_rank);

    // Returns True, if allocation size is known. Otherwise returns False - allocation size is undefined
    bool is_defined() const;

    // Returns True, if the memory is independent - expression doesn't have parents (source)
    bool is_independent_memory() const { return get_input_count() == 0; }

protected:
    BufferExpression(const std::shared_ptr<Node>& n, const std::shared_ptr<IShapeInferSnippetsFactory>& factory);

    ExpressionPtr clone() const override;

    size_t m_allocation_size = utils::get_dynamic_value<size_t>();
    size_t m_reg_group = 0;
    size_t m_cluster_id = 0;
    size_t m_offset = utils::get_dynamic_value<size_t>();
};
using BufferExpressionPtr = std::shared_ptr<BufferExpression>;

} // namespace lowered
} // namespace snippets
} // namespace ov
