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
 * @brief The operation is for intermediate data storage
 *        - m_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank].
 *                 It's needed to allocate needed memory size that depends on Tile rank, for example.
 *                 Default value is -1 (full shape)
 *        Notes:
 *               - All buffers in a graph have the same memory pointer. So if we have a few buffers,
 *                 each the corresponding MemoryAccess op for Buffer should have offset for common memory pointer of this Buffer
 *               - Buffer should be a single consumer for operation output port
 * @ingroup snippets
 */
class Buffer : public ngraph::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    BWDCMP_RTTI_DECLARATION;

    Buffer(const Output<Node>& x, const int32_t allocation_rank = -1);
    Buffer() = default;

    int32_t get_allocation_rank() const { return m_allocation_rank; }
    void set_allocation_rank(int32_t rank) { m_allocation_rank = rank; }

    size_t get_byte_size() const;

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    int32_t m_allocation_rank = -1;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
