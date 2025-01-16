// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_info.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SetBufferRegGroup
 * @brief The pass groups Buffers by Register groups.
 *        The buffers with the same RegGroup will be assigned the same data register.
 *        The pass uses greedy graph coloring algorithm using adjacency matrix:
 *          - Buffers - are vertices of graph;
 *          - Loops, Brgemm (the same other ops) - are "edges" between Buffers (hub of edges).
 *                   The buffers are connected to the same Loop - are adjacent in graph sense bounds.
 *          - The vertices (buffers) are adjacent if they are connected to the same Loop and
 *            their data pointers cannot be proportionally incremented in Loops: different ptr increments or data sizes -
 *            or one of the Buffers is in some a Loop but another Buffer is not;
 *          - Firstly, create adjacency matrix using the definition above;
 *          - Secondly, assign the same color to non-adjacent vertices of graph (buffers), and use different colors otherwise.
 * @ingroup snippets
 */
class SetBufferRegGroup: public RangedPass {
public:
    OPENVINO_RTTI("SetBufferRegGroup", "", RangedPass)
    SetBufferRegGroup() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

    /**
     * @brief Check if two Buffers can be in one register group by LoopDesc < data_size, ptr_increment, finalization_offset >
     * @param lhs LoopPortInfo (Port and Data pointer shift params for first Buffer)
     * @param rhs LoopPortInfo (Port and Data pointer shift params for second Buffer)
     * @return Returns True if params are valid to reuse one register. Otherwise returns False
     */
    static bool can_be_in_one_reg_group(const UnifiedLoopInfo::LoopPortInfo& lhs, const UnifiedLoopInfo::LoopPortInfo& rhs);

private:
    using BufferPool = std::vector<BufferExpressionPtr>;
    using BufferMap = std::map<BufferExpressionPtr, UnifiedLoopInfo::LoopPortInfo>;

    /**
     * @brief Get Buffer Index in Buffer set
     * @param target the target Buffer expression
     * @param pool set of Buffers from the Linear IR
     * @return index of target Buffer expression in set
     */
    static size_t get_buffer_idx(const BufferExpressionPtr& target, const BufferPool& pool);
    /**
     * @brief Create adjacency matrix for Buffer system. See comment in the method for more details.
     * @param loop_manager the loop manager
     * @param begin begin iterator
     * @param end end iterator
     * @param pool set of Buffers from the Linear IR
     * @return adjacency matrix where True value means that Buffers are adjacent and cannot have the same ID
     */
    static std::vector<bool> create_adjacency_matrix(const LoopManagerPtr& loop_manager, LinearIR::constExprIt begin, LinearIR::constExprIt end,
                                                     const BufferPool& pool);
    /**
     * @brief Algorithm of Graph coloring where vertices are Buffers
     * @param buffers set of Buffers from the Linear IR
     * @param adj adjacency matrix
     * @return map [color id -> Buffer set]
     */
    static std::map<size_t, BufferPool> coloring(BufferPool& buffers, std::vector<bool>& adj);
    /**
     * @brief Update the adjacency matrix:
     *         - If Buffers are from the same Loops and connected to the same Loop and
     *           they have not proportionally ptr shift params for this Loop, the Buffers are adjacent - set value True in the matrix;
     *         - If one of Buffer inside Loop but another Buffer is connected to this Loop and this Buffer has not zero data shift params,
     *           the Buffers are adjacent - set value True in the matrix;
     * @param lhs Pair where first value if Expression with first Buffer and second value is data pointer shift params for its
     * @param rhs Pair where first value if Expression with second Buffer and second value is data pointer shift params for its
     * @param buffers set of Buffers from the Linear IR
     * @param adj Target adjacency matrix
     */
    static void update_adj_matrix(const BufferMap::value_type& lhs, const BufferMap::value_type& rhs, const BufferPool& buffers,
                                  std::vector<bool>& adj);

    /**
     * @brief Check if two Buffers are adjacent and cannot have the same ID
     * @param lhs LoopPortInfo (Port and Data pointer shift params for first Buffer)
     * @param rhs LoopPortInfo (Port and Data pointer shift params for second Buffer)
     * @return Returns True if they are adjacent, otherwise returns False
     */
    static bool are_adjacent(const BufferMap::value_type& lhs, const BufferMap::value_type& rhs);

    /**
     * @brief Find all buffers that are connected to the current Loop
     * @param loop_info current unified loop info
     * @return buffer map
     */
    static BufferMap get_buffer_loop_neighbours(const UnifiedLoopInfoPtr& loop_info);
    /**
     * @brief Find all buffers that are inside the current Loop.
     * @param loop_end_it expression iterator in LinearIR of the target LoopEnd
     * @return set of inner buffers
     */
    static BufferMap get_buffer_loop_inside(const LinearIR::constExprIt& loop_end_it);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
