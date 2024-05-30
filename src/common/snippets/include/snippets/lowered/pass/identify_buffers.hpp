// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface IdentifyBuffers
 * @brief The pass set identifiers for Buffers in common Buffer system.
 *        The buffers with the same identifier will be assigned the same data register.
 *        The pass uses greedy graph coloring algorithm using adjacency matrix:
 *          - Buffers - are vertices of graph;
 *          - Loops, Brgemm (the same other ops) - are "edges" between Buffers (hub of edges).
 *                   The buffers are connected to the same Loop - are adjacent in graph sense bounds.
 *          - The vertices (buffers) are adjacent if they are connected to the same Loop and
 *            their data pointers cannot be proportionally incremented in Loops: different ptr increments or data sizes -
 *            or one of the Buffers is in some a Loop but another Buffer is not;
 *          - Firstly, create adjacency matrix using the definition above;
 *          - Secondly, assign the same color to non-adjacent vertices of graph (buffers), and use different colors otherwise.
 *        Note: should be called before ResetBuffer() pass to have correct offsets
 * @ingroup snippets
 */
class IdentifyBuffers: public RangedPass {
public:
    OPENVINO_RTTI("IdentifyBuffers", "RangedPass")
    IdentifyBuffers() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    struct ShiftPtrParams {
        ShiftPtrParams() = default;
        ShiftPtrParams(int64_t ds, int64_t pi, int64_t fo) : data_size(ds), ptr_increment(pi), finalization_offset(fo) {}
        int64_t data_size = 0;
        int64_t ptr_increment = 0;
        int64_t finalization_offset = 0;

        inline bool is_static() const {
            return !utils::is_dynamic_value(ptr_increment) && !utils::is_dynamic_value(finalization_offset);
        }

        friend bool operator==(const ShiftPtrParams& lhs, const ShiftPtrParams& rhs);
        friend bool operator!=(const ShiftPtrParams& lhs, const ShiftPtrParams& rhs);
    };

    /**
     * @brief Check if two Buffers can reuse ID by ShiftPtrParams < data_size, ptr_increment, finalization_offset >
     * @param lhs Data pointer shift params for first Buffer
     * @param rhs Data pointer shift params for second Buffer
     * @return Returns True if params are valid for reusing. Otherwise returns False
     */
    static bool can_reuse_id(const ShiftPtrParams& lhs, const ShiftPtrParams& rhs);

private:
    using BufferPool = std::vector<ExpressionPtr>;
    using BufferMap = std::map<ExpressionPtr, ShiftPtrParams>;

    /**
     * @brief Get Buffer Index in Buffer set
     * @param target the target Buffer expression
     * @param pool set of Buffers from the Linear IR
     * @return index of target Buffer expression in set
     */
    static size_t get_buffer_idx(const ExpressionPtr& target, const BufferPool& pool);
    /**
     * @brief Create adjacency matrix for Buffer system. See comment in the method for more details.
     * @param linear_ir the target Linear IR
     * @param pool set of Buffers from the Linear IR
     * @return adjacency matrix where True value means that Buffers are adjacent and cannot have the same ID
     */
    static std::vector<bool> create_adjacency_matrix(lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end, const BufferPool& pool);
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
    static void update_adj_matrix(const std::pair<ExpressionPtr, ShiftPtrParams>& lhs,
                                  const std::pair<ExpressionPtr, ShiftPtrParams>& rhs,
                                  const BufferPool& buffers,
                                  std::vector<bool>& adj);
    /**
     * @brief Check if two Buffers are adjacent and cannot have the same ID
     * @param lhs Pair where first value is Expression with first Buffer and second value is data pointer shift params for it
     * @param rhs Pair where first value is Expression with second Buffer and second value is data pointer shift params for it
     * @return Returns True if they are adjacent, otherwise returns False
     */
    static bool are_adjacent(const std::pair<ExpressionPtr, ShiftPtrParams>& lhs,
                             const std::pair<ExpressionPtr, ShiftPtrParams>& rhs);

    /**
     * @brief Find all buffers that are connected to the current LoopEnd
     * @param loop_end_expr expression of the target LoopEnd
     * @return buffer map [buffer expr -> ShiftDataPtrs]
     */
    static BufferMap get_buffer_loop_neighbours(const ExpressionPtr& loop_end_expr);
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
