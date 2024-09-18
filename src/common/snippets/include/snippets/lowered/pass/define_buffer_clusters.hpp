// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface DefineBufferClusters
 * @brief The pass defines buffer clusters. The buffers from one cluster share the
 *        same memory (has the same offset relative to the data pointer of buffer scratchpad).
 *         - If MemoryAccess op or Loop can read and write to the same (inplace behavior), the Buffers should be in the one cluster.
 *         - If Buffer is in the Loop which read or write from/to the other Buffers, this Buffer can emulate `window` slidings.
 *           It means that Buffer inside can reuse memory of Buffers outside in bounds of full Loop work.
 *           Demonstration:
 *                               |-----------------------------------------------------|
 *                               | |------------|                       |------------| |                        InnerLoops have work amount 128
 *             Buffer0 [3x128]-> | | InnerLoop0 | -> Buffer1 [3x128] -> | InnerLoop1 | | -> Buffer2 [3x128]     OuterLoop has work amount 3
 *                               | |------------|      OuterLoop        |------------| |
 *                               |-----------------------------------------------------|
 *           Buffer1 can reuse memory [128] of Buffer0 or Buffer2 in each iteration of OuterLoop
 *           Note: The pass requires expression enumeration and buffer identification (for nested Buffers inplace).
 *                 These passes should be executed separately before this pass!
 * @ingroup snippets
 */
class DefineBufferClusters : public RangedPass {
public:
    OPENVINO_RTTI("DefineBufferClusters", "RangedPass")

    DefineBufferClusters() = default;

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    bool run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    using BufferCluster = std::set<BufferExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;
    using BufferPorts = std::unordered_map<BufferExpressionPtr, std::set<size_t>>;
    /**
     * @brief Finds Buffer cluster in set of clusters which contains the target expression with Buffer
     * @param target target expression with Buffer op
     * @return vector iterator which refers to the found cluster
     */
    BufferClusters::iterator find_cluster_by_expr(const BufferExpressionPtr& target);
    /**
     * @brief Returns True if Buffer is direct source for the target expr (there aren't other loop between the Buffer and target expr)
     * @param buffer_expr expression with assumed Buffer op
     * @param target_expr expression with target op - LoopEnd or MemoryAccess op
     * @return boolean value
     */
    bool is_direct_buffer(const BufferExpressionPtr& buffer_expr, const ExpressionPtr& target_expr) const;
    /**
     * @brief Creates new buffer cluster if buffer_exprs is missed in clusters. If buffer_exprs is already in clusters, do nothing
     * @param buffer_expr expression with Buffer op
     */
    void create_new_cluster(const BufferExpressionPtr& buffer_expr);
    /**
     * @brief Returns common ID of cluster if all buffer inside have the same Buffer ID. Otherwise returns the default value SIZE_MAX
     *        that means that Buffers in cluster have different IDs.
     * @param cluster set of Buffer expressions - cluster
     * @return common buffer ID or SIZE_MAX - size value
     */
    size_t get_cluster_buffer_id(const BufferCluster& cluster) const;

    /**
     * @brief Analyzes Loop: if Loop has Buffer ops on inputs and outputs, Loop can read and write from/to the same memory.
     * @param expr_it iterator of Linear IR which refers to the expression with LoopEnd
     */
    void parse_loop(const LinearIR::constExprIt& expr_it);
    /**
     * @brief Analyzes full MemoryAccess op: if the op has Buffer ops on I/O, the op can read and write from/to the same memory.
     * @param expr expression with full MemoryAccess op
     */
    void parse_memory_access_op(const ExpressionPtr& expr);
    /**
     * @brief Gets input outputs buffers of Loop
     * @param loop_expr expression with LoopEnd op
     * @return unordered map [Expression -> set of input ports] which represents input Buffers of Loop
     */
    BufferPorts get_input_buffers(const ExpressionPtr& loop_expr) const;
    /**
     * @brief Gets output buffers of Loop
     * @param loop_expr expression with LoopEnd op
     * @return unordered map [Expression -> set of input ports] which represents output Buffers of Loop
     */
    BufferPorts get_output_buffers(const ExpressionPtr& loop_expr) const;
    /**
     * @brief Analyzes nested Loops: unite nested buffer clusters if they can reproduce `window` sliding
     * @param input_buffers unordered map [Expression -> set of input ports] which represents input Buffers of Loop
     * @param output_buffers unordered map [Expression -> set of output ports (one)] which represents output Buffers of Loop
     * @param outer_loop_end_expr_it iterator of Linear IR which refers to the expression with outer LoopEnd
     */
    void parse_nested_loops(const BufferPorts& input_buffers, const BufferPorts& output_buffers, const LinearIR::constExprIt& outer_loop_end_expr_it);
    /**
     * @brief Finds the last connected Loop to the target Buffer and returns the corresponding finalization offset
     * @param buffer_expr expression with Buffer op
     * @return finalization offset - int64_t value
     */
    int64_t get_buffer_finalization_offset(const BufferExpressionPtr& buffer_expr) const;
    /**
     * @brief Check if two Buffer expressions are connected to the same Loop. Set common LoopEnd as `loop` parameter and
     *        indexes of Loop ports `up_idx` and `down_idx` if Buffers are really neighbours
     * @param up expression with upper Buffer op
     * @param down expression with lower Buffer op
     * @param loop expression with common LoopEnd op
     * @param up_idx the reference to port index of upper Buffer op to the Loop
     * @param down_idx the reference to port index of lower Buffer op to the Loop
     * @return Return True if the Buffers are connected to the same Loop
     */
    static bool are_buffer_neighbours(const BufferExpressionPtr& up, const BufferExpressionPtr& down, ExpressionPtr& loop,
                                      size_t& up_idx, size_t& down_idx);
    /**
     * @brief Unite clusters
     * @param inner_cluster_it iterator to inner cluster - buffer cluster is in the loop
     * @param outer_cluster buffer clusters with buffers outside the Loop
     * @param outer_buffer target Buffer from outer_cluster
     * @param is_outer_up true if outer buffer is upper in Linear IR than inner Buffers
     * @return Return True if clusters have been united
     */
    bool unite_nested_clusters(const BufferClusters::iterator& inner_cluster_it, BufferCluster& outer_cluster,
                               const BufferExpressionPtr& outer_buffer, bool is_outer_up);

    BufferClusters m_clusters;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
