// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "allocate_buffer_memory.hpp"
#include "snippets/op/buffer.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface DefineBufferClusters
 * @brief TODO
 * @ingroup snippets
 */

class DefineBufferClusters : public Pass {
public:
    OPENVINO_RTTI("DefineBufferClusters", "Pass")
    bool run(lowered::LinearIR& linear_ir) override;

    AllocateBufferMemory::BufferClusters get_clusters() const { return m_clusters; }

private:
    AllocateBufferMemory::BufferClusters::iterator find_cluster_by_expr(const ExpressionPtr& target);
    // Return True if Buffer is direct source for the target expr (there aren't other loop between the Buffer and target expr)
    bool is_direct_buffer(const ExpressionPtr& buffer_expr, const ExpressionPtr& target_expr) const;
    void create_new_cluster(const ExpressionPtr& buffer_expr);
    size_t get_cluster_buffer_id(const AllocateBufferMemory::BufferCluster& cluster) const;

    void parse_loop(const LinearIR::constExprIt& expr_it);
    void parse_memory_access_op(const ExpressionPtr& expr);

    void unite_clusters_in_nested_loops(const std::unordered_map<ExpressionPtr, std::set<size_t>>& input_buffers,
                                        const std::unordered_map<ExpressionPtr, size_t>& output_buffers,
                                        const LinearIR::constExprIt& outer_loop_end_expr_it);

    int64_t get_buffer_finalization_offsets(const ExpressionPtr& buffer_expr) const;

    AllocateBufferMemory::BufferClusters m_clusters;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
