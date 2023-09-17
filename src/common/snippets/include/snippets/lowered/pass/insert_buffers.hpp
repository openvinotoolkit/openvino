// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertBuffers
 * @brief The pass inserts Buffer between exit points of one loop (or Brgemm) and
 *        entry points of another loop (or Brgemm) to store intermediate data.
 *        The pass should be called after FuseLoops.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank]
 * @ingroup snippets
 */
class InsertBuffers : public Pass {
public:
    OPENVINO_RTTI("InsertBuffers", "Pass")
    InsertBuffers(int32_t buffer_allocation_rank);
    bool run(LinearIR& linear_ir) override;

private:
    void insertion(LinearIR& linear_ir, const LinearIR::constExprIt& expr_it, const LinearIR::LoopManagerPtr& loop_manager,
                   const std::vector<LinearIR::LoopManager::LoopPort>& loop_entries,
                   const std::vector<LinearIR::LoopManager::LoopPort>& loop_exits);

    LinearIR::constExprIt insertion_position(const LinearIR& linear_ir,
                                             const LinearIR::LoopManagerPtr& loop_manager,
                                             const ExpressionPtr& expr,
                                             const ExpressionPtr& down_expr);

    int32_t m_buffer_allocation_rank;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov