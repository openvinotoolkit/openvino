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
 * @brief The pass inserts Buffer between output ports of one loop (or Brgemm) and
 *        input ports of another loop (or Brgemm) to store intermediate data.
 *        The pass should be called after FuseLoops.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank]
 * @ingroup snippets
 */
class InsertBuffers : public RangedPass {
public:
    OPENVINO_RTTI("InsertBuffers", "RangedPass")
    InsertBuffers() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    void insertion(LinearIR& linear_ir,
                   const LinearIR::constExprIt& begin_it,
                   const LinearIR::constExprIt& end_it,
                   const LoopManagerPtr& loop_manager,
                   const std::vector<LoopPort>& loop_entries,
                   const std::vector<LoopPort>& loop_exits) const;

    static LinearIR::constExprIt insertion_position(const LinearIR& linear_ir,
                                                    const LoopManagerPtr& loop_manager,
                                                    const ExpressionPtr& expr,
                                                    const ExpressionPtr& down_expr);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov