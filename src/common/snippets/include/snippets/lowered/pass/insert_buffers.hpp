// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface InsertBuffers
 * @brief The pass inserts Buffer between output ports of one loop (or Brgemm) and
 *        input ports of another loop (or Brgemm) to store intermediate data.
 *        The pass should be called after FuseLoops.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation:
 *        shape[shape_rank - normalize(m_allocation_rank) : shape_rank]
 * @ingroup snippets
 */
class InsertBuffers : public RangedPass {
public:
    OPENVINO_RTTI("InsertBuffers", "", RangedPass);
    InsertBuffers() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    static void insertion(LinearIR& linear_ir,
                          const LinearIR::constExprIt& begin_it,
                          const LoopManagerPtr& loop_manager,
                          const std::vector<ExpressionPort>& loop_entries,
                          const std::vector<ExpressionPort>& loop_exits);

    static LinearIR::constExprIt insertion_position(const LinearIR& linear_ir,
                                                    const LoopManagerPtr& loop_manager,
                                                    const ExpressionPtr& expr,
                                                    const ExpressionPtr& down_expr);
};

}  // namespace ov::snippets::lowered::pass
