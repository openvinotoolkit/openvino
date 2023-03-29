// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface BufferInsertion
 * @brief The pass inserts Buffer between exit points of one loop (or Brgemm) and
 *        entry points of another loop (or Brgemm) to store intermediate data.
 *        The pass should be called after LoopFusion.
 * @param m_buffer_allocation_rank - rank of shape for memory allocation: shape[shape_rank - normalize(m_allocation_rank) : shape_rank]
 * @ingroup snippets
 */
class BufferInsertion : public LinearIRTransformation {
public:
    OPENVINO_RTTI("BufferInsertion", "LinearIRTransformation")
    BufferInsertion(int32_t buffer_allocation_rank);
    bool run(LoweredExprIR& linear_ir) override;

private:
    void insertion(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManagerPtr& loop_manager, size_t loop_id,
                   const std::vector<LoweredExprPort>& loop_entries, const std::vector<LoweredExprPort>& loop_exits);

    LoweredExprIR::constExprIt insertion_position(const LoweredExprIR& linear_ir,
                                                  const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                                  const LoweredExprPtr& up_expr, const LoweredExprPtr& down_expr);

    int32_t m_buffer_allocation_rank;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph