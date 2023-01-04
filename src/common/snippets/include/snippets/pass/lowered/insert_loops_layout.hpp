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
 * @interface InsertLoops
 * @brief Insert explicit Loop operations into the body to process multiple data entities during one kernel execution
 * @param vector_size - the number of entities processed on one iteration of vector loop
 * @param explicit_loop_insertion - true, if we can just insert LoopBegin on inputs and LoopEnd on outputs, othwerwise
 *                           the pass goes all over the body analyzing where LoopBegin and LoopEnd should be inserted:
 *                           synchronization nodes are MatMul, Buffer and other already existing Loops.
 * @ingroup snippets
 */
class InsertLoopsLayout : public LinearIRTransformation {
    size_t m_vector_size;
    int32_t m_buffer_allocation_rank;
    LoweredExprIR::exprIt inject_store_buffer_load(LoweredExprIR::exprIt loop_end_pos, const LoweredExprPtr& ancor_expr,
                                                                      LoweredExprIR& linear_ir) const;
public:
    OPENVINO_RTTI("InsertLoopsLayout", "LinearIRTransformation")
    InsertLoopsLayout(size_t vector_size, int32_t buffer_allocation_rank);
    bool run(LoweredExprIR& linear_ir) override;
    bool static inject_loops(LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos,
                             LoweredExprIR& linear_ir, size_t loop_depth, size_t vector_size);
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
