// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface LoopInit
 * @brief The pass explicitly insert LoadBegin and LoadEnd in Linear IR using Loop markup
 * @ingroup snippets
 */
class LoopInit : public LinearIRTransformation {
public:
    OPENVINO_RTTI("InsertLoops", "LinearIRTransformation")
    LoopInit(size_t vector_size);
    bool run(LoweredExprIR& linear_ir) override;

private:
    bool insertion(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                   size_t loop_id, size_t dim_idx, bool has_outer_loop);
    std::vector<int64_t> init_ptr_increments(LoweredExprIR& linear_ir,
                                             const std::vector<LoweredExprPtr>& loop_in_exprs,
                                             const std::vector<LoweredExprPtr>& loop_out_exprs,
                                             size_t dim_idx) const;
    std::vector<int64_t> init_finalization_offsets(const std::vector<int64_t>& ptr_increments, size_t work_amount) const;

    size_t m_vector_size;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
