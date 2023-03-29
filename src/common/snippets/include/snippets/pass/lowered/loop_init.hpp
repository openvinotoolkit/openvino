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
    LoopInit();
    bool run(LoweredExprIR& linear_ir) override;

private:
    bool insertion(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                   size_t loop_id, size_t dim_idx, bool has_outer_loop);
    std::vector<int64_t> init_ptr_increments(const std::vector<LoweredExprPort>& loop_inputs,
                                             const std::vector<LoweredExprPort>& loop_outputs,
                                             size_t dim_idx) const;
    std::vector<int64_t> init_finalization_offsets(const std::vector<int64_t>& finalization_offsets, size_t work_amount) const;
    std::vector<int64_t> init_element_type_sizes(const std::vector<LoweredExprPort>& loop_inputs,
                                                 const std::vector<LoweredExprPort>& loop_outputs);
    void reuse_buffer_increments(std::vector<int64_t>& ptr_increments,
                                 std::vector<int64_t>& finalization_offsets,
                                 const LoweredExprIR& linear_ir,
                                 const std::vector<LoweredExprPort>& loop_inputs,
                                 const std::vector<LoweredExprPort>& loop_outputs);
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
