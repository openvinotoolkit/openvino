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
 * @interface LoopFusion
 * @brief The pass fuses marking Loops.
 * @ingroup snippets
 */
class LoopFusion : public LinearIRTransformation {
public:
    OPENVINO_RTTI("LoopFusion", "LinearIRTransformation")
    LoopFusion();
    bool run(LoweredExprIR& linear_ir) override;

private:
    static bool can_be_fused(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_current,
                             const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_target);
    static bool fuse_upper_into_current(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                        const LoweredExprPort& current_entry_point, const LoweredExprPort& target_exit_point,
                                        size_t current_loop_id, size_t target_loop_id, size_t dim_idx,
                                        LoweredExprIR::constExprIt& current_loop_begin_pos, LoweredExprIR::constExprIt& current_loop_end_pos);
    static bool fuse_lower_into_current(LoweredExprIR& linear_ir, const LoweredExprIR::LoweredLoopManagerPtr& loop_manager,
                                        const LoweredExprPort& current_entry_point, const LoweredExprPort& target_exit_point,
                                        size_t current_loop_id, size_t target_loop_id, size_t dim_idx,
                                        LoweredExprIR::constExprIt& current_loop_begin_pos, LoweredExprIR::constExprIt& current_loop_end_pos);
    static void fuse_points(LoweredExprIR& linear_ir, std::vector<LoweredExprPort>& exit_points, std::vector<LoweredExprPort>& entry_points,
                            LoweredExprIR::constExprIt loop_begin_pos, LoweredExprIR::constExprIt loop_end_pos);
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
