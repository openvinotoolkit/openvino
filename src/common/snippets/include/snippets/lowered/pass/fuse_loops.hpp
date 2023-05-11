// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface FuseLoops
 * @brief The pass fuses marking Loops.
 * @ingroup snippets
 */
class FuseLoops : public Transformation {
public:
    OPENVINO_RTTI("FuseLoops", "Transformation")
    FuseLoops();
    bool run(LinearIR& linear_ir) override;

private:
    static bool can_be_fused(const LinearIR::LoopManager::LoopInfoPtr& loop_current,
                             const LinearIR::LoopManager::LoopInfoPtr& loop_target);
    static bool fuse_upper_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, const ExpressionPort& current_entry_point,
                                        size_t current_loop_id, size_t target_loop_id, size_t dim_idx,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos);
    static bool fuse_lower_into_current(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, const ExpressionPort& current_entry_point,
                                        size_t current_loop_id, size_t target_loop_id, size_t dim_idx,
                                        LinearIR::constExprIt& current_loop_begin_pos, LinearIR::constExprIt& current_loop_end_pos);
    static void fuse_points(std::vector<ExpressionPort>& exit_points, std::vector<ExpressionPort>& entry_points,
                            LinearIR::constExprIt loop_begin_pos, LinearIR::constExprIt loop_end_pos);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
