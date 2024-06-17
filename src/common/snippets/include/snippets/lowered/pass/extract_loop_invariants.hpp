// Copyright (C) 2024 Intel Corporation
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
 * @interface ExtractLoopInvariants
 * @brief Extract the exprs that produce identical result in each loop iteration outside the loop
 * @ingroup snippets
 */
class ExtractLoopInvariants : public RangedPass {
public:
    OPENVINO_RTTI("ExtractLoopInvariants", "RangedPass")
    ExtractLoopInvariants() = default;
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    static std::set<ExpressionPtr> get_potential_extractable_exprs(const std::vector<LoopPort>& loop_in_ports);
    static bool is_extraction_applicable(const ExpressionPtr& expr, const UnifiedLoopInfoPtr& inner_loop_info);
    static void extract_expr(const ExpressionPtr& expr, LinearIR& linear_ir,
                      LinearIR::constExprIt& inner_loop_begin_pos, LinearIR::constExprIt& inner_loop_end_pos);
    static void update_loop_ports(const ExpressionPtr& expr, const LoopManagerPtr& loop_manager, size_t inner_loop_id,
                           LinearIR::constExprIt& inner_loop_begin_pos, LinearIR::constExprIt& inner_loop_end_pos);
    static bool extract_from_loop(const size_t& inner_loop_id, LinearIR& linear_ir);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
