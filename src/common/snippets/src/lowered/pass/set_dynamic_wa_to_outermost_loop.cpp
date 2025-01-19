// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_dynamic_wa_to_outermost_loop.hpp"

#include  "snippets/lowered/pass/mha_parallel_wa_optimizer.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/utils/loop_utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SetDynamicWAToOuterMostLoop::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetDynamicWAToOuterMostLoop")
    if (linear_ir.empty() || !linear_ir.is_dynamic() || linear_ir.get_config().m_enable_domain_optimization)
        return false;

    const auto linear_ir_ptr = std::make_shared<const LinearIR>(linear_ir);
    const auto brgemms = MHAParallelWAOptimizer::find_applicable_brgemms(linear_ir_ptr, false);
    if (brgemms.empty())
        return false;

    const auto unsqueezed_params = MHAParallelWAOptimizer::find_unsqueezed_params(linear_ir_ptr, brgemms);
    OPENVINO_ASSERT(!unsqueezed_params.empty(), "unsqueezed_params mustn't be empty after initialization");


    const auto& loop_manager = linear_ir_ptr->get_loop_manager();
    std::unordered_set<lowered::UnifiedLoopInfoPtr> affected_loops;
    size_t prev_loop_id = std::numeric_limits<size_t>::max();
    static const size_t dim_M_idx = 1;

    auto add_affected_loop = [&](const lowered::ExpressionPtr& expr) {
        const auto& loop_idces = expr->get_loop_ids();
        if (loop_idces.empty() || loop_idces.front() == prev_loop_id)
            return;

        prev_loop_id = loop_idces.front();
        const auto loop_info = loop_manager->get_loop_info<lowered::UnifiedLoopInfo>(prev_loop_id);
        if (loop_info->get_dim_idx() == dim_M_idx) {
            affected_loops.insert(loop_info);
        }
    };

    size_t i = 0;
    std::unordered_set<lowered::ExpressionPtr> visited;
    for (const auto& param : linear_ir_ptr->get_parameters()) {
        if (unsqueezed_params.count(i++))
            continue;
        utils::visit_path(param, visited, add_affected_loop, false);
    }

    bool modified = false;
    for (const auto& loop : affected_loops) {
        if (!utils::is_dynamic_value(loop->get_work_amount())) {
            loop->set_work_amount(utils::get_dynamic_value<size_t>());
            ov::snippets::utils::update_data_pointer_shifts(loop);
            modified = true;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov