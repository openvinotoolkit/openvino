// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_expanded_loops.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

#define INFORMATIVE_ASSERT(cond, ...) \
    OPENVINO_ASSERT((cond), "Failed to validate ExpandedLoops: ", __VA_ARGS__)

namespace {
bool is_inner_splitted_tail(const ExpressionPtr& loop_expr, const LoopManagerPtr& loop_manager) {
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(loop_expr->get_node());
    INFORMATIVE_ASSERT(loop_end, "expects LoopEnd");
    const auto loop_id = loop_end->get_id();
    const auto expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_manager->get_loop_info(loop_id));
    INFORMATIVE_ASSERT(expanded_loop_info, "expects only ExpandedLoopInfo in LoopManager");
    // Inner Splitted Tail Loop has `MAIN_BODY` type for now after InsertSpecificIteration pass
    if (expanded_loop_info->get_type() != SpecificLoopIterType::MAIN_BODY)
        return false;
    const auto outer_loops = loop_expr->get_loop_ids();
    if (outer_loops.empty())
        return false;
    const auto outer_loop_id = outer_loops.front();
    const auto outer_expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_manager->get_loop_info(outer_loop_id));
    INFORMATIVE_ASSERT(outer_expanded_loop_info, "expects only ExpandedLoopInfo in LoopManager");
    return outer_expanded_loop_info->get_type() == SpecificLoopIterType::LAST_ITER &&
           expanded_loop_info->get_dim_idx() == outer_expanded_loop_info->get_dim_idx();
}

} // namespace

void ValidateExpandedLoops::validate_loop_information(const LinearIR& linear_ir) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_map = loop_manager->get_map();

    UnifiedLoopInfoPtr current_unified_loop_info = nullptr;
    std::vector<int64_t> total_finalization_offsets;
    size_t current_work_amount = 0;
    size_t num_ports = 0;

    for (const auto& p : loop_map) {
        const auto& expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(p.second);
        INFORMATIVE_ASSERT(expanded_loop_info, "expects only ExpandedLoopInfo in LoopManager");

        if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info) {
            // If there is `current_unified_loop_info` - the previos loop is finished and need to validate total information
            if (current_unified_loop_info) {
                INFORMATIVE_ASSERT(current_work_amount == current_unified_loop_info->get_work_amount(),
                                   "total work amount of expanded loops is not equal to work amount of undefined loop");
                INFORMATIVE_ASSERT(total_finalization_offsets == current_unified_loop_info->get_finalization_offsets(),
                                   "total finalization offsets are not equal to finalization offsets of undefined loop");
            }

            current_unified_loop_info = expanded_loop_info->get_unified_loop_info();

            INFORMATIVE_ASSERT(current_unified_loop_info->get_input_count() == expanded_loop_info->get_input_count() &&
                               current_unified_loop_info->get_output_count() == expanded_loop_info->get_output_count(),
                               "incompatible loop ports with UnifiedLoopInfo");

            current_work_amount = 0;
            num_ports = expanded_loop_info->get_input_count() + expanded_loop_info->get_output_count();
            total_finalization_offsets.clear();
            total_finalization_offsets.resize(num_ports, 0);
        }

        current_work_amount = utils::dynamic_safe_add(current_work_amount, expanded_loop_info->get_work_amount());
        INFORMATIVE_ASSERT(current_unified_loop_info->get_ptr_increments() == expanded_loop_info->get_ptr_increments(),
                           "incompatible pointer increments with UnifiedLoopInfo");

        const auto& finalization_offsets = expanded_loop_info->get_finalization_offsets();
        INFORMATIVE_ASSERT(finalization_offsets.size() == total_finalization_offsets.size(),
                           "incompatible finalization offset count");
        for (size_t i = 0; i < num_ports; ++i)
            total_finalization_offsets[i] = utils::dynamic_safe_add(total_finalization_offsets[i], finalization_offsets[i]);
    }
}

void ValidateExpandedLoops::validate_loop_expressions(const LinearIR& linear_ir) {
    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> unique_loop_ids;
    for (const auto& expr : linear_ir) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node())) {
            const auto loop_id = loop_end->get_id();
            unique_loop_ids.insert(loop_id);

            // At the moment, InnerSpliitedTail LoopEnd is not compatible with ExpandedLoopInfo
            // Validation is not supported
            if (is_inner_splitted_tail(expr, loop_manager))
                continue;

            const auto expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(loop_manager->get_loop_info(loop_id));
            INFORMATIVE_ASSERT(expanded_loop_info, "expects only ExpandedLoopInfo in LoopManager");

            INFORMATIVE_ASSERT(loop_end->get_work_amount() == expanded_loop_info->get_work_amount(),
                               "incompatible work amount of LoopEnd and ExpandedLoopInfo");
            INFORMATIVE_ASSERT(loop_end->get_increment() == expanded_loop_info->get_increment(),
                               "incompatible increment of LoopEnd and ExpandedLoopInfo");
            INFORMATIVE_ASSERT(loop_end->get_element_type_sizes() == expanded_loop_info->get_data_sizes(),
                               "incompatible element sizes of LoopEnd and ExpandedLoopInfo");
            INFORMATIVE_ASSERT(loop_end->get_ptr_increments() == expanded_loop_info->get_ptr_increments(),
                               "incompatible pointer increments of LoopEnd and ExpandedLoopInfo");
            INFORMATIVE_ASSERT(loop_end->get_finalization_offsets() == expanded_loop_info->get_finalization_offsets(),
                               "incompatible finalization offsets of LoopEnd and ExpandedLoopInfo");
        }
    }
    INFORMATIVE_ASSERT(unique_loop_ids.size() == loop_manager->get_map().size(),
                      "incompatible loopIDs of inserted LoopEnd expressions and LoopInfo in LoopManager");
}

bool ValidateExpandedLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateExpandedLoops")

    // Firstly, validate mapping compatibility between ExpandedLoopInfo and original UnifiedLoopInfo
    validate_loop_information(linear_ir);
    // Secondly, validate that LoopEnd contains the information from ExpandedLoopInfo
    validate_loop_expressions(linear_ir);

    return true;
}

#undef INFORMATIVE_ASSERT

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
