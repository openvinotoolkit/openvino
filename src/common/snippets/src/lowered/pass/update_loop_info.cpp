// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/update_loop_info.hpp"

#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void UpdateLoopInfo::init_data_ptr_shifts(const UnifiedLoopInfoPtr& unified_loop_info, std::vector<int64_t>& ptr_increments,
                                          std::vector<int64_t>& finalization_offsets) {
    const auto count = unified_loop_info->get_input_count() + unified_loop_info->get_output_count();
    ptr_increments.resize(count);
    finalization_offsets.resize(count);

    size_t idx = 0;
    unified_loop_info->iterate_through_descs(
        [&ptr_increments, &finalization_offsets, &idx](const UnifiedLoopInfo::LoopPortDesc& desc) {
            ptr_increments[idx] = desc.ptr_increment;
            finalization_offsets[idx] = desc.finalization_offset;
            ++idx;
        });
}

bool UpdateLoopInfo::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::UpdateLoopInfo")

    // Initialized UnifiedLoopInfo
    struct CurrentUnifiedLoopInfo {
        UnifiedLoopInfoPtr updated_unified_loop_info = nullptr;
        size_t current_work_amount = 0;
        std::vector<int64_t> ptr_increments;
        std::vector<int64_t> finalization_offsets;
    };
    std::unordered_map<UnifiedLoopInfoPtr, CurrentUnifiedLoopInfo> initializated_info_map;

    const auto& loop_map = linear_ir.get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        const auto& expanded_loop_info = ov::as_type_ptr<ExpandedLoopInfo>(p.second);
        OPENVINO_ASSERT(expanded_loop_info, "UpdateLoopInfo expects ExpandedLoopInfo in LoopManager");

        // First visiting of unified (whole) loop
        const auto& current_unified_loop_info = expanded_loop_info->get_unified_loop_info();
        if (initializated_info_map.count(current_unified_loop_info) == 0) {
            auto& current_info = initializated_info_map[current_unified_loop_info];
            // make a copy to avoid original loop info corruption
            current_info.updated_unified_loop_info = std::make_shared<UnifiedLoopInfo>(*current_unified_loop_info);
            InitLoops::init_loop_info(current_info.updated_unified_loop_info, true);

            current_info.current_work_amount = current_info.updated_unified_loop_info->get_work_amount();
            init_data_ptr_shifts(current_info.updated_unified_loop_info, current_info.ptr_increments, current_info.finalization_offsets);
        }

        auto& initializated_info = initializated_info_map.at(current_unified_loop_info);
        auto& current_work_amount = initializated_info.current_work_amount;
        const auto& updated_unified_loop_info = initializated_info.updated_unified_loop_info;
        const auto& ptr_increments = initializated_info.ptr_increments;
        const auto& finalization_offsets = initializated_info.finalization_offsets;

        const auto& decomposed_loop_type = expanded_loop_info->get_type();

        // If the specific iteration is not needed, we skip loop evaluation - set zero as work amount is enough
        if (!InsertSpecificIterations::is_decomposed_loop_needed(updated_unified_loop_info, decomposed_loop_type, current_work_amount)) {
            expanded_loop_info->set_work_amount(0);
            continue;
        }

        expanded_loop_info->set_work_amount(
            InsertSpecificIterations::get_decomposed_loop_work_amount(updated_unified_loop_info, decomposed_loop_type, current_work_amount));
        // Update remaining Loop work amount
        current_work_amount -= expanded_loop_info->get_work_amount();

        expanded_loop_info->update_ptr_increments(ptr_increments);
        if (current_work_amount > 0) {
            expanded_loop_info->update_finalization_offsets(std::vector<int64_t>(finalization_offsets.size(), 0));
        } else {
            expanded_loop_info->update_finalization_offsets(finalization_offsets);
        }
    }
    return true;
}
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
