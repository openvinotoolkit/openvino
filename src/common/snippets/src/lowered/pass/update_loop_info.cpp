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
    const auto& entry_ports = unified_loop_info->get_entry_points();
    const auto& exit_ports = unified_loop_info->get_exit_points();
    const auto in_num = entry_ports.size();
    const auto out_num = exit_ports.size();
    ptr_increments.resize(in_num + out_num);
    finalization_offsets.resize(in_num + out_num);

    auto init_shifts = [&](const std::vector<LoopPort>& loop_ports, size_t start_index) {
        for (size_t i = 0; i < loop_ports.size(); ++i) {
            const auto& loop_port = loop_ports[i];
            ptr_increments[start_index + i] = loop_port.ptr_increment;
            finalization_offsets[start_index + i] = loop_port.finalization_offset;
        }
    };
    init_shifts(entry_ports, 0);
    init_shifts(exit_ports, in_num);
}

bool UpdateLoopInfo::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::UpdateLoopInfo")

    std::vector<int64_t> ptr_increments, finalization_offsets;
    UnifiedLoopInfoPtr current_unified_loop_info = nullptr, updated_unified_loop_info = nullptr;
    size_t current_work_amount = 0;
    size_t num_ports = 0;

    const auto& loop_map = linear_ir.get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        const auto& expanded_loop_info = std::dynamic_pointer_cast<ExpandedLoopInfo>(p.second);
        OPENVINO_ASSERT(expanded_loop_info, "UpdateLoopInfo expects ExpandedLoopInfo in LoopManager");
        if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info) {
            current_unified_loop_info = expanded_loop_info->get_unified_loop_info();

            // make a copy to avoid original loop info corruption
            updated_unified_loop_info = std::make_shared<UnifiedLoopInfo>(*current_unified_loop_info);
            InitLoops::init_loop_info(updated_unified_loop_info, true);

            current_work_amount = updated_unified_loop_info->get_work_amount();
            init_data_ptr_shifts(updated_unified_loop_info, ptr_increments, finalization_offsets);
            num_ports = ptr_increments.size();
        }

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

        std::memcpy(expanded_loop_info->get_dense_ptr_increments().data(), ptr_increments.data(), num_ports * sizeof(int64_t));
        if (current_work_amount > 0) {
            std::fill(expanded_loop_info->get_dense_finalization_offsets().begin(), expanded_loop_info->get_dense_finalization_offsets().end(), 0);
        } else {
            std::memcpy(expanded_loop_info->get_dense_finalization_offsets().data(), finalization_offsets.data(), num_ports * sizeof(int64_t));
        }
    }
    return true;
}
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
