// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

ValidateLoops::ValidateLoops() {}

bool ValidateLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();

    // Already validated vectors of Loop IDs
    std::set<std::vector<size_t>> validated_nested_loops;
    auto is_already_verified = [&validated_nested_loops](const std::vector<size_t>& ids) {
        // If `ids` is subsequence of one of loop_ids in `validated_nested_loops` set,
        // it means that `ids` is already validated too
        for (const auto& loop : validated_nested_loops) {
            if (std::search(ids.cbegin(), ids.cend(), loop.cbegin(), loop.cend()) != ids.cend()) {
                return true;
            }
        }
        return false;
    };

    std::vector<size_t> dim_indexes;

    auto validate_loop_ports = [&loop_manager, &dim_indexes, &validated_nested_loops, &is_already_verified](std::vector<LoopPort>& loop_ports) {
        for (auto& loop_port : loop_ports) {
            const auto expr = loop_port.expr_port->get_expr();
            const auto loop_ids = expr->get_loop_ids();
            // If loop_ids of the current port is subsequence of already validated IDs, skip
            if (is_already_verified(loop_ids))
                continue;

            dim_indexes.clear();
            dim_indexes.reserve(loop_ids.size());
            // Outer Loop -> Inner Loop
            for (size_t i = 0; i < loop_ids.size(); ++i) {
                const auto id = loop_ids[i];
                const auto dim_idx = loop_manager->get_loop_info(id)->dim_idx;
                if (std::find(dim_indexes.cbegin(), dim_indexes.cend(), dim_idx) != dim_indexes.cend()) {
                    OPENVINO_ASSERT(*dim_indexes.rbegin() == dim_idx,
                                    "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                    OPENVINO_ASSERT(loop_manager->get_loop_info(loop_ids[i - 1])->increment == loop_manager->get_loop_info(id)->work_amount,
                                    "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                    OPENVINO_ASSERT(loop_manager->get_loop_info(loop_ids[i - 1])->outer_splited_loop,
                                    "Incorrect Loop ID configuration: the outer Loop with splitted dimension should have `outer_splited_loop=True`");
                }
                OPENVINO_ASSERT(i == 0 || loop_manager->get_loop_info(loop_ids[i - 1])->dim_idx >= dim_idx,
                                "Incorrect Loop ID configuration: dim_idx should be sorted in accordance with loop nesting");
                dim_indexes.push_back(dim_idx);
            }
            validated_nested_loops.insert(loop_ids);
        }
    };

    for (const auto& pair : loops) {
        const auto& loop_info = pair.second;
        validate_loop_ports(loop_info->entry_points);
        validate_loop_ports(loop_info->exit_points);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
