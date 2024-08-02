// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/normalize_loop_ids.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void NormalizeLoopIDs::update_linear_ir(lowered::LinearIR& linear_ir, const IDMapper& loop_id_map) {
    std::pair<std::vector<size_t>, std::vector<size_t>> previous_loop_ids;
    for (const auto& expr : linear_ir) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node())) {
            const auto current_id = loop_end->get_id();
            OPENVINO_ASSERT(loop_id_map.count(current_id) > 0, "ID of the LoopEnd has not been found in the map!");
            loop_end->set_id(loop_id_map.at(current_id));
        }

        auto expr_loop_ids = expr->get_loop_ids();
        if (expr_loop_ids.empty())
            continue;
        if (expr_loop_ids == previous_loop_ids.first) {
            expr->set_loop_ids(previous_loop_ids.second);
            continue;
        }

        previous_loop_ids.first = expr_loop_ids;
        std::for_each(expr_loop_ids.begin(), expr_loop_ids.end(), [&loop_id_map](size_t& id) {
            OPENVINO_ASSERT(loop_id_map.count(id) > 0, "Expression is marked by LoopID that has not been found in the map!");
            id = loop_id_map.at(id);
        });
        expr->set_loop_ids(expr_loop_ids);
        previous_loop_ids.second = expr_loop_ids;
    }
}

bool NormalizeLoopIDs::run(lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::NormalizeLoopIDs");

    // Firstly, we create map of the current and the target Loop IDs
    IDMapper loop_id_map;
    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node)) {
            const auto old_id = loop_end->get_id();
            if (loop_id_map.count(old_id) == 0) {
                const auto new_id = loop_id_map.size();
                loop_id_map[old_id] = new_id;
                continue;
            }
            OPENVINO_ASSERT(m_has_specific_loops, "NormalizeLoopIDs failed: LinearIR contains unified loops with the same IDs!");
        }
    }

    // Secondly, we blend `LoopInfo` in the LoopManager::m_map by new Loop IDs
    const auto updated = linear_ir.get_loop_manager()->reorder_identifiers(loop_id_map);
    if (!updated)
        return false;

    // Thirdly, we should update expressions in LinearIR
    update_linear_ir(linear_ir, loop_id_map);

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
