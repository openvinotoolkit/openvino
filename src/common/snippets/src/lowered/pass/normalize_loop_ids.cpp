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

bool NormalizeLoopIDs::run(lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::NormalizeLoopIDs");

    // [ original Loop ID -> new normalized and sorted ]
    std::map<size_t, size_t> loop_id_map;
    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node)) {
            const auto old_id = loop_end->get_id();
            if (loop_id_map.count(old_id) == 0) {
                const auto new_id = loop_id_map.size();
                loop_id_map[old_id] = new_id;
                continue;
            }
            OPENVINO_ASSERT(m_has_specific_loops, "NormalizeLoopIDs failed: LinearIR contains solid loops with the same IDs!");
        }
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    return loop_manager->blend(linear_ir, loop_id_map);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
