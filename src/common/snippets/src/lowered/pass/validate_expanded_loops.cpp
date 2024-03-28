// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_expanded_loops.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool ValidateExpandedLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateExpandedLoops")

    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> unique_loop_ids;
    for (const auto& expr : linear_ir) {
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node())) {
            const auto loop_id = loop_end->get_id();
            unique_loop_ids.insert(loop_id);
            OPENVINO_ASSERT(std::dynamic_pointer_cast<ExpandedLoopInfo>(loop_manager->get_loop_info(loop_id)),
                            "ValidateExpandedLoops expects only ExpandedLoopInfo in LoopManager");
        }
    }
    OPENVINO_ASSERT(unique_loop_ids.size() == loop_manager->get_map().size(),
                    "ValidateExpandedLoops failed: incompatible loopIDs of inserted LoopEnd expressions and LoopInfo in LoopManager");

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
