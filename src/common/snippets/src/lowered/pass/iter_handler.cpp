// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/iter_handler.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
UpdateMemoryAccessCounts::UpdateMemoryAccessCounts(size_t count) : RangedPass(), m_count(count) {}

bool UpdateMemoryAccessCounts::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    bool status = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        // Skip inner Loops
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(expr_it->get()->get_node());
        if (loop_begin) {
            expr_it = linear_ir.find(expr_it, end, linear_ir.get_expr_by_node(loop_begin->get_loop_end()));
            if (expr_it == end)
                return status;
            continue;
        }

        const auto& node = expr_it->get()->get_node();
        if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(node)) {
            for (const auto p : memory_access->get_memory_access_input_ports()) {
                const auto port = p.first;
                if (memory_access->get_input_count(port) > 1) {
                    memory_access->set_input_count(m_count, port);
                }
            }
            for (const auto p : memory_access->get_memory_access_output_ports()) {
                const auto port = p.first;
                if (memory_access->get_output_count(port) > 1) {
                    memory_access->set_output_count(m_count, port);
                }
            }
            status = true;
        }
    }
    return status;
}

std::shared_ptr<pass::PassBase> UpdateMemoryAccessCounts::merge(const std::shared_ptr<pass::PassBase>& other) {
    if (!other)
        return shared_from_this();
    const auto casted_pass = ov::as_type_ptr<UpdateMemoryAccessCounts>(other);
    size_t merged_count;
    if (!casted_pass || !ov::snippets::utils::merge_dynamic_dim(merged_count, m_count, casted_pass->m_count))
        return nullptr;
    return std::make_shared<UpdateMemoryAccessCounts>(merged_count);
}

SetFillOffset::SetFillOffset(size_t offset) : RangedPass(), m_offset(offset) {}

bool SetFillOffset::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& node = expr_it->get()->get_node();
        if (const auto fill = ov::as_type_ptr<ov::snippets::op::Fill>(node)) {
            fill->set_offset(m_offset);
        }
    }
    return true;
}

std::shared_ptr<pass::PassBase> SetFillOffset::merge(const std::shared_ptr<pass::PassBase>& other) {
    if (!other)
        return shared_from_this();
    const auto casted_pass = ov::as_type_ptr<SetFillOffset>(other);
    size_t merged_offset;
    if (!casted_pass || !ov::snippets::utils::merge_dynamic_dim(merged_offset, m_offset, casted_pass->m_offset))
        return nullptr;
    return std::make_shared<SetFillOffset>(merged_offset);
}

bool SetLoopIncrementOne::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(end->get()->get_node());
    OPENVINO_ASSERT(loop_end, "SetLoopIncrementOne expected LoopEnd node in iterator `end`.");
    const auto& loop_info = linear_ir.get_loop_manager()->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_end->get_id());
    loop_info->set_increment(1);
    loop_end->set_increment(1);
    return true;
}

std::shared_ptr<snippets::lowered::pass::PassBase> SetLoopIncrementOne::merge(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    return !other || ov::is_type<SetLoopIncrementOne>(other) ? std::make_shared<SetLoopIncrementOne>() : nullptr;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

