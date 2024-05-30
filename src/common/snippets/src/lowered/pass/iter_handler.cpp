// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/iter_handler.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

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
    const auto merged_pass = std::make_shared<UpdateMemoryAccessCounts>(m_count);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<UpdateMemoryAccessCounts>(other);
    if (!casted_pass || m_count != casted_pass->m_count)
        return nullptr;
    return merged_pass;
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
    const auto merged_pass = std::make_shared<SetFillOffset>(m_offset);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<SetFillOffset>(other);
    if (!casted_pass || m_offset != casted_pass->m_offset)
        return nullptr;
    return merged_pass;
}

TransformInnerSplitLoop::TransformInnerSplitLoop(size_t tail_size) : RangedPass(), m_tail_size(tail_size) {}

bool TransformInnerSplitLoop::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    OPENVINO_ASSERT(loop_end, "the last operation in range must be LoopEnd");

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    const auto current_dim_idx = loop_info->get_dim_idx();
    OPENVINO_ASSERT(current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX,
                    "Outer splitted loop unexpectedly iterates by several dimension indices");

    bool modified = false;
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!inner_loop_end)
            continue;
        // There is already ExpandedLoopInfo
        const auto inner_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(inner_loop_end->get_id());
        const auto inner_dim_idx = inner_loop_info->get_dim_idx();
        if (inner_dim_idx != current_dim_idx)
            continue;
        // TODO [141735] : At the moment Splitted loops are not supported in dynamic case
        OPENVINO_ASSERT(!inner_loop_end->has_dynamic_params(), "inner loop must be static in TransformInnerSplitLoop");
        const auto inner_loop_begin = inner_loop_end->get_loop_begin();
        const auto inner_loop_work_amount = static_cast<int64_t>(inner_loop_end->get_work_amount());
        const auto inner_loop_increment = inner_loop_end->get_increment();
        auto inner_finalization_offsets = inner_loop_end->get_finalization_offsets();
        for (auto& offset : inner_finalization_offsets) {
            offset = offset / inner_loop_work_amount * static_cast<int64_t>(m_tail_size);
        }
        inner_loop_end->set_work_amount(m_tail_size);
        // TODO: if m_tail_size more than inner loop increment,
        // handlers of the inner loop must be reset with new tail size
        inner_loop_end->set_increment(std::min(inner_loop_increment, m_tail_size));
        inner_loop_end->set_finalization_offsets(inner_finalization_offsets);
        const auto inner_loop_begin_it = std::find(begin, it, linear_ir.get_expr_by_node(inner_loop_begin));
        const auto inner_loop_end_it = std::next(it);
        OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
        const auto& last_iter_handlers = inner_loop_info->get_unified_loop_info()->get_handlers().get_passes<SpecificLoopIterType::LAST_ITER>();
        last_iter_handlers.run(linear_ir, std::next(inner_loop_begin_it), inner_loop_end_it);
        modified = true;
    }
    return modified;
}

std::shared_ptr<pass::PassBase> TransformInnerSplitLoop::merge(const std::shared_ptr<pass::PassBase>& other) {
    const auto merged_pass = std::make_shared<TransformInnerSplitLoop>(m_tail_size);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<TransformInnerSplitLoop>(other);
    if (!casted_pass || m_tail_size != casted_pass->m_tail_size)
        return nullptr;
    return merged_pass;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

