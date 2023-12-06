// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/iter_handler.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

SetSingleIterationWithWorkAmount::SetSingleIterationWithWorkAmount(size_t work_amount)
    : SubgraphPass(),
      m_work_amount(work_amount) {}

bool SetSingleIterationWithWorkAmount::run(const LinearIR& linear_ir,
                                           LinearIR::constExprIt begin,
                                           LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    if (loop_end->get_work_amount() == m_work_amount && loop_end->get_increment() == m_work_amount)
        return false;
    loop_end->set_work_amount(m_work_amount);
    loop_end->set_increment(m_work_amount);
    loop_info->set_work_amount(m_work_amount);
    loop_info->set_increment(m_work_amount);
    return true;
}

UpdateMemoryAccessOps::UpdateMemoryAccessOps(size_t count) : SubgraphPass(), m_count(count) {}

bool UpdateMemoryAccessOps::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = std::next(begin); expr_it != end; expr_it++) {
        // Skip inner Loops
        const auto loop_begin = ov::as_type_ptr<op::LoopBegin>(expr_it->get()->get_node());
        if (loop_begin) {
            expr_it = linear_ir.find(expr_it, end, linear_ir.get_expr_by_node(loop_begin->get_loop_end()));
            continue;
        }

        const auto& node = expr_it->get()->get_node();
        if (const auto memory_access = ov::as_type_ptr<ov::snippets::op::MemoryAccess>(node)) {
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
        }
    }
    return true;
}

ReduceWorkAmount::ReduceWorkAmount(size_t reduce_value) : SubgraphPass(), m_reduce_value(reduce_value) {}

bool ReduceWorkAmount::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    const auto work_amount = loop_end->get_work_amount();
    const auto new_work_amount = work_amount - m_reduce_value;
    loop_end->set_work_amount(new_work_amount);

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    loop_info->set_work_amount(new_work_amount);
    return true;
}

bool ZeroFinalizationOffsets::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    loop_end->set_finalization_offsets(std::vector<int64_t>(loop_end->get_finalization_offsets().size(), 0));
    return true;
}

SetFillOffset::SetFillOffset(size_t offset) : SubgraphPass(), m_offset(offset) {}

bool SetFillOffset::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = std::next(begin); expr_it != end; expr_it++) {
        const auto& node = expr_it->get()->get_node();
        if (const auto fill = ov::as_type_ptr<ov::snippets::op::Fill>(node)) {
            fill->set_offset(m_offset);
        }
    }
    return true;
}

TransformInnerSplitLoop::TransformInnerSplitLoop(size_t tail_size) : SubgraphPass(), m_tail_size(tail_size) {}

bool TransformInnerSplitLoop::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    const auto current_dim_idx = loop_info->get_dim_idx();
    OPENVINO_ASSERT(current_dim_idx != LinearIR::LoopManager::LoopInfo::UNDEFINED_DIM_IDX,
                    "Outer splitted loop unexpectedly iterates by several dimension indices");

    bool modified = false;
    for (auto it = std::next(begin); it != end; ++it) {
        const auto& expr = *it;
        const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!inner_loop_end)
            continue;
        const auto inner_loop_info = loop_manager->get_loop_info(inner_loop_end->get_id());
        const auto inner_dim_idx = inner_loop_info->get_dim_idx();
        if (inner_dim_idx != current_dim_idx)
            continue;
        const auto inner_loop_begin = inner_loop_end->get_loop_begin();
        const auto inner_tail_work_amount = static_cast<int64_t>(inner_loop_end->get_work_amount());
        const auto inner_tail_increment = inner_loop_end->get_increment();
        auto inner_finalization_offsets = inner_loop_end->get_finalization_offsets();
        for (auto& offset : inner_finalization_offsets) {
            offset = offset / inner_tail_work_amount * static_cast<int64_t>(m_tail_size);
        }
        inner_loop_end->set_work_amount(m_tail_size);
        // TODO: if the new m_tail_size increment is set, all last iter handlers must be updated with new tail value
        // We can also don't split loops in case if inner loop has increment not equal to 1
        inner_loop_end->set_increment(std::min(inner_tail_increment, m_tail_size));
        inner_loop_end->set_finalization_offsets(inner_finalization_offsets);
        const auto inner_loop_begin_it = std::find(begin, it, linear_ir.get_expr_by_node(inner_loop_begin));
        const auto inner_loop_end_it = std::next(end);
        OPENVINO_ASSERT(inner_loop_begin_it != it, "LoopBegin has not been found!");
        const auto& last_iter_handlers = inner_loop_info->handlers[LinearIR::LoopManager::LoopInfo::LAST_ITER];
        last_iter_handlers.run(linear_ir, inner_loop_begin_it, inner_loop_end_it);
        modified = true;
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

