// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_parallel_loop_emitters.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/parallel_loop.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "jit_loop_base_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_parallel_loop_begin_emitter::jit_parallel_loop_begin_emitter(jit_generator_t* h,
                                                                 cpu_isa_t isa,
                                                                 const ov::snippets::lowered::ExpressionPtr& expr,
                                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_emitter(h, isa),
      jit_loop_begin_base_emitter(h, isa, expr, true),
      jit_binary_call_emitter(h, isa, expr->get_live_regs()),
      m_loop_preamble_label{std::make_shared<Xbyak::Label>()},
      m_parallel_section_reg_spiller(std::make_shared<EmitABIRegSpills>(h)) {
    const auto loop_end_expr = get_loop_end_expr(expr);
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(loop_end_expr->get_node());
    m_loop_args = jit_loop_end_base_emitter::compose_loop_args(loop_end);
    m_is_dynamic = loop_end->has_dynamic_params();

    const auto& loop_end_input_regs = loop_end_expr->get_reg_info().first;
    OV_CPU_JIT_EMITTER_ASSERT(!loop_end_input_regs.empty(), "Invalid LoopEnd reg info");

    // Note: work amount and memory ptrs regs must be stored explicitly,
    // since they are passed to LoopEnd node, not LoopBegin.
    // However, in case of parallel loops, control logic is implemented in loop begin emitter
    for (size_t i = 0; i < loop_end_input_regs.size() - 1; ++i) {
        const auto& r = loop_end_input_regs[i];
        if (r.type == snippets::RegType::gpr) {
            m_mem_ptr_regs_idxs.emplace_back(r.idx);
        }
    }
    m_executor = kernel_table->register_kernel<ParallelLoopExecutor>(expr, ParallelLoopConfig(m_wa_increment));
}

std::set<snippets::Reg> jit_parallel_loop_begin_emitter::get_regs_to_spill_except_mem_ptr_regs() const {
    auto regs_to_spill = get_regs_to_spill();
    for (auto i : m_mem_ptr_regs_idxs) {
        regs_to_spill.erase({snippets::RegType::gpr, i});
    }
    return regs_to_spill;
}

void jit_parallel_loop_begin_emitter::emit_parallel_executor_call(std::vector<Xbyak::Reg>& used_regs) const {
    init_binary_call_regs(3, m_mem_ptr_regs_idxs);
    // Note: m_mem_ptr_regs_idxs regs are not spilled, since they are handled manually:
    // before the parallel region call, they are passed via stack as ParallelLoopExecutor::execute parameter,
    // and restored after it with applied finalization offsets.
    EmitABIRegSpills binary_call_reg_spiller(h);
    binary_call_reg_spiller.preamble(get_regs_to_spill_except_mem_ptr_regs());

    const auto call_args_size = sizeof(typename ParallelLoopExecutor::call_args);
    const auto mem_ptrs_size = m_mem_ptr_regs_idxs.size() * sizeof(uintptr_t*);
    const auto reserved_stack_size = call_args_size + mem_ptrs_size;
    // Spill before parallel for => we'll need them to update data ptrs afterwards
    h->sub(h->rsp, reserved_stack_size);

    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        utils::push_ptr_with_static_offset_on_stack(h,
                                                    call_args_size + i * sizeof(uintptr_t*),
                                                    Reg64(m_mem_ptr_regs_idxs[i]));
    }

    const auto& aux_reg = get_call_address_reg();
    used_regs = binary_call_reg_spiller.get_spilled_regs();
    const auto memory_buf_size = EmitABIRegSpills::compute_memory_buffer_size(used_regs);
    m_common_registers_buffer.resize(memory_buf_size);
    // Note: parallel loop emitter needs to spill registers to common buffer
    // to propagate register states in each thread (stack of the main thread can't be used for such purpose).
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(m_common_registers_buffer.data()));
    EmitABIRegSpills::store_regs_to_memory(h, used_regs, aux_reg);

    if (m_is_dynamic) {
        h->mov(aux_reg, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->lea(aux_reg, h->ptr[aux_reg + m_loop_id_offset]);
    } else {
        h->mov(aux_reg, reinterpret_cast<uintptr_t>(&m_loop_args));
    }
    utils::push_ptr_with_static_offset_on_stack(h, GET_OFF_PARALLEL_LOOP_ARGS(loop_args), aux_reg);
    h->mov(aux_reg, *m_loop_preamble_label);
    utils::push_ptr_with_static_offset_on_stack(h, GET_OFF_PARALLEL_LOOP_ARGS(preamble_ptr), aux_reg);
    h->lea(aux_reg, h->qword[h->rsp + call_args_size]);
    utils::push_ptr_with_static_offset_on_stack(h, GET_OFF_PARALLEL_LOOP_ARGS(mem_ptrs), aux_reg);

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(ParallelLoopExecutor::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_executor.get()));
    h->mov(abi_param2, h->rsp);

    binary_call_reg_spiller.rsp_align(get_callee_saved_reg().getIdx());
    // Note: we will return from this call only when the parallel region is finished
    // (h->ret() from jit_parallel_loop_end_emitter)
    h->call(aux_reg);
    binary_call_reg_spiller.rsp_restore();

    // Restore data ptrs with applied finalization offsets
    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        h->mov(Reg64(m_mem_ptr_regs_idxs[i]), h->qword[h->rsp + call_args_size + i * sizeof(uintptr_t*)]);
    }
    h->add(h->rsp, reserved_stack_size);
    binary_call_reg_spiller.postamble();

    h->jmp(*m_loop_end_label, CodeGenerator::T_NEAR);
}

void jit_parallel_loop_begin_emitter::emit_parallel_region_initialization(
    const std::vector<Xbyak::Reg>& regs_to_restore,
    size_t work_amount_reg_idx) const {
    h->L(*m_loop_preamble_label);

    std::set<snippets::Reg> loop_premble_spill;
    for (auto i : get_callee_saved_reg_idxs()) {
        loop_premble_spill.emplace(snippets::RegType::gpr, i);
    }
    m_parallel_section_reg_spiller->preamble(loop_premble_spill);

    // Note: some of m_mem_ptr_regs_idxs might coincide with abi_param_2.
    // abi_param_1 is always reserved for runtime parameters storage,
    // so it can't coincide with any of m_mem_ptr_regs_idxs.
    std::optional<size_t> abi_param2_collision_index;
    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        auto reg_to_restore = Reg64(m_mem_ptr_regs_idxs[i]);
        OPENVINO_ASSERT(
            std::find(regs_to_restore.begin(), regs_to_restore.end(), reg_to_restore) == regs_to_restore.end(),
            "Expected to restore all registers except for m_mem_ptr_regs_idxs");
        if (reg_to_restore == abi_param2) {
            abi_param2_collision_index = i;
        } else {
            h->mov(reg_to_restore, h->ptr[abi_param2 + i * sizeof(uintptr_t*)]);
        }
    }
    if (const auto collision_idx = abi_param2_collision_index) {
        h->mov(abi_param2, h->ptr[abi_param2 + collision_idx.value() * sizeof(uintptr_t*)]);
        OPENVINO_ASSERT(work_amount_reg_idx != static_cast<size_t>(abi_param2.getIdx()),
                        "Unexpected collision: the same reg is allocated for work_amount and memory pointer");
    }
    h->mov(Reg64(work_amount_reg_idx), abi_param1);

    const auto& aux_reg = get_call_address_reg();
    OV_CPU_JIT_EMITTER_ASSERT(
        std::find(regs_to_restore.begin(), regs_to_restore.end(), aux_reg) == regs_to_restore.end(),
        "aux_reg mustn't coincide with any reg from regs_to_restore");
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(m_common_registers_buffer.data()));
    EmitABIRegSpills::load_regs_from_memory(h, regs_to_restore, aux_reg, m_common_registers_buffer.size());

    h->L(*m_loop_begin_label);
}

void jit_parallel_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                                const std::vector<size_t>& out) const {
    const bool is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(m_loop_args.m_work_amount);
    emit_loop_begin_work_amount_check(out, is_work_amount_dynamic, m_loop_args.m_work_amount);

    std::vector<Xbyak::Reg> regs_to_restore;
    emit_parallel_executor_call(regs_to_restore);
    // Note: parallel region starts here. The only legal entry point is from ParallelLoopExecutor::execute(...)
    emit_parallel_region_initialization(regs_to_restore, out.back());
}

jit_parallel_loop_end_emitter::jit_parallel_loop_end_emitter(jit_generator_t* h,
                                                             cpu_isa_t isa,
                                                             const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_base_emitter(h, isa, expr, true) {
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter =
        std::dynamic_pointer_cast<jit_parallel_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_parallel_loop_begin_emitter");
    m_parallel_section_reg_spiller = loop_begin_emitter->get_parallel_section_reg_spiller();
}

void jit_parallel_loop_end_emitter::validate_arguments(const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) const {
    jit_loop_end_base_emitter::validate_arguments(in, out);
    OV_CPU_JIT_EMITTER_ASSERT(m_parallel_section_reg_spiller != nullptr,
                              "parallel section reg spiller is not initialized");
}

void jit_parallel_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                              [[maybe_unused]] const std::vector<size_t>& out) const {
    // Note: finalization offsets are applied in ParallelLoopExecutor::execute after parallel region is ended
    // so we don't apply them here
    emit_loop_end_impl(in, false);
    m_parallel_section_reg_spiller->postamble();
    // Note: parallel region ends here:
    h->ret();
    h->L(*m_loop_end_label);
}

}  // namespace ov::intel_cpu
