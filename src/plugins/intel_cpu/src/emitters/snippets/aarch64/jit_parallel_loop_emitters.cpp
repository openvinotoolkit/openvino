// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_parallel_loop_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h>

#include <algorithm>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/aarch64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/parallel_loop.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64 {
namespace {

bool has_xreg(const std::vector<Xbyak_aarch64::Reg>& regs, const Xbyak_aarch64::XReg& xreg) {
    return std::any_of(regs.begin(), regs.end(), [&xreg](const Xbyak_aarch64::Reg& reg) {
        return reg.isRReg() && reg.getIdx() == xreg.getIdx();
    });
}

bool has_xreg_idx(const std::vector<size_t>& regs, size_t xreg_idx) {
    return std::find(regs.begin(), regs.end(), xreg_idx) != regs.end();
}

std::vector<Xbyak_aarch64::Reg> get_parallel_section_callee_saved_regs() {
    std::vector<Xbyak_aarch64::Reg> regs;
    regs.reserve(20);
    for (int i = Operand::X19; i <= Operand::X30; ++i) {
        regs.emplace_back(XReg(i));
    }
    for (int i = 8; i <= 15; ++i) {
        regs.emplace_back(QReg(i));
    }
    return regs;
}

}  // namespace

jit_parallel_loop_begin_emitter::jit_parallel_loop_begin_emitter(jit_generator_t* h,
                                                                 cpu_isa_t isa,
                                                                 const ov::snippets::lowered::ExpressionPtr& expr,
                                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_emitter(h, isa),
      jit_loop_begin_base_emitter(h, isa, expr, true),
      jit_binary_call_emitter(h, isa, expr->get_live_regs()),
      m_loop_preamble_label(std::make_shared<Xbyak_aarch64::Label>()),
      m_parallel_section_reg_spiller(std::make_shared<EmitABIRegSpills>(h)) {
    const auto loop_end_expr = get_loop_end_expr(expr);
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(loop_end_expr->get_node());
    m_loop_args = ov::intel_cpu::utils::compose_loop_args(loop_end);
    m_is_dynamic = loop_end->has_dynamic_params();

    const auto& loop_end_input_regs = loop_end_expr->get_reg_info().first;
    OV_CPU_JIT_EMITTER_ASSERT(!loop_end_input_regs.empty(), "Invalid LoopEnd reg info");
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

void jit_parallel_loop_begin_emitter::emit_parallel_executor_call(std::vector<Xbyak_aarch64::Reg>& used_regs) const {
    init_binary_call_regs(3, m_mem_ptr_regs_idxs);

    EmitABIRegSpills binary_call_reg_spiller(h);
    binary_call_reg_spiller.preamble(get_regs_to_spill_except_mem_ptr_regs());

    const auto call_args_size = sizeof(typename ParallelLoopExecutor::call_args);
    const auto mem_ptrs_size = m_mem_ptr_regs_idxs.size() * sizeof(uintptr_t*);
    const auto reserved_stack_size =
        ov::intel_cpu::rnd_up(call_args_size + mem_ptrs_size, static_cast<size_t>(jit_emitter::sp_alignment));
    if (reserved_stack_size > 0) {
        h->sub(h->sp, h->sp, reserved_stack_size);
    }

    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        h->str(XReg(static_cast<int>(m_mem_ptr_regs_idxs[i])),
               ptr(h->sp, static_cast<int32_t>(call_args_size + i * sizeof(uintptr_t*))));
    }

    const auto& aux_reg = get_call_address_reg();
    used_regs = binary_call_reg_spiller.get_spilled_regs();
    const auto memory_buf_size = EmitABIRegSpills::compute_memory_buffer_size(used_regs);
    m_common_registers_buffer.resize(memory_buf_size);
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(m_common_registers_buffer.data()));
    EmitABIRegSpills::store_regs_to_memory(h, used_regs, aux_reg);

    if (m_is_dynamic) {
        h->ldr(aux_reg, ptr(XReg(Operand::X0), static_cast<int32_t>(GET_OFF(loop_args))));
        h->add_imm(aux_reg, aux_reg, m_loop_id_offset, h->X_TMP_1);
    } else {
        h->mov(aux_reg, reinterpret_cast<uintptr_t>(&m_loop_args));
    }
    h->str(aux_reg, ptr(h->sp, static_cast<int32_t>(GET_OFF_PARALLEL_LOOP_ARGS(loop_args))));

    h->adr(aux_reg, *m_loop_preamble_label);
    h->str(aux_reg, ptr(h->sp, static_cast<int32_t>(GET_OFF_PARALLEL_LOOP_ARGS(preamble_ptr))));

    h->add(aux_reg, h->sp, call_args_size);
    h->str(aux_reg, ptr(h->sp, static_cast<int32_t>(GET_OFF_PARALLEL_LOOP_ARGS(mem_ptrs))));

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(ParallelLoopExecutor::execute));
    h->mov(XReg(Operand::X0), reinterpret_cast<uintptr_t>(m_executor.get()));
    h->mov(XReg(Operand::X1), h->sp);
    h->blr(aux_reg);

    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        h->ldr(XReg(static_cast<int>(m_mem_ptr_regs_idxs[i])),
               ptr(h->sp, static_cast<int32_t>(call_args_size + i * sizeof(uintptr_t*))));
    }
    if (reserved_stack_size > 0) {
        h->add(h->sp, h->sp, reserved_stack_size);
    }
    binary_call_reg_spiller.postamble();

    h->b(*m_loop_end_label);
}

void jit_parallel_loop_begin_emitter::emit_parallel_region_initialization(
    const std::vector<Xbyak_aarch64::Reg>& regs_to_restore,
    size_t work_amount_reg_idx) const {
    h->L(*m_loop_preamble_label);

    m_parallel_section_reg_spiller->preamble(get_parallel_section_callee_saved_regs());

    const auto reg_work_amount = XReg(static_cast<int>(work_amount_reg_idx));
    OV_CPU_JIT_EMITTER_ASSERT(!has_xreg_idx(m_mem_ptr_regs_idxs, work_amount_reg_idx),
                              "Unexpected collision: the same reg is allocated for work_amount and memory pointer");
    h->mov(reg_work_amount, XReg(Operand::X0));

    std::optional<size_t> abi_param2_collision_index;
    for (size_t i = 0; i < m_mem_ptr_regs_idxs.size(); ++i) {
        const auto reg_to_restore = XReg(static_cast<int>(m_mem_ptr_regs_idxs[i]));
        OPENVINO_ASSERT(!has_xreg(regs_to_restore, reg_to_restore),
                        "Expected to restore all registers except for m_mem_ptr_regs_idxs");
        if (reg_to_restore.getIdx() == Operand::X1) {
            abi_param2_collision_index = i;
        } else {
            h->ldr(reg_to_restore, ptr(XReg(Operand::X1), static_cast<int32_t>(i * sizeof(uintptr_t*))));
        }
    }
    if (const auto collision_idx = abi_param2_collision_index) {
        h->ldr(XReg(Operand::X1),
               ptr(XReg(Operand::X1), static_cast<int32_t>(collision_idx.value() * sizeof(uintptr_t*))));
    }

    const auto& aux_reg = get_call_address_reg();
    OV_CPU_JIT_EMITTER_ASSERT(!has_xreg(regs_to_restore, aux_reg),
                              "aux_reg mustn't coincide with any reg from regs_to_restore");
    OV_CPU_JIT_EMITTER_ASSERT(static_cast<size_t>(aux_reg.getIdx()) != work_amount_reg_idx,
                              "aux_reg mustn't coincide with the work amount reg");
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(m_common_registers_buffer.data()));
    EmitABIRegSpills::load_regs_from_memory(h,
                                            regs_to_restore,
                                            aux_reg,
                                            static_cast<uint32_t>(m_common_registers_buffer.size()));

    h->L(*m_loop_begin_label);
}

void jit_parallel_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                                const std::vector<size_t>& out) const {
    const bool is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(m_loop_args.m_work_amount);
    emit_loop_begin_work_amount_check(out, is_work_amount_dynamic, m_loop_args.m_work_amount);

    std::vector<Xbyak_aarch64::Reg> regs_to_restore;
    emit_parallel_executor_call(regs_to_restore);
    emit_parallel_region_initialization(regs_to_restore, out.back());
}

jit_parallel_loop_end_emitter::jit_parallel_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                                             dnnl::impl::cpu::aarch64::cpu_isa_t isa,
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
    emit_loop_end_impl(in, false);
    m_parallel_section_reg_spiller->postamble();
    h->ret();
    h->L(*m_loop_end_label);
}

}  // namespace ov::intel_cpu::aarch64
