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

jit_parallel_loop_base_emitter::jit_parallel_loop_base_emitter(jit_generator_t* h,
                                                               cpu_isa_t isa,
                                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    std::shared_ptr<snippets::op::LoopEnd> loop_end;
    std::vector<snippets::Reg> loop_end_input_regs;
    if (auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node())) {
        loop_end = loop_begin->get_loop_end();
        // todo: A long-term solution would be to introduce loop expressions so LoopBeginExpr->get_loop_end() would
        // return LoopEndExpr directly
        const auto& consumers = expr->get_output_port_connector(expr->get_output_count() - 1)->get_consumers();
        OV_CPU_JIT_EMITTER_ASSERT(!consumers.empty(), "LoopBegin must have LoopEnd as the last consumer");
        const auto& loop_end_expr = consumers.rbegin()->get_expr();
        OV_CPU_JIT_EMITTER_ASSERT(loop_end_expr && loop_end_expr->get_node() == loop_end,
                                  "Failed to find valid LoopEnd expression");
        loop_end_input_regs = loop_end_expr->get_reg_info().first;
    } else {
        loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
        OV_CPU_JIT_EMITTER_ASSERT(loop_end, "loop end node is expected");
        loop_end_input_regs = expr->get_reg_info().first;
    }
    OV_CPU_JIT_EMITTER_ASSERT(loop_end, "Failed to initialize LoopEnd in jit_parallel_loop_base_emitter");
    io_num = loop_end->get_input_num() + loop_end->get_output_num();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id_offset = loop_end->get_id() * sizeof(jit_snippets_call_args::loop_args_t);

    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& fin_offsets = loop_end->get_finalization_offsets();
    is_dynamic = snippets::utils::is_dynamic_value(loop_end->get_work_amount()) ||
                 std::any_of(ptr_increments.begin(),
                             ptr_increments.end(),
                             [](const auto& x) {
                                 return snippets::utils::is_dynamic_value(x);
                             }) ||
                 std::any_of(fin_offsets.begin(), fin_offsets.end(), [](const auto& x) {
                     return snippets::utils::is_dynamic_value(x);
                 });

    const auto int_work_amount = ov::snippets::utils::is_dynamic_value(loop_end->get_work_amount())
                                     ? ov::snippets::utils::get_dynamic_value<int64_t>()
                                     : static_cast<int64_t>(loop_end->get_work_amount());
    // We initialize loop_args with the known values already at compilation stage.
    // In case of static loop, only these loop_args will be used.
    // In case of dynamic loop, loop_args from jit_snippets_call_args will be used for dynamic values.
    loop_args = jit_snippets_call_args::loop_args_t(int_work_amount, ptr_increments, fin_offsets);
    const auto& data_sizes = loop_end->get_element_type_sizes();
    for (int64_t i = 0; i < loop_args.m_num_data_ptrs; ++i) {
        // Note: behavior is aligned with runtime configurator:
        // data_sizes and increment are already taken into account in the offsets
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_ptr_increments[i])) {
            loop_args.m_ptr_increments[i] *= (wa_increment * data_sizes[i]);
        }
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_finalization_offsets[i])) {
            loop_args.m_finalization_offsets[i] *= data_sizes[i];
        }
    }

    OV_CPU_JIT_EMITTER_ASSERT(!loop_end_input_regs.empty(), "Invalid LoopEnd reg info");
    work_amount_reg_idx = loop_end_input_regs.back().idx;
    loop_end_input_regs.pop_back();
    mem_ptr_regs_idxs.reserve(loop_end_input_regs.size());
    for (const auto& r : loop_end_input_regs) {
        if (r.type == snippets::RegType::gpr) {
            mem_ptr_regs_idxs.emplace_back(r.idx);
        }
    }
}

jit_parallel_loop_begin_emitter::jit_parallel_loop_begin_emitter(jit_generator_t* h,
                                                                 cpu_isa_t isa,
                                                                 const ov::snippets::lowered::ExpressionPtr& expr,
                                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{new Label()},
      loop_preamble_label{new Label()},
      loop_end_label(nullptr),
      m_parallel_section_reg_spiller(std::make_shared<EmitABIRegSpills>(h)) {
    auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin && loop_begin->get_is_parallel(), "expects parallel LoopBegin expression");
    m_executor = kernel_table->register_kernel<ParallelLoopExecutor>(expr, ParallelLoopConfig(wa_increment));
}

void jit_parallel_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in,
                                                         const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs: expected 1 got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.back() == work_amount_reg_idx,
                              "Invalid out reg: expected ",
                              work_amount_reg_idx,
                              " got ",
                              out.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr && loop_end_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
}

void jit_parallel_loop_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                                     const std::vector<size_t>& out,
                                                     const std::vector<size_t>& pool_vec_idxs,
                                                     const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs);
}

std::set<snippets::Reg> jit_parallel_loop_begin_emitter::get_regs_to_spill_except_mem_ptr_regs() const {
    auto regs_to_spill = get_regs_to_spill();
    for (auto i : mem_ptr_regs_idxs) {
        regs_to_spill.erase({snippets::RegType::gpr, i});
    }
    return regs_to_spill;
}

void jit_parallel_loop_begin_emitter::emit_parallel_executor_call(std::vector<Xbyak::Reg>& used_regs) const {
    init_binary_call_regs(3, mem_ptr_regs_idxs);
    // Note: mem_ptr_regs_idxs regs are not spilled, since they are handled manually:
    // before the parallel region call, they are passed via stack as ParallelLoopExecutor::execute parameter,
    // and restored after it with applied finalization offsets.
    EmitABIRegSpills binary_call_reg_spiller(h);
    binary_call_reg_spiller.preamble(get_regs_to_spill_except_mem_ptr_regs());

    const auto call_args_size = sizeof(typename ParallelLoopExecutor::call_args);
    const auto mem_ptrs_size = mem_ptr_regs_idxs.size() * sizeof(uintptr_t*);
    const auto reserved_stack_size = call_args_size + mem_ptrs_size;
    // Spill before parallel for => we'll need them to update data ptrs afterwards
    h->sub(h->rsp, reserved_stack_size);

    auto push_reg_on_stack = [&](Reg64 reg, size_t offset) {
        utils::push_ptr_with_static_offset_on_stack(h, offset, reg);
    };
    for (size_t i = 0; i < mem_ptr_regs_idxs.size(); ++i) {
        push_reg_on_stack(Reg64(mem_ptr_regs_idxs[i]), call_args_size + i * sizeof(uintptr_t*));
    }

    const auto& aux_reg = get_call_address_reg();
    used_regs = binary_call_reg_spiller.get_spilled_regs();
    const auto memory_buf_size = EmitABIRegSpills::compute_memory_buffer_size(used_regs);
    m_common_registers_buffer.resize(memory_buf_size);
    // Note: parallel loop emitter needs to spill registers to common buffer
    // to propagate register states in each thread (stack of the main thread can't be used for such purpose).
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(m_common_registers_buffer.data()));
    EmitABIRegSpills::store_regs_to_memory(h, used_regs, aux_reg);

    if (is_dynamic) {
        h->mov(aux_reg, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->lea(aux_reg, h->ptr[aux_reg + loop_id_offset]);
        push_reg_on_stack(aux_reg, GET_OFF_PARALLEL_LOOP_ARGS(loop_args));
    } else {
        h->mov(aux_reg, reinterpret_cast<uintptr_t>(&loop_args));
        push_reg_on_stack(aux_reg, GET_OFF_PARALLEL_LOOP_ARGS(loop_args));
    }
    h->mov(aux_reg, *loop_preamble_label);
    push_reg_on_stack(aux_reg, GET_OFF_PARALLEL_LOOP_ARGS(preamble_ptr));
    h->lea(aux_reg, h->qword[h->rsp + call_args_size]);
    push_reg_on_stack(aux_reg, GET_OFF_PARALLEL_LOOP_ARGS(mem_ptrs));

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(ParallelLoopExecutor::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_executor.get()));
    h->mov(abi_param2, h->rsp);

    binary_call_reg_spiller.rsp_align(get_callee_saved_reg().getIdx());
    // Note: we will return from this call only when the parallel region is finished (return from
    // jit_parallel_loop_end_emitter)
    h->call(aux_reg);
    binary_call_reg_spiller.rsp_restore();

    // Restore data ptrs with applied finalization offsets
    for (size_t i = 0; i < mem_ptr_regs_idxs.size(); ++i) {
        h->mov(Reg64(mem_ptr_regs_idxs[i]), h->qword[h->rsp + call_args_size + i * sizeof(uintptr_t*)]);
    }
    h->add(h->rsp, reserved_stack_size);
    binary_call_reg_spiller.postamble();

    h->jmp(*loop_end_label, CodeGenerator::T_NEAR);
}

void jit_parallel_loop_begin_emitter::emit_parallel_region_initialization(
    const std::vector<Xbyak::Reg>& regs_to_restore) const {
    h->L(*loop_preamble_label);

    std::set<snippets::Reg> loop_premble_spill;
    // todo: we don't have to spill all calle+saved_regs, only the ones that will be used in the loop's body
    for (auto i : get_callee_saved_reg_idxs()) {
        loop_premble_spill.insert({snippets::RegType::gpr, i});
    }
    m_parallel_section_reg_spiller->preamble(loop_premble_spill);

    // Note: some of mem_ptr_regs_idxs might coincide with abi_param_2.
    // abi_param_1 is always reserved for runtime parameters storage,
    // so it can't coincide with any of mem_ptr_regs_idxs.
    std::optional<size_t> abi_param2_collision_index;
    for (size_t i = 0; i < mem_ptr_regs_idxs.size(); ++i) {
        auto reg_to_restore = Reg64(mem_ptr_regs_idxs[i]);
        OPENVINO_ASSERT(
            std::find(regs_to_restore.begin(), regs_to_restore.end(), reg_to_restore) == regs_to_restore.end(),
            "Expected to restore all registers except for mem_ptr_regs_idxs");
        if (reg_to_restore == abi_param2) {
            abi_param2_collision_index = i;
        } else {
            h->mov(reg_to_restore, h->ptr[abi_param2 + i * sizeof(uintptr_t*)]);
        }
    }
    if (abi_param2_collision_index.has_value()) {
        const auto collision_idx = abi_param2_collision_index.value();
        h->mov(abi_param2, h->ptr[abi_param2 + collision_idx * sizeof(uintptr_t*)]);
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

    h->L(*loop_begin_label);
}

void jit_parallel_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                                [[maybe_unused]] const std::vector<size_t>& out) const {
    // TODO: reuse loop_begin_emitter code
    auto reg_work_amount = Reg64(static_cast<int>(out.back()));
    if (ov::snippets::utils::is_dynamic_value(loop_args.m_work_amount)) {
        utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, out);
        Reg64 reg_loop_args_ptr = gpr_holder.get_reg();
        h->mov(reg_loop_args_ptr, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + loop_id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);
    } else {
        h->mov(reg_work_amount, loop_args.m_work_amount);
    }
    // if wa < increment, skip the loop
    // Note : If the loop should be evaluated once and increment is dynamic,
    //        we should manually set `increment = 1` to compare the dynamic work amount
    //        with `1` at least before loop execution
    //        (work amount can be zero and we should skip this loop even `evaluate_once = 1`)
    auto increment = evaluate_once && snippets::utils::is_dynamic_value(wa_increment) ? 1 : wa_increment;
    h->cmp(reg_work_amount, increment);
    h->jl(*loop_end_label, Xbyak::CodeGenerator::T_NEAR);

    std::vector<Xbyak::Reg> regs_to_restore;
    emit_parallel_executor_call(regs_to_restore);
    // Note: parallel region starts here. The only legal entry point is from ParallelLoopExecutor::execute(...)
    emit_parallel_region_initialization(regs_to_restore);
}

jit_parallel_loop_end_emitter::jit_parallel_loop_end_emitter(jit_generator_t* h,
                                                             cpu_isa_t isa,
                                                             const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{nullptr},
      loop_end_label{new Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end && loop_end->get_is_parallel(), "expected parallel LoopEnd expr");
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter =
        std::dynamic_pointer_cast<jit_parallel_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
    m_parallel_section_reg_spiller = loop_begin_emitter->get_parallel_section_reg_spiller();
}

ov::snippets::lowered::ExpressionPtr jit_parallel_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(begin_expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin && loop_begin->get_is_parallel(),
                              "LoopEnd expression must have the last port connector to parallel LoopBegin");
    return begin_expr;
}

void jit_parallel_loop_end_emitter::validate_arguments(const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Invalid number of out arguments: expected 0 got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_num + 1,
                              "Invalid number of in arguments: expected ",
                              io_num + 1,
                              " got ",
                              in.size());
    OV_CPU_JIT_EMITTER_ASSERT(is_incremented.size() == io_num,
                              "Invalid is_incremented size: expected ",
                              io_num,
                              " got ",
                              is_incremented.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
    OV_CPU_JIT_EMITTER_ASSERT(m_parallel_section_reg_spiller != nullptr,
                              "parallel section reg spiller is not initialized");
}

void jit_parallel_loop_end_emitter::emit_code_impl(const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out,
                                                   const std::vector<size_t>& pool_vec_idxs,
                                                   const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_parallel_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                              [[maybe_unused]] const std::vector<size_t>& out) const {
    if (!evaluate_once) {
        Reg64 reg_increments;
        auto add_increments = [&]() {
            for (size_t idx = 0; idx < mem_ptr_regs_idxs.size(); idx++) {
                const auto& increment = loop_args.m_ptr_increments[idx];
                if (is_incremented[idx] && increment != 0) {
                    if (ov::snippets::utils::is_dynamic_value(increment)) {
                        OV_CPU_JIT_EMITTER_ASSERT(is_dynamic, "Loop argument structure cannot be pushed to aux GPR");
                        h->add(Reg64(static_cast<int>(mem_ptr_regs_idxs[idx])),
                               h->ptr[reg_increments + idx * sizeof(int64_t)]);
                    } else {
                        h->add(Reg64(static_cast<int>(mem_ptr_regs_idxs[idx])), increment);
                    }
                }
            }
        };

        if (is_dynamic) {
            utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, in);
            reg_increments = gpr_holder.get_reg();
            h->mov(reg_increments, wa_increment);
            h->mov(reg_increments, h->ptr[abi_param1 + GET_OFF(loop_args)]);
            h->mov(reg_increments, h->ptr[reg_increments + loop_id_offset + GET_OFF_LOOP_ARGS(m_ptr_increments)]);
            add_increments();
        } else {
            add_increments();
        }

        auto reg_work_amount = Reg64(in.back());
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(*loop_begin_label, CodeGenerator::T_NEAR);
    }
    m_parallel_section_reg_spiller->postamble();
    // Note: parallel region ends here:
    h->ret();
    h->L(*loop_end_label);
    // Note: finalization offsets are applied in ParallelLoopExecutor::execute after parallel region is ended
}

}  // namespace ov::intel_cpu
