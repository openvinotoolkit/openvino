// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_parallel_loop_emitters.hpp"

#include <cpu/x64/xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/parallel_loop.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_parallel_loop_base_emitter::jit_parallel_loop_base_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                               dnnl::impl::cpu::x64::cpu_isa_t isa,
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
        loop_end_input_regs = expr->get_reg_info().first;
    }
    OV_CPU_JIT_EMITTER_ASSERT(loop_end, "Failed to initialize LoopEnd in jit_parallel_loop_base_emitter");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();

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

    // If the loop is static, we can initialize loop_args with the known values already at compilation stage.
    // In case of dynamic loop, loop_args from jit_snippets_call_args will be used
    if (!is_dynamic) {
        loop_args = jit_snippets_call_args::loop_args_t(loop_end->get_work_amount(), ptr_increments, fin_offsets);
        // Note: behavior is aligned with runtime configurator:
        // data_sizes and increment are already taken into account in the offsets
        const auto& data_sizes = loop_end->get_element_type_sizes();
        for (int64_t i = 0; i < loop_args.m_num_data_ptrs; ++i) {
            loop_args.m_ptr_increments[i] *= (wa_increment * data_sizes[i]);
            loop_args.m_finalization_offsets[i] *= data_sizes[i];
        }
    } else {
        OPENVINO_THROW("Dynamic parallel loop begin is not supported yet");
    }

    OV_CPU_JIT_EMITTER_ASSERT(!loop_end_input_regs.empty(), "Invalid LoopEnd reg info");
    internal_work_amount_reg_idx = loop_end_input_regs.rbegin()->idx;
    loop_end_input_regs.pop_back();
    mem_ptr_regs_idxs.reserve(loop_end_input_regs.size());
    for (const auto& r : loop_end_input_regs) {
        if (r.type == snippets::RegType::gpr) {
            mem_ptr_regs_idxs.emplace_back(r.idx);
        }
    }
}

jit_parallel_loop_begin_emitter::jit_parallel_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                                 const ov::snippets::lowered::ExpressionPtr& expr,
                                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{new Xbyak::Label()},
      loop_preamble_label{new Xbyak::Label()},
      loop_end_label(nullptr),
      m_loop_reg_spiller(std::make_shared<EmitABIRegSpills>(h)) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(expr->get_node()), "expects LoopBegin expression");
    m_executor = kernel_table->register_kernel<ParallelLoopExecutor>(expr, ParallelLoopConfig(wa_increment));
    // todo: we need to validate that the body expressions don't rely on any other registers except for loop port memory
    // pointers if they do, we need to spill them before the call and restore in the multithread section
}

void jit_parallel_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in,
                                                         const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs: expected 1 got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.back() == internal_work_amount_reg_idx,
                              "Invalid out reg: expected ",
                              internal_work_amount_reg_idx,
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
    // todo: validate that the parameters obtained in the base class correspond to in & out
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

void jit_parallel_loop_begin_emitter::emit_parallel_executor_call() const {
    init_binary_call_regs(3, mem_ptr_regs_idxs);
    EmitABIRegSpills spill(h);
    // Note: mem_ptr_regs_idxs regs are not spilled, since they are handled manually:
    // before the parallel region call, they are passed via stack as ParallelLoopExecutor::execute parameter,
    // and restored after it with applied finalization offsets.
    spill.preamble(get_regs_to_spill_except_mem_ptr_regs());

    const auto call_args_size = sizeof(typename ParallelLoopExecutor::call_args);
    const auto mem_ptrs_size = mem_ptr_regs_idxs.size() * sizeof(uintptr_t*);
    const auto reserved_stack_size = call_args_size + mem_ptrs_size;
    // Spill before parallel for => we'll need them to update data ptrs afterwards
    h->sub(h->rsp, reserved_stack_size);

    for (auto i : mem_ptr_regs_idxs) {
        h->mov(h->qword[h->rsp + call_args_size + i * sizeof(uintptr_t*)], Xbyak::Reg64(i));
    }

    const auto& aux_reg = get_call_address_reg();
    if (is_dynamic) {
        OPENVINO_THROW("dynamic parallel loop begin is not supported yet");
    } else {
        h->mov(aux_reg, reinterpret_cast<uintptr_t>(&loop_args));
        // TODO: use helper push_ptr_with_static_offset_on_stack and push_ptr_with_runtime_offset_on_stack
        // here and below
        h->mov(h->qword[h->rsp + GET_OFF_PARALLEL_LOOP_ARGS(loop_args)], aux_reg);
    }
    h->mov(aux_reg, *loop_preamble_label);
    h->mov(h->qword[h->rsp + GET_OFF_PARALLEL_LOOP_ARGS(preamble_ptr)], aux_reg);
    h->lea(aux_reg, h->qword[h->rsp + call_args_size]);
    h->mov(h->qword[h->rsp + GET_OFF_PARALLEL_LOOP_ARGS(mem_ptrs)], aux_reg);

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(ParallelLoopExecutor::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_executor.get()));
    h->mov(abi_param2, h->rsp);

    spill.rsp_align(get_callee_saved_reg().getIdx());
    // Note: we will return from this call only when the parallel region is finished (return from
    // jit_parallel_loop_end_emitter)
    h->call(aux_reg);
    spill.rsp_restore();

    // Restore data ptrs with applied finalization offsets
    for (auto i : mem_ptr_regs_idxs) {
        h->mov(Xbyak::Reg64(i), h->qword[h->rsp + call_args_size + i * sizeof(uintptr_t*)]);
    }
    h->add(h->rsp, reserved_stack_size);
    spill.postamble();

    h->jmp(*loop_end_label, Xbyak::CodeGenerator::T_NEAR);
}

void jit_parallel_loop_begin_emitter::emit_parallel_region_initialization() const {
    h->L(*loop_preamble_label);

    std::set<snippets::Reg> loop_premble_spill;
    // todo: we don't have to spill all calle+saved_regs, only the ones that will be used in the loop's body
    for (auto i : get_callee_saved_reg_idxs()) {
        loop_premble_spill.insert({snippets::RegType::gpr, i});
    }
    m_loop_reg_spiller->preamble(loop_premble_spill);
    // Note: work_amount reg is guaranteed to differ from any mem_ptr_regs_idxs.
    // However some of mem_ptr_regs_idxs might coincide with abi_param_1 or abi_param_2.
    h->mov(Xbyak::Reg64(internal_work_amount_reg_idx), abi_param1);
    bool abi_param2_collision = false;
    for (int i : mem_ptr_regs_idxs) {
        if (i == abi_param2.getIdx()) {
            abi_param2_collision = true;
        } else {
            h->mov(Xbyak::Reg64(i), h->ptr[abi_param2 + i * sizeof(uintptr_t*)]);
        }
    }
    if (abi_param2_collision) {
        h->mov(abi_param2, h->ptr[abi_param2 + abi_param2.getIdx() * sizeof(uintptr_t*)]);
    }

    h->L(*loop_begin_label);
}

void jit_parallel_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                                [[maybe_unused]] const std::vector<size_t>& out) const {
    emit_parallel_executor_call();
    // Note: parallel region starts here. The only legal entry point is from ParallelLoopExecutor::execute(...)
    emit_parallel_region_initialization();
}

jit_parallel_loop_end_emitter::jit_parallel_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                             dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                             const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{nullptr},
      loop_end_label{new Xbyak::Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::ParallelLoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter =
        std::dynamic_pointer_cast<jit_parallel_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
    m_loop_reg_spiller = loop_begin_emitter->get_loop_reg_spiller();
}

ov::snippets::lowered::ExpressionPtr jit_parallel_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::ParallelLoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

void jit_parallel_loop_end_emitter::validate_arguments(const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Invalid number of out arguments: expected 0 got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1,
                              "Invalid number of in arguments: expected ",
                              io_size + 1,
                              " got ",
                              in.size());
    OV_CPU_JIT_EMITTER_ASSERT(is_incremented.size() == io_size,
                              "Invalid is_incremented size: expected ",
                              io_size,
                              " got ",
                              is_incremented.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
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
    for (size_t idx = 0; idx < mem_ptr_regs_idxs.size(); idx++) {
        OPENVINO_ASSERT(!is_dynamic, "dynamic parallel loop end is not supported yet");
        const auto& ptr_increment = loop_args.m_ptr_increments[idx];
        if (is_incremented[idx] && ptr_increment != 0) {
            // TODO: take increment from call_args in case of dynamic loop
            h->add(Reg64(static_cast<int>(mem_ptr_regs_idxs[idx])), ptr_increment);
        }
    }

    auto reg_work_amount = Reg64(in.back());
    h->sub(reg_work_amount, wa_increment);
    h->cmp(reg_work_amount, wa_increment);
    h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    m_loop_reg_spiller->postamble();
    // todo: we can return at the end of the tail processing loop instead
    // Note: parallel region ends here:
    h->ret();
    h->L(*loop_end_label);
}

}  // namespace ov::intel_cpu
