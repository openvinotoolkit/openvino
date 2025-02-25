// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_parallel_loop_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/plugin/x64/utils.hpp"
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
            // todo: A long-term solution would be to introduce loop expressions so LoopBeginExpr->get_loop_end() would return LoopEndExpr directly
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
        work_amount = loop_end->get_work_amount();
        wa_increment = loop_end->get_increment();
        is_incremented = loop_end->get_is_incremented();
        ptr_increments = loop_end->get_ptr_increments();
        finalization_offsets = loop_end->get_finalization_offsets();
        data_sizes = loop_end->get_element_type_sizes();
        evaluate_once = loop_end->get_evaluate_once();
        loop_id = loop_end->get_id();
        // todo: it's redundant to save both isolated fields and loop_args_t. choose only one implementation
        loop_args = jit_snippets_call_args::loop_args_t(work_amount, ptr_increments, finalization_offsets, data_sizes);

        OV_CPU_JIT_EMITTER_ASSERT(loop_end_input_regs.size() == num_inputs + num_outputs && !loop_end_input_regs.empty(),
                                  "Invalid LoopEnd reg info");
        work_amount_reg_idx = loop_end_input_regs.rbegin()->idx;
        loop_end_input_regs.pop_back();
        mem_ptr_regs_idxs.reserve(loop_end_input_regs.size());
        for (const auto& r : loop_end_input_regs) {
            if (r.type == snippets::RegType::gpr)
                mem_ptr_regs_idxs.emplace_back(r.idx);
        }
}

void jit_parallel_loop_base_emitter::emit_pointer_increments(size_t scale) const {
    for (size_t idx = 0; idx < mem_ptr_regs_idxs.size(); idx++) {
        const auto& increment = ptr_increments[idx];
        if (is_incremented[idx] && increment != 0) {
            h->add(Reg64(static_cast<int>(mem_ptr_regs_idxs[idx])), increment * scale * data_sizes[idx]);
        }
    }
}

/* ================== jit_loop_begin_emitter ====================== */

jit_parallel_loop_begin_emitter::jit_parallel_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                                 const ov::snippets::lowered::ExpressionPtr& expr,
                                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{new Xbyak::Label()},
      loop_end_label(nullptr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(expr->get_node()), "expects LoopBegin expression");

    int num_threads = 2;
    ParallelLoopConfig kernel_config(loop_args, num_threads);
    m_parallel_loop_executor =
            kernel_table->register_kernel<ParallelLoopExecutor>(expr, kernel_config);

}

void jit_parallel_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
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


void jit_parallel_loop_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {

    init_binary_call_regs(3, mem_ptr_regs_idxs);

    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    auto reserved_stack_size = sizeof(Xbyak::Reg64) * mem_ptr_regs_idxs.size();
    // Spill before parallel for => we'll need them to update data ptrs afterwards
    h->sub(h->rsp, reserved_stack_size);
    for (auto i : mem_ptr_regs_idxs)
        h->mov(h->qword[h->rsp + i * sizeof(Xbyak::Reg64)], Xbyak::Reg64(i));

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(ParallelLoopExecutor::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_parallel_loop_executor.get()));
    h->mov(abi_param2, h->rsp);
//    h->mov(abi_param3, reinterpret_cast<const ParallelLoopExecutor::loop_preamble_t>(loop_begin_label->getAddress()));
    h->mov(abi_param3, reinterpret_cast<const uintptr_t>(loop_begin_label->getAddress()));

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    spill.postamble();
    
    // Restore incremented data ptrs
    for (auto i : mem_ptr_regs_idxs)
        h->mov(Xbyak::Reg64(i), h->qword[h->rsp + i * sizeof(Xbyak::Reg64)]);

    h->add(h->rsp, reserved_stack_size);

    h->jmp(*loop_end_label);

    h->L(*loop_begin_label);
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_parallel_loop_end_emitter::jit_parallel_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_parallel_loop_base_emitter(h, isa, expr),
      loop_begin_label{nullptr},
      loop_end_label{new Xbyak::Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_parallel_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

ov::snippets::lowered::ExpressionPtr jit_parallel_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

void jit_parallel_loop_end_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
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
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == io_size,
                              "Invalid ptr_increments size: expected ",
                              io_size,
                              " got ",
                              ptr_increments.size());
    OV_CPU_JIT_EMITTER_ASSERT(finalization_offsets.size() == io_size,
                              "Invalid finalization_offsets size: expected: ",
                              io_size,
                              " got ",
                              finalization_offsets.size());
    OV_CPU_JIT_EMITTER_ASSERT(data_sizes.size() == io_size,
                              "Invalid data_sizes size: expected: ",
                              io_size,
                              " got ",
                              data_sizes.size());
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

void jit_parallel_loop_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

        
    emit_pointer_increments(wa_increment);
    Reg64 reg_work_amount = Reg64(in.back());
    h->sub(reg_work_amount, wa_increment);
    h->cmp(reg_work_amount, wa_increment);
    h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    // todo: we can return at the end of the tail processing loop instead
    h->ret();
    h->L(*loop_end_label);
}

/* ============================================================== */

}  // namespace ov::intel_cpu
