// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Xbyak::Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){ return Xbyak::Reg64(static_cast<int>(idx)); });
}

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{new Xbyak::Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

std::shared_ptr<ov::snippets::op::LoopEnd> jit_loop_begin_emitter::get_loop_end(const ov::snippets::lowered::ExpressionPtr& expr) {
    OV_CPU_JIT_EMITTER_ASSERT(expr->get_output_port_connectors().size() == 1, "has invalid LoopBegin expression configuration");
    const auto& consumers = expr->get_output_port_connector(0)->get_consumers();
    OV_CPU_JIT_EMITTER_ASSERT(consumers.size() == 1, "has invalid LoopBegin expression configuration");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(consumers.cbegin()->get_expr()->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "has invalid LoopBegin expression configuration");
    return loop_end;
}

jit_loop_begin_static_emitter::jit_loop_begin_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                             const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_begin_emitter(h, isa, expr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBeginStatic>(expr->get_node()),
                              "expects LoopBeginStatic expression");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEndStatic>(get_loop_end(expr));
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
}

void jit_loop_begin_static_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
}

void jit_loop_begin_static_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    Xbyak::Reg64 reg_work_amount = Xbyak::Reg64(static_cast<int>(out.back()));
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    h->L(*loop_begin_label);
}

void jit_loop_begin_static_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                              const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

jit_loop_begin_dynamic_emitter::jit_loop_begin_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_begin_emitter(h, isa, expr), loop_end_label(nullptr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBeginDynamic>(expr->get_node()), "expects LoopBeginDynamic expression");
    const auto loop_end = get_loop_end(expr);
    wa_increment = loop_end->get_increment();
    loop_id = loop_end->get_id();
}

void jit_loop_begin_dynamic_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    // Note: the only expected input is the reg_runtime_params_idx
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
}

void jit_loop_begin_dynamic_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                               const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out);
}

void jit_loop_begin_dynamic_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    Xbyak::Reg64 reg_runtime_params = abi_param1;  // defined by jit_kernel_emitter
    Xbyak::Reg64 reg_work_amount = Xbyak::Reg64(static_cast<int>(out.back()));
    Xbyak::Reg64 reg_loop_args_ptr = Xbyak::Reg64(static_cast<int>(aux_gpr_idxs[0]));
    const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
    h->mov(reg_loop_args_ptr, h->ptr[reg_runtime_params + GET_OFF(loop_args)]);
    h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);

    // if wa < increment, skip the loop
    h->cmp(reg_work_amount, wa_increment);
    h->jl(*loop_end_label, Xbyak::CodeGenerator::T_NEAR);

    h->L(*loop_begin_label);
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{nullptr} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

jit_loop_end_static_emitter::jit_loop_end_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                         const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_emitter(h, isa, expr) {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEndStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEndStatic expr");
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    is_incremented = loop_end->get_is_incremented();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    data_sizes = loop_end->get_element_type_sizes();
    evaluate_once = loop_end->get_evaluate_once();
}

void jit_loop_end_static_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1, "Invalid number of in arguments: expected ", io_size + 1, " got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == io_size, "Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    OV_CPU_JIT_EMITTER_ASSERT(finalization_offsets.size() == io_size,
                              "Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
    OV_CPU_JIT_EMITTER_ASSERT(data_sizes.size() == io_size, "Invalid data_sizes size: expected: ", io_size, " got ", data_sizes.size());
}

void jit_loop_end_static_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                            const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_end_static_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs - 1);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    Reg64 reg_work_amount = Reg64(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            if (!is_incremented[idx] || ptr_increments[idx] == 0)
                continue;
            Reg64 data_reg = Reg64(static_cast<int>(data_ptr_reg_idxs[idx]));
            h->add(data_reg, ptr_increments[idx] * wa_increment * data_sizes[idx]);
        }
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(*loop_begin_label);
    }

    for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
        if (!is_incremented[idx] || finalization_offsets[idx] == 0)
            continue;
        Reg64 data_reg = Reg64(static_cast<int>(data_ptr_reg_idxs[idx]));
        h->add(data_reg, finalization_offsets[idx] * data_sizes[idx]);
    }
}

jit_loop_end_dynamic_emitter::jit_loop_end_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_emitter(h, isa, expr), loop_end_label{new Xbyak::Label()} {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEndDynamic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEndDynamic expr");
    loop_id = loop_end->get_id();

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_dynamic_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBeginDynamic expected jit_loop_begin_dynamic_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
}

void jit_loop_end_dynamic_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    // Note: there must be additional input argument for runtime parameters
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1, "Invalid number of in arguments: expected ", io_size + 1, " got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
}

void jit_loop_end_dynamic_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                             const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out);
}

void jit_loop_end_dynamic_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    Xbyak::Reg64 reg_runtime_params = abi_param1;  // defined by jit_kernel_emitter
    Xbyak::Reg64 reg_work_amount = Xbyak::Reg64(static_cast<int>(in[in.size() - 1]));
    Xbyak::Reg64 reg_increments = Xbyak::Reg64(static_cast<int>(aux_gpr_idxs[0]));
    const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);

    std::vector<Xbyak::Reg64> data_ptr_regs;
    transform_idxs_to_regs(std::vector<size_t>(in.begin(), in.end() - 1), data_ptr_regs);

    // todo: Note that we can pre-save reg_loop_args_ptr in jit_loop_begin_dynamic_emitter and pass it here like work_amount_reg
    //        this would save us one dereferencing here and in finalization offsets
    h->mov(reg_increments, h->ptr[reg_runtime_params + GET_OFF(loop_args)]);
    h->mov(reg_increments, h->ptr[reg_increments + id_offset + GET_OFF_LOOP_ARGS(m_ptr_increments)]);
    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (is_incremented[idx])
            h->add(data_ptr_regs[idx], h->ptr[reg_increments + idx * sizeof(int64_t)]);
    }
    h->sub(reg_work_amount, wa_increment);
    h->cmp(reg_work_amount, wa_increment);
    h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);

    h->mov(reg_increments, h->ptr[reg_runtime_params + GET_OFF(loop_args)]);
    h->mov(reg_increments, h->ptr[reg_increments + id_offset + GET_OFF_LOOP_ARGS(m_finalization_offsets)]);
    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (is_incremented[idx])
            h->add(data_ptr_regs[idx], h->ptr[reg_increments + idx * sizeof(int64_t)]);
    }

    h->L(*loop_end_label);
}

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
