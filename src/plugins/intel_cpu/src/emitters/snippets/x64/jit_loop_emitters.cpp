// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{new Xbyak::Label()}, loop_end_label(nullptr) {
    const auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "expects LoopBegin expression");
    const auto loop_end = loop_begin->get_loop_end();
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();
    is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(work_amount);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

size_t jit_loop_begin_emitter::aux_gprs_count() const {
    // We should have aux GPR to store Loop arguments from `runtime_args`
    // where we will take all needed information about the current loop: work amount
    return is_work_amount_dynamic ? 1 : 0;
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr && loop_end_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(implication(is_work_amount_dynamic, !evaluate_once), "with dynamic work_amount cannot evaluate once!");
}

void jit_loop_begin_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_loop_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    // If the loop evaulate once, we can skip loop begin code emission
    if (evaluate_once)
        return;

    Reg64 reg_work_amount = Reg64(static_cast<int>(out.back()));
    if (is_work_amount_dynamic) {
        Reg64 reg_runtime_params = abi_param1;  // defined by jit_kernel_emitter
        Reg64 reg_loop_args_ptr = Reg64(static_cast<int>(aux_gpr_idxs[0]));
        const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
        h->mov(reg_loop_args_ptr, h->ptr[reg_runtime_params + GET_OFF(loop_args)]);
        h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);
    } else {
        h->mov(reg_work_amount, work_amount);
    }

    // if wa < increment, skip the loop
    h->cmp(reg_work_amount, wa_increment);
    h->jl(*loop_end_label, Xbyak::CodeGenerator::T_NEAR);

    h->L(*loop_begin_label);
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{nullptr}, loop_end_label{new Xbyak::Label()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
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

    are_ptr_increments_dynamic =
        std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic =
        std::any_of(finalization_offsets.cbegin(), finalization_offsets.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_ptr_shifts_dynamic = are_ptr_increments_dynamic || are_final_offsets_dynamic;

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1, "Invalid number of in arguments: expected ", io_size + 1, " got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(is_incremented.size() == io_size, "Invalid is_incremented size: expected ", io_size, " got ", is_incremented.size());
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == io_size, "Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    OV_CPU_JIT_EMITTER_ASSERT(finalization_offsets.size() == io_size,
                              "Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
    OV_CPU_JIT_EMITTER_ASSERT(data_sizes.size() == io_size, "Invalid data_sizes size: expected: ", io_size, " got ", data_sizes.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(implication(are_ptr_shifts_dynamic, !evaluate_once), "with dynamic data pointer shifts cannot evaluate once!");
}

void jit_loop_end_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code(in, out, pool_vec_idxs, pool_gpr_idxs);
}

size_t jit_loop_end_emitter::aux_gprs_count() const {
    // We should have aux GPR to store Loop arguments from `runtime_args`
    // where we will take all needed information about the current loop: data pointer shifts
    return are_ptr_shifts_dynamic ? 1 : 0;
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
    Reg64 reg_increments = are_ptr_shifts_dynamic ? Reg64(static_cast<int>(aux_gpr_idxs[0])) : Reg64();

#define APPLY_INCREMENTS(USE_RUNTIME_ARGS, TYPE, INCREMENTS, SCALE)                                                         \
    if (USE_RUNTIME_ARGS) {                                                                                                 \
        Reg64 reg_runtime_params = abi_param1; /* defined by jit_kernel_emitter */                                          \
        h->mov(reg_increments, h->ptr[reg_runtime_params + GET_OFF(loop_args)]);                                            \
        h->mov(reg_increments, h->ptr[reg_increments + id_offset + GET_OFF_LOOP_ARGS(TYPE)]);                               \
    }                                                                                                                       \
    for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {                                                           \
        const auto& increment = INCREMENTS[idx];                                                                            \
        if (is_incremented[idx] && increment != 0) {                                                                        \
            if (ov::snippets::utils::is_dynamic_value(increment)) {                                                         \
                OV_CPU_JIT_EMITTER_ASSERT(USE_RUNTIME_ARGS, "Loop argument structure cannot be pushed to aux GPR");         \
                h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), h->ptr[reg_increments + idx * sizeof(int64_t)]);    \
            } else {                                                                                                        \
                h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), increment * SCALE * data_sizes[idx]);               \
            }                                                                                                               \
        }                                                                                                                   \
    }

    if (!evaluate_once) {
        APPLY_INCREMENTS(are_ptr_increments_dynamic, m_ptr_increments, ptr_increments, wa_increment);

        Reg64 reg_work_amount = Reg64(in.back());
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    }

    APPLY_INCREMENTS(are_final_offsets_dynamic, m_finalization_offsets, finalization_offsets, 1);

    h->L(*loop_end_label);

#undef APPLY_INCREMENTS
}

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
