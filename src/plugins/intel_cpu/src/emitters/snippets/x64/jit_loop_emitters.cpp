// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "jit_loop_base_emitters.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa),
      loop_begin_label{new Xbyak::Label()},
      loop_end_label(nullptr) {
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

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr && loop_end_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
}

void jit_loop_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            const std::vector<size_t>& pool_vec_idxs,
                                            const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                       const std::vector<size_t>& out) const {
    // If the loop evaulate once, we can skip loop begin code emission
    // If work_amount is dynamic, we should get runtime `work_amount` - it might be `zero` and we should skip loop
    // evaluation
    if (evaluate_once && !is_work_amount_dynamic) {
        return;
    }

    auto reg_work_amount = Reg64(static_cast<int>(out.back()));
    if (is_work_amount_dynamic) {
        utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, out);  // loop_begin has only output registers
        Reg64 reg_loop_args_ptr = gpr_holder.get_reg();
        const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
        h->mov(reg_loop_args_ptr, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);
    } else {
        h->mov(reg_work_amount, work_amount);
    }

    // if wa < increment, skip the loop
    // Note : If the loop should be evaluated once and increment is dynamic,
    //        we should manually set `increment = 1` to compare the dynamic work amount
    //        with `1` at least before loop execution
    //        (work amount can be zero and we should skip this loop even `evaluate_once = 1`)
    auto increment = evaluate_once && snippets::utils::is_dynamic_value(wa_increment) ? 1 : wa_increment;
    h->cmp(reg_work_amount, increment);
    h->jl(*loop_end_label, Xbyak::CodeGenerator::T_NEAR);

    h->L(*loop_begin_label);
}

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_base_emitter(h, isa, expr) {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());

    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& finalization_offsets = loop_end->get_finalization_offsets();

    are_ptr_increments_dynamic =
        std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic = std::any_of(finalization_offsets.cbegin(),
                                            finalization_offsets.cend(),
                                            ov::snippets::utils::is_dynamic_value<int64_t>);

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                     [[maybe_unused]] const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(io_num);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    if (!evaluate_once) {
        apply_increments_to_ptrs(data_ptr_reg_idxs,
                                 loop_args.m_ptr_increments,
                                 are_ptr_increments_dynamic,
                                 GET_OFF_LOOP_ARGS(m_ptr_increments),
                                 in);

        auto reg_work_amount = Reg64(in.back());
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(*loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    }

    apply_increments_to_ptrs(data_ptr_reg_idxs,
                             loop_args.m_finalization_offsets,
                             are_final_offsets_dynamic,
                             GET_OFF_LOOP_ARGS(m_finalization_offsets),
                             in);

    h->L(*loop_end_label);
}

}  // namespace ov::intel_cpu
