// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <algorithm>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator_t;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa),
      loop_begin_label{new Xbyak_aarch64::Label()},
      loop_end_label{nullptr} {
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
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr, "has not inited begin label!");
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr, "has not inited end label!");
    if (ov::snippets::utils::is_dynamic_value(wa_increment)) {
        OV_CPU_JIT_EMITTER_ASSERT(evaluate_once, "wa_increment can be dynamic only when evaluate_once is true");
    }
    if (!is_work_amount_dynamic) {
        OV_CPU_JIT_EMITTER_ASSERT(work_amount != 0, "Static work_amount must not be 0");
    }
}

void jit_loop_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            [[maybe_unused]] const std::vector<size_t>& pool_vec_idxs,
                                            [[maybe_unused]] const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                       const std::vector<size_t>& out) const {
    if (evaluate_once && !is_work_amount_dynamic) {
        // If the loop evaluates once, we can skip loop begin code emission
        return;
    }

    auto reg_work_amount = XReg(out[0]);
    auto reg_runtime_params = XReg(Operand::X0);
    if (is_work_amount_dynamic) {
        XReg reg_aux = h->X_TMP_1;
        const auto id_offset = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
        h->ldr(reg_aux, ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(loop_args))));
        h->ldr(reg_work_amount, ptr(reg_aux, static_cast<int32_t>(id_offset + GET_OFF_LOOP_ARGS(m_work_amount))));

        auto increment = evaluate_once && snippets::utils::is_dynamic_value(wa_increment) ? 1 : wa_increment;
        h->cmp(reg_work_amount, increment);
        h->b(LT, *loop_end_label);
    } else {
        h->mov(reg_work_amount, work_amount);
    }
    h->L(*loop_begin_label);
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa),
      loop_begin_label{nullptr},
      loop_end_label{std::make_shared<Xbyak_aarch64::Label>()} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();
    are_ptr_increments_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_ptr_increments());
    are_final_offsets_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_finalization_offsets());
    loop_args = ov::intel_cpu::utils::compose_loop_args(loop_end);
    OV_CPU_JIT_EMITTER_ASSERT(loop_args.m_num_data_ptrs == static_cast<int64_t>(num_inputs + num_outputs),
                              "Invalid loop args size for LoopEnd");

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_label = loop_begin_emitter->get_begin_label();
    loop_begin_emitter->set_loop_end_label(loop_end_label);
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_expr_ptr(expr->get_input_count() - 1);
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have th last port connector to LoopBegin");
    return begin_expr;
}

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1,
                              "Invalid number of in arguments: expected ",
                              io_size + 1,
                              " got ",
                              in.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_args.m_num_data_ptrs == static_cast<int64_t>(io_size),
                              "Invalid loop args size: expected ",
                              io_size,
                              " got ",
                              loop_args.m_num_data_ptrs);
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr, "has not inited begin label!");
}

void jit_loop_end_emitter::emit_code_impl(const std::vector<size_t>& in,
                                          const std::vector<size_t>& out,
                                          [[maybe_unused]] const std::vector<size_t>& pool_vec_idxs,
                                          [[maybe_unused]] const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                     [[maybe_unused]] const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    auto reg_work_amount = XReg(in.back());
    auto reg_runtime_params = XReg(Operand::X0);
    XReg reg_aux = h->X_TMP_1;

    auto apply_increments = [&](const int64_t* increments, bool use_runtime_args, int32_t runtime_offset) {
        XReg reg_increments = h->X_TMP_0;
        if (use_runtime_args) {
            h->ldr(reg_increments, ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(loop_args))));
            h->ldr(reg_increments,
                   ptr(reg_increments,
                       static_cast<int32_t>(loop_id * sizeof(jit_snippets_call_args::loop_args_t) + runtime_offset)));
        }

        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            const auto increment = increments[idx];
            if (increment == 0) {
                continue;
            }
            auto data_reg = XReg(static_cast<int>(data_ptr_reg_idxs[idx]));
            if (snippets::utils::is_dynamic_value(increment)) {
                OV_CPU_JIT_EMITTER_ASSERT(use_runtime_args, "Dynamic increments require runtime loop arguments");
                h->ldr(reg_aux, ptr(reg_increments, static_cast<int32_t>(idx * sizeof(int64_t))));
                h->add(data_reg, data_reg, reg_aux);
            } else {
                if (increment > 0) {
                    h->add_imm(data_reg, data_reg, increment, reg_aux);
                } else if (increment < 0) {
                    h->sub_imm(data_reg, data_reg, -increment, reg_aux);
                }
            }
        }
    };

    if (!evaluate_once) {
        apply_increments(loop_args.m_ptr_increments, are_ptr_increments_dynamic, GET_OFF_LOOP_ARGS(m_ptr_increments));
        h->sub_imm(reg_work_amount, reg_work_amount, wa_increment, reg_aux);
        h->cmp(reg_work_amount, wa_increment);
        h->b(GE, *loop_begin_label);
    }

    apply_increments(loop_args.m_finalization_offsets,
                     are_final_offsets_dynamic,
                     GET_OFF_LOOP_ARGS(m_finalization_offsets));

    h->L(*loop_end_label);
}

/* ============================================================== */

}  // namespace ov::intel_cpu::aarch64
