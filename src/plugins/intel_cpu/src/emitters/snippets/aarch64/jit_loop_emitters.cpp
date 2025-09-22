// Copyright (C) 2024 Intel Corporation
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
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
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

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
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
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    data_sizes = loop_end->get_element_type_sizes();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();

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

    auto apply_increments = [&](const std::vector<int64_t>& increments_vec,
                                int64_t increment_multiplier,
                                int32_t runtime_offset) {
        XReg reg_increments = h->X_TMP_0;
        bool has_dynamic = std::any_of(increments_vec.begin(), increments_vec.end(), [](int64_t val) {
            return snippets::utils::is_dynamic_value(val);
        });
        if (has_dynamic) {
            h->ldr(reg_increments, ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(loop_args))));
            h->ldr(reg_increments,
                   ptr(reg_increments,
                       static_cast<int32_t>(loop_id * sizeof(jit_snippets_call_args::loop_args_t) + runtime_offset)));
        }

        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            if (!is_incremented[idx] || increments_vec[idx] == 0) {
                continue;
            }
            auto data_reg = XReg(static_cast<int>(data_ptr_reg_idxs[idx]));
            if (snippets::utils::is_dynamic_value(increments_vec[idx])) {
                h->ldr(reg_aux, ptr(reg_increments, static_cast<int32_t>(idx * sizeof(int64_t))));
                h->add(data_reg, data_reg, reg_aux);
            } else {
                int64_t offset = increments_vec[idx] * increment_multiplier * data_sizes[idx];
                if (offset > 0) {
                    h->add_imm(data_reg, data_reg, offset, reg_aux);
                } else if (offset < 0) {
                    h->sub_imm(data_reg, data_reg, -offset, reg_aux);
                }
            }
        }
    };

    if (!evaluate_once) {
        apply_increments(ptr_increments, wa_increment, GET_OFF_LOOP_ARGS(m_ptr_increments));
        h->sub_imm(reg_work_amount, reg_work_amount, wa_increment, reg_aux);
        h->cmp(reg_work_amount, wa_increment);
        h->b(GE, *loop_begin_label);
    }

    apply_increments(finalization_offsets, 1, GET_OFF_LOOP_ARGS(m_finalization_offsets));

    h->L(*loop_end_label);
}

/* ============================================================== */

}  // namespace ov::intel_cpu::aarch64
