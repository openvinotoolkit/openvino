// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_base_emitters.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_loop_begin_base_emitter::jit_loop_begin_base_emitter(jit_generator_t* h,
                                                         cpu_isa_t isa,
                                                         const ov::snippets::lowered::ExpressionPtr& expr,
                                                         bool is_parallel)
    : jit_emitter(h, isa),
      m_loop_begin_label(std::make_shared<Xbyak::Label>()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin && loop_begin->get_is_parallel() == is_parallel,
                              "expects LoopBegin expression with is parallel = ",
                              is_parallel);
    auto loop_end = loop_begin->get_loop_end();
    m_wa_increment = loop_end->get_increment();
    m_evaluate_once = loop_end->get_evaluate_once();
    m_loop_id_offset = loop_end->get_id() * sizeof(jit_snippets_call_args::loop_args_t);
}

void jit_loop_begin_base_emitter::validate_arguments(const std::vector<size_t>& in,
                                                     const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(m_loop_begin_label != nullptr && m_loop_end_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!ov::snippets::utils::is_dynamic_value(m_wa_increment) || m_evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
}

ov::snippets::lowered::ExpressionPtr jit_loop_begin_base_emitter::get_loop_end_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "Expected LoopBegin expression");

    const auto& consumers = expr->get_output_port_connector(expr->get_output_count() - 1)->get_consumers();
    OV_CPU_JIT_EMITTER_ASSERT(!consumers.empty(), "LoopBegin must have LoopEnd as the last consumer");
    const auto& loop_end_expr = consumers.rbegin()->get_expr();

    const auto expected_loop_end = loop_begin->get_loop_end();
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_expr && loop_end_expr->get_node() == expected_loop_end,
                              "Failed to find valid LoopEnd expression");
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin->get_is_parallel() == expected_loop_end->get_is_parallel(),
                              "LoopBegin and LoopEnd must have the same is_parallel attribute");
    return loop_end_expr;
}

void jit_loop_begin_base_emitter::emit_loop_begin_work_amount_check(const std::vector<size_t>& out,
                                                                    bool is_work_amount_dynamic,
                                                                    int64_t work_amount_static) const {
    auto reg_work_amount = Reg64(static_cast<int>(out.back()));
    if (is_work_amount_dynamic) {
        utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, out);
        Reg64 reg_loop_args_ptr = gpr_holder.get_reg();
        h->mov(reg_loop_args_ptr, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->mov(reg_work_amount, h->ptr[reg_loop_args_ptr + m_loop_id_offset + GET_OFF_LOOP_ARGS(m_work_amount)]);
    } else {
        h->mov(reg_work_amount, work_amount_static);
    }

    // if wa < increment, skip the loop
    // Note : If the loop should be evaluated once and increment is dynamic,
    //        we should manually set `increment = 1` to compare the dynamic work amount
    //        with `1` at least before loop execution
    //        (work amount can be zero and we should skip this loop even `m_evaluate_once = 1`)
    auto increment = m_evaluate_once && ov::snippets::utils::is_dynamic_value(m_wa_increment) ? 1 : m_wa_increment;
    h->cmp(reg_work_amount, increment);
    h->jl(*m_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
}

void jit_loop_begin_base_emitter::emit_code_impl(const std::vector<size_t>& in_idxs,
                                                 const std::vector<size_t>& out_idxs,
                                                 const std::vector<size_t>& pool_vec_idxs,
                                                 const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in_idxs, out_idxs);
    jit_emitter::emit_code_impl(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
}

jit_loop_end_base_emitter::jit_loop_end_base_emitter(jit_generator_t* h,
                                                     cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr,
                                                     bool is_parallel)
    : jit_emitter(h, isa),
      m_loop_end_label(std::make_shared<Xbyak::Label>()) {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end && loop_end->get_is_parallel() == is_parallel,
                              "Expected LoopEnd node with is parallel = ",
                              is_parallel);

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_base_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_base_emitter");
    loop_begin_emitter->set_loop_end_label(m_loop_end_label);
    m_loop_begin_label = loop_begin_emitter->get_begin_label();

    m_io_num = loop_end->get_input_num() + loop_end->get_output_num();
    m_wa_increment = loop_end->get_increment();
    m_loop_id_offset = loop_end->get_id() * sizeof(jit_snippets_call_args::loop_args_t);
    m_evaluate_once = loop_end->get_evaluate_once();
    m_are_ptr_increments_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_ptr_increments());
    m_are_final_offsets_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_finalization_offsets());
    m_loop_args = compose_loop_args(loop_end);
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_base_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have the last port connector to LoopBegin");
    return begin_expr;
}

jit_snippets_call_args::loop_args_t jit_loop_end_base_emitter::compose_loop_args(
    const std::shared_ptr<snippets::op::LoopEnd>& loop_end) {
    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& fin_offsets = loop_end->get_finalization_offsets();
    const auto& is_incremented = loop_end->get_is_incremented();
    const auto wa_increment = loop_end->get_increment();

    const auto int_work_amount = ov::snippets::utils::is_dynamic_value(loop_end->get_work_amount())
                                     ? ov::snippets::utils::get_dynamic_value<int64_t>()
                                     : static_cast<int64_t>(loop_end->get_work_amount());
    auto loop_args = jit_snippets_call_args::loop_args_t(int_work_amount, ptr_increments, fin_offsets);

    const auto& data_sizes = loop_end->get_element_type_sizes();
    for (int64_t i = 0; i < loop_args.m_num_data_ptrs; ++i) {
        // Increments for non-incremented indices should be zeroed
        if (!is_incremented[i]) {
            loop_args.m_ptr_increments[i] = 0;
            loop_args.m_finalization_offsets[i] = 0;
            continue;
        }

        // Note: behavior is aligned with runtime configurator:
        // data_sizes and increment are already taken into account in the offsets
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_ptr_increments[i])) {
            loop_args.m_ptr_increments[i] *= (wa_increment * data_sizes[i]);
        }
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_finalization_offsets[i])) {
            loop_args.m_finalization_offsets[i] *= data_sizes[i];
        }
    }

    return loop_args;
}

void jit_loop_end_base_emitter::validate_arguments(const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == m_io_num + 1,
                              "Invalid number of in arguments: expected ",
                              m_io_num + 1,
                              " got ",
                              in.size());
    OV_CPU_JIT_EMITTER_ASSERT(m_loop_end_label != nullptr && m_loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(m_wa_increment) || m_evaluate_once,
                              "loop increment might be dynamic only if loop evaluates once!");
}

void jit_loop_end_base_emitter::emit_code_impl(const std::vector<size_t>& in,
                                               const std::vector<size_t>& out,
                                               const std::vector<size_t>& pool_vec_idxs,
                                               const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_loop_end_base_emitter::apply_increments_to_ptrs(const std::vector<size_t>& data_ptr_reg_idxs,
                                                         const int64_t* increments,
                                                         bool use_runtime_args,
                                                         size_t field_offset,
                                                         const std::vector<size_t>& used_aux_gprs) const {
    auto add_increments = [&](std::optional<Reg64> reg_increments = std::nullopt) {
        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            const auto& increment = increments[idx];
            if (increment != 0) {
                if (ov::snippets::utils::is_dynamic_value(increment)) {
                    OV_CPU_JIT_EMITTER_ASSERT(reg_increments.has_value(),
                                              "Loop argument structure cannot be pushed to aux GPR");
                    h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])),
                           h->ptr[reg_increments.value() + idx * sizeof(int64_t)]);
                } else {
                    // Use pre-computed increment value from loop_args (already scaled)
                    h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), increment);
                }
            }
        }
    };

    if (use_runtime_args) {
        utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, used_aux_gprs);
        auto reg_increments = gpr_holder.get_reg();
        h->mov(reg_increments, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->mov(reg_increments, h->ptr[reg_increments + m_loop_id_offset + field_offset]);
        add_increments(reg_increments);
    } else {
        add_increments();
    }
}

void jit_loop_end_base_emitter::emit_loop_end_impl(const std::vector<size_t>& in,
                                                   bool apply_finalization_offsets) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(m_io_num);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    if (!m_evaluate_once) {
        apply_increments_to_ptrs(data_ptr_reg_idxs,
                                 m_loop_args.m_ptr_increments,
                                 m_are_ptr_increments_dynamic,
                                 GET_OFF_LOOP_ARGS(m_ptr_increments),
                                 in);

        auto reg_work_amount = Reg64(in.back());
        h->sub(reg_work_amount, m_wa_increment);
        h->cmp(reg_work_amount, m_wa_increment);
        h->jge(*m_loop_begin_label, Xbyak::CodeGenerator::T_NEAR);
    }

    if (apply_finalization_offsets) {
        apply_increments_to_ptrs(data_ptr_reg_idxs,
                                 m_loop_args.m_finalization_offsets,
                                 m_are_final_offsets_dynamic,
                                 GET_OFF_LOOP_ARGS(m_finalization_offsets),
                                 in);
    }
}

}  // namespace ov::intel_cpu
