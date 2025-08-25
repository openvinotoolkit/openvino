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

jit_loop_end_base_emitter::jit_loop_end_base_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end, "Expected LoopEnd node");

    wa_increment = loop_end->get_increment();
    loop_begin_label = std::make_shared<const Xbyak::Label>();
    loop_end_label = std::make_shared<Xbyak::Label>();

    io_num = loop_end->get_input_num() + loop_end->get_output_num();
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id_offset = loop_end->get_id() * sizeof(jit_snippets_call_args::loop_args_t);

    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& finalization_offsets = loop_end->get_finalization_offsets();
    are_ptr_increments_dynamic =
        std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic = std::any_of(finalization_offsets.cbegin(),
                                            finalization_offsets.cend(),
                                            ov::snippets::utils::is_dynamic_value<int64_t>);

    loop_args = compose_loop_args(loop_end);
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
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_num + 1,
                              "Invalid number of in arguments: expected ",
                              io_num + 1,
                              " got ",
                              in.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr && loop_begin_label != nullptr, "has not inited labels!");
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
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
    Reg64 reg_increments;
    auto add_increments = [&]() {
        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            const auto& increment = increments[idx];
            if (increment != 0) {
                if (ov::snippets::utils::is_dynamic_value(increment)) {
                    OV_CPU_JIT_EMITTER_ASSERT(use_runtime_args, "Loop argument structure cannot be pushed to aux GPR");
                    h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])),
                           h->ptr[reg_increments + idx * sizeof(int64_t)]);
                } else {
                    // Use pre-computed increment value from loop_args (already scaled)
                    h->add(Reg64(static_cast<int>(data_ptr_reg_idxs[idx])), increment);
                }
            }
        }
    };

    if (use_runtime_args) {
        utils::jit_aux_gpr_holder gpr_holder(h, aux_gpr_idxs, used_aux_gprs);
        reg_increments = gpr_holder.get_reg();
        h->mov(reg_increments, h->ptr[abi_param1 + GET_OFF(loop_args)]);
        h->mov(reg_increments, h->ptr[reg_increments + loop_id_offset + field_offset]);
        add_increments();
    } else {
        add_increments();
    }
}

}  // namespace ov::intel_cpu
