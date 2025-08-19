// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"
#include "utils.hpp"
#include "openvino/core/except.hpp"

#define OV_CPU_JIT_EMITTER_ASSERT(cond, msg) OPENVINO_ASSERT(cond, msg)

using namespace Xbyak_riscv;

namespace ov::intel_cpu::riscv64 {

using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                               ov::intel_cpu::riscv64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : ov::intel_cpu::riscv64::jit_emitter(h, isa), isa(isa), h(h) {
    const auto loop_begin = ov::as_type_ptr<ov::snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "Expected LoopBegin expression");

    const auto loop_end = loop_begin->get_loop_end();
    work_amount = loop_end->get_work_amount();
    increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(work_amount);

    loop_begin_label = Xbyak_riscv::Label();
    loop_end_label = nullptr;

    // LoopBegin communicates work_amount via GPR to LoopEnd
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // The only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
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
    auto reg_work_amount = Xbyak_riscv::Reg(out[0]);
    if (!is_work_amount_dynamic) {
        // Load static work amount into provided GPR
        h->uni_li(reg_work_amount, static_cast<size_t>(work_amount));
    }
    if (evaluate_once) {
        // If evaluate once, just mark begin label (LoopEnd will handle single iteration)
        h->L(loop_begin_label);
        return;
    }
    // Regular loop: branch to end if zero
    h->L(loop_begin_label);
    OV_CPU_JIT_EMITTER_ASSERT(loop_end_label != nullptr, "Loop end label is not inited");
    h->beqz(reg_work_amount, *loop_end_label);
}

/* =================== jit_loop_end_emitter ======================= */

jit_loop_end_emitter::jit_loop_end_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                           ov::intel_cpu::riscv64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : ov::intel_cpu::riscv64::jit_emitter(h, isa), isa(isa), h(h) {
    const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end, "Expected LoopEnd expression");

    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    work_amount = loop_end->get_work_amount();
    increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    data_sizes = loop_end->get_element_type_sizes();
    evaluate_once = loop_end->get_evaluate_once();
    is_increment_dynamic = false; // simplified
    are_ptr_increments_dynamic = std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic = std::any_of(finalization_offsets.cbegin(), finalization_offsets.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);

    // Get corresponding LoopBegin
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_label = loop_begin_emitter->get_begin_label();
    loop_begin_emitter->set_loop_end_label(loop_end_label);
}

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Invalid number of out arguments: expected 0 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1,
                              "Invalid number of in arguments: expected " + std::to_string(io_size + 1) +
                                  " got " + std::to_string(in.size()));
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
    // in layout: [work_amount, ptr0, ptr1, ..., ptrN]
    auto reg_work_amount = Xbyak_riscv::Reg(in[0]);
    if (evaluate_once) {
        h->L(loop_end_label);
        return;
    }

    // Apply pointer increments for inputs/outputs
    const size_t io_size = num_inputs + num_outputs;
    for (size_t i = 0; i < io_size; ++i) {
        if (i < is_incremented.size() && is_incremented[i]) {
            auto ptr_reg = Xbyak_riscv::Reg(in[1 + i]);
            // load increment immediate and add
            Xbyak_riscv::Reg tmp = Xbyak_riscv::t2;
            h->uni_li(tmp, static_cast<size_t>(ptr_increments[i]));
            h->add(ptr_reg, ptr_reg, tmp);
        }
    }

    // Decrement work amount and loop
    Xbyak_riscv::Reg reg_increment = Xbyak_riscv::t1;
    h->uni_li(reg_increment, static_cast<size_t>(increment));
    h->sub(reg_work_amount, reg_work_amount, reg_increment);
    h->bnez(reg_work_amount, loop_begin_label);
    h->L(loop_end_label);
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have the last port connector to LoopBegin");
    return begin_expr;
}

}  // namespace ov::intel_cpu::riscv64
