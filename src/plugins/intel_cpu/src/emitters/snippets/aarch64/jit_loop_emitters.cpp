// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"
#include "jit_kernel_emitter.hpp"
#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{new Xbyak_aarch64::Label()} {
    const auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "expects LoopBegin expression");
    const auto loop_end = loop_begin->get_loop_end();
    OV_CPU_JIT_EMITTER_ASSERT(!loop_end->has_dynamic_params(), "supports only static loops!");
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
    // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr, "has not inited label!");
}

void jit_loop_begin_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    XReg reg_work_amount = XReg(out[0]);
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    h->L(*loop_begin_label);
}

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), loop_begin_label{nullptr} {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end != nullptr, "expected LoopEnd expr");
    OV_CPU_JIT_EMITTER_ASSERT(!loop_end->has_dynamic_params(), "supports only static loops!");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    is_incremented = loop_end->get_is_incremented();
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    data_sizes = loop_end->get_element_type_sizes();
    evaluate_once = loop_end->get_evaluate_once();

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

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 0, "Invalid number of out arguments: expected ", 0, " got ", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1, "Invalid number of in arguments: expected ", io_size + 1, " got ", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(is_incremented.size() == io_size, "Invalid is_incremented size: expected ", io_size, " got ", is_incremented.size());
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == io_size, "Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    OV_CPU_JIT_EMITTER_ASSERT(finalization_offsets.size() == io_size,
                              "Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
    OV_CPU_JIT_EMITTER_ASSERT(data_sizes.size() == io_size, "Invalid data_sizes size: expected: ", io_size, " got ", data_sizes.size());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr, "has not inited begin label!");
}

void jit_loop_end_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    data_ptr_reg_idxs.reserve(num_inputs + num_outputs);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    XReg reg_work_amount = XReg(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
            if (!is_incremented[idx] || ptr_increments[idx] == 0)
                continue;
            XReg data_reg = XReg(data_ptr_reg_idxs[idx]);
            if (ptr_increments[idx] > 0) {
                h->add_imm(data_reg, data_reg, ptr_increments[idx] * wa_increment * data_sizes[idx], h->X_TMP_0);
            } else if (ptr_increments[idx] < 0) {
                h->sub_imm(data_reg, data_reg, - ptr_increments[idx] * wa_increment * data_sizes[idx], h->X_TMP_0);
            }
        }
        h->sub_imm(reg_work_amount, reg_work_amount, wa_increment, h->X_TMP_0);
        h->cmp(reg_work_amount, wa_increment);
        h->b(GE, *loop_begin_label);
    }

    for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); idx++) {
        if (!is_incremented[idx] || finalization_offsets[idx] == 0)
            continue;
        XReg data_reg = XReg(static_cast<int>(data_ptr_reg_idxs[idx]));
        if (finalization_offsets[idx] > 0) {
            h->add_imm(data_reg, data_reg, finalization_offsets[idx] * data_sizes[idx], h->X_TMP_0);
        } else if (finalization_offsets[idx] < 0) {
            h->sub_imm(data_reg, data_reg, - finalization_offsets[idx] * data_sizes[idx], h->X_TMP_0);
        }
    }
}

/* ============================================================== */

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
