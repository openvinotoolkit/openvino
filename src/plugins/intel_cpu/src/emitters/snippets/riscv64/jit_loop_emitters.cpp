// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <optional>
#include <string>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/common/jit_loop_args_helper.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"
#include "utils.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

using namespace Xbyak_riscv;

namespace ov::intel_cpu::riscv64 {

using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

namespace {
// RAII holder for one temporary GPR: uses pool if available, otherwise preserves a caller-saved reg on stack
class jit_aux_gpr_holder {
public:
    jit_aux_gpr_holder(ov::intel_cpu::riscv64::jit_generator_t* host,
                       std::vector<size_t>& pool_gpr_idxs,
                       const std::vector<size_t>& used_gpr_idxs)
        : m_h(host),
          m_pool_gpr_idxs(pool_gpr_idxs) {
        if (!m_pool_gpr_idxs.empty()) {
            m_reg = Xbyak_riscv::Reg(static_cast<int>(m_pool_gpr_idxs.back()));
            m_pool_gpr_idxs.pop_back();
        } else {
            // choose an available caller-saved reg not in used set
            m_reg = ov::intel_cpu::riscv64::utils::get_aux_gpr(used_gpr_idxs);
            m_preserved = true;
            // Maintain 16-byte alignment; reserve 16 bytes and save at 0
            m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -16);
            m_h->sd(m_reg, Xbyak_riscv::sp, 0);
        }
    }
    ~jit_aux_gpr_holder() {
        if (m_preserved) {
            m_h->ld(m_reg, Xbyak_riscv::sp, 0);
            m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, 16);
        } else {
            m_pool_gpr_idxs.push_back(static_cast<size_t>(m_reg.getIdx()));
        }
    }
    [[nodiscard]] const Xbyak_riscv::Reg& get_reg() const {
        return m_reg;
    }

private:
    ov::intel_cpu::riscv64::jit_generator_t* m_h;
    std::vector<size_t>& m_pool_gpr_idxs;
    Xbyak_riscv::Reg m_reg;
    bool m_preserved = false;
};
}  // namespace

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                               ov::intel_cpu::riscv64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : ov::intel_cpu::riscv64::jit_emitter(h, isa),
      isa(isa),
      h(h) {
    const auto loop_begin = ov::as_type_ptr<ov::snippets::op::LoopBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "Expected LoopBegin expression");

    const auto loop_end = loop_begin->get_loop_end();
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    loop_id = loop_end->get_id();
    is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(work_amount);
    OV_CPU_JIT_EMITTER_ASSERT(wa_increment > 0, "Loop increment must be > 0");

    loop_begin_label = std::make_shared<Xbyak_riscv::Label>();
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
                                            const std::vector<size_t>& pool_vec_idxs,
                                            const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    // Use base preamble/postamble to manage aux regs consistently
    ov::intel_cpu::riscv64::jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs, {});
}

void jit_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                       const std::vector<size_t>& out) const {
    auto reg_work_amount = Xbyak_riscv::Reg(out[0]);
    if (is_work_amount_dynamic) {
        const auto id_offset = loop_id * sizeof(ov::intel_cpu::jit_snippets_call_args::loop_args_t);
        // Acquire two scratch regs
        std::vector<size_t> used = {out[0]};
        jit_aux_gpr_holder h_ptr(h, aux_gpr_idxs, used);
        jit_aux_gpr_holder h_tmp(h, aux_gpr_idxs, used);
        auto reg_loop_args_ptr = h_ptr.get_reg();
        auto addr = h_tmp.get_reg();
        // reg_loop_args_ptr = *(a0 + GET_OFF(loop_args))
        h->uni_li(addr, GET_OFF(loop_args));
        h->add(addr, Xbyak_riscv::a0, addr);
        h->ld(reg_loop_args_ptr, addr, 0);
        // reg_loop_args_ptr += id_offset + OFF(m_work_amount)
        h->uni_li(addr, id_offset + GET_OFF_LOOP_ARGS(m_work_amount));
        h->add(reg_loop_args_ptr, reg_loop_args_ptr, addr);
        // load m_work_amount
        h->ld(reg_work_amount, reg_loop_args_ptr, 0);
    } else {
        h->uni_li(reg_work_amount, static_cast<size_t>(work_amount));
    }
    h->L(*loop_begin_label);
    // If evaluate_once and not dynamic increment, skip branch to end
    if (evaluate_once && !ov::snippets::utils::is_dynamic_value(wa_increment)) {
        return;
    }
    // Compare work amount with increment and jump to end if less
    size_t eff_inc =
        (evaluate_once && ov::snippets::utils::is_dynamic_value(wa_increment)) ? 1 : static_cast<size_t>(wa_increment);
    // Use scratch for increment immediate
    std::vector<size_t> used2 = {out[0]};
    jit_aux_gpr_holder h_inc(h, aux_gpr_idxs, used2);
    Xbyak_riscv::Reg reg_inc = h_inc.get_reg();
    h->uni_li(reg_inc, eff_inc);
    h->blt(reg_work_amount, reg_inc, *loop_end_label);
}

/* =================== jit_loop_end_emitter ======================= */

jit_loop_end_emitter::jit_loop_end_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                                           ov::intel_cpu::riscv64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : ov::intel_cpu::riscv64::jit_emitter(h, isa),
      isa(isa),
      h(h) {
    const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(loop_end, "Expected LoopEnd expression");

    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    work_amount = loop_end->get_work_amount();
    wa_increment = loop_end->get_increment();
    evaluate_once = loop_end->get_evaluate_once();
    are_ptr_increments_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_ptr_increments());
    are_final_offsets_dynamic = ov::snippets::utils::has_dynamic_values(loop_end->get_finalization_offsets());
    OV_CPU_JIT_EMITTER_ASSERT(wa_increment > 0, "Loop increment must be > 0");
    loop_id = loop_end->get_id();
    loop_args_offset = loop_id * sizeof(ov::intel_cpu::jit_snippets_call_args::loop_args_t);
    loop_args = ov::intel_cpu::snippets_common::compose_loop_args(loop_end);
    OV_CPU_JIT_EMITTER_ASSERT(loop_args.m_num_data_ptrs == static_cast<int64_t>(num_inputs + num_outputs),
                              "Invalid loop args size for LoopEnd");

    // Get corresponding LoopBegin
    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_label = loop_begin_emitter->get_begin_label();
    loop_end_label = std::make_shared<Xbyak_riscv::Label>();
    loop_begin_emitter->set_loop_end_label(loop_end_label);
}

void jit_loop_end_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    const auto io_size = num_inputs + num_outputs;
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(),
                              "Invalid number of out arguments: expected 0 got " + std::to_string(out.size()));
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == io_size + 1,
                              "Invalid number of in arguments: expected " + std::to_string(io_size + 1) + " got " +
                                  std::to_string(in.size()));
    OV_CPU_JIT_EMITTER_ASSERT(loop_args.m_num_data_ptrs == static_cast<int64_t>(io_size),
                              "Invalid loop args size: expected " + std::to_string(io_size) + " got " +
                                  std::to_string(loop_args.m_num_data_ptrs));
}

void jit_loop_end_emitter::emit_code_impl(const std::vector<size_t>& in,
                                          const std::vector<size_t>& out,
                                          const std::vector<size_t>& pool_vec_idxs,
                                          const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    ov::intel_cpu::riscv64::jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs, {});
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                     [[maybe_unused]] const std::vector<size_t>& out) const {
    // Build list of data pointer regs: in[0..io_size-1], work_amount is in.back()
    std::vector<size_t> data_ptr_reg_idxs;
    const size_t io_size = num_inputs + num_outputs;
    data_ptr_reg_idxs.reserve(io_size);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));

    auto apply_increments = [&](const int64_t* increments, bool use_runtime_args, size_t field_offset) {
        if (increments == nullptr || data_ptr_reg_idxs.empty()) {
            return;
        }

        std::vector<size_t> used = in;
        std::unique_ptr<jit_aux_gpr_holder> reg_increments_holder;
        std::optional<Xbyak_riscv::Reg> reg_increments;
        if (use_runtime_args) {
            reg_increments_holder = std::make_unique<jit_aux_gpr_holder>(h, aux_gpr_idxs, used);
            reg_increments = reg_increments_holder->get_reg();
            used.push_back(static_cast<size_t>(reg_increments->getIdx()));
        }
        jit_aux_gpr_holder h_tmp(h, aux_gpr_idxs, used);
        Xbyak_riscv::Reg tmp = h_tmp.get_reg();

        if (use_runtime_args) {
            h->uni_li(tmp, GET_OFF(loop_args));
            h->add(tmp, Xbyak_riscv::a0, tmp);
            h->ld(*reg_increments, tmp, 0);
            h->uni_li(tmp, loop_args_offset + field_offset);
            h->add(*reg_increments, *reg_increments, tmp);
            h->ld(*reg_increments, *reg_increments, 0);
        }

        for (size_t idx = 0; idx < data_ptr_reg_idxs.size(); ++idx) {
            const auto increment = increments[idx];
            if (increment == 0) {
                continue;
            }
            auto ptr_reg = Xbyak_riscv::Reg(data_ptr_reg_idxs[idx]);
            if (ov::snippets::utils::is_dynamic_value(increment)) {
                OV_CPU_JIT_EMITTER_ASSERT(use_runtime_args, "Dynamic increments require runtime loop arguments");
                h->uni_li(tmp, idx * sizeof(int64_t));
                h->add(tmp, *reg_increments, tmp);
                h->ld(tmp, tmp, 0);
                h->add(ptr_reg, ptr_reg, tmp);
            } else {
                h->uni_li(tmp, static_cast<size_t>(increment));
                h->add(ptr_reg, ptr_reg, tmp);
            }
        }
    };

    if (!evaluate_once) {
        apply_increments(loop_args.m_ptr_increments, are_ptr_increments_dynamic, GET_OFF_LOOP_ARGS(m_ptr_increments));

        auto reg_work_amount = Xbyak_riscv::Reg(in.back());
        // reg_work_amount -= wa_increment
        // use scratch for increment immediate
        jit_aux_gpr_holder h_inc(h, aux_gpr_idxs, in);
        auto reg_inc = h_inc.get_reg();
        h->uni_li(reg_inc, static_cast<size_t>(wa_increment));
        h->sub(reg_work_amount, reg_work_amount, reg_inc);
        // if reg_work_amount >= wa_increment -> loop
        h->bge(reg_work_amount, reg_inc, *loop_begin_label);
    }

    apply_increments(loop_args.m_finalization_offsets,
                     are_final_offsets_dynamic,
                     GET_OFF_LOOP_ARGS(m_finalization_offsets));

    h->L(*loop_end_label);
}

ov::snippets::lowered::ExpressionPtr jit_loop_end_emitter::get_loop_begin_expr(
    const ov::snippets::lowered::ExpressionPtr& expr) {
    auto begin_expr = expr->get_input_port_connectors().back()->get_source().get_expr();
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<ov::snippets::op::LoopBegin>(begin_expr->get_node()),
                              "LoopEnd expression must have the last port connector to LoopBegin");
    return begin_expr;
}

}  // namespace ov::intel_cpu::riscv64
