// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_reg_spill_emitters.hpp"

#include "emitters/plugin/x64/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

/* ================== jit_reg_spill_begin_emitters ====================== */

jit_reg_spill_begin_emitter::jit_reg_spill_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                         const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto& reg_spill_node = ov::as_type_ptr<snippets::op::RegSpillBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(reg_spill_node, "expects RegSpillBegin expression");
    const auto& rinfo = expr->get_reg_info();
    m_regs_to_spill = std::set<snippets::Reg>(rinfo.second.begin(), rinfo.second.end());
    m_abi_reg_spiller = std::make_shared<EmitABIRegSpills>(h);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_reg_spill_begin_emitter::validate_arguments(const std::vector<size_t>& in,
                                                     const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "In regs should be empty for reg_spill_begin emitter");
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == m_regs_to_spill.size(),
                              "Invalid number of out regs for reg_spill_begin emitter");
}

void jit_reg_spill_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                                 const std::vector<size_t>& out,
                                                 const std::vector<size_t>& pool_vec_idxs,
                                                 const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    m_abi_reg_spiller->preamble(m_regs_to_spill);
}

/* ============================================================== */

/* ================== jit_reg_spill_end_emitter ====================== */

jit_reg_spill_end_emitter::jit_reg_spill_end_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::RegSpillEnd>(expr->get_node()) && expr->get_input_count() > 0,
                              "Invalid expression in RegSpillEnd emitter");
    const auto& parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
    const auto& reg_spill_begin_emitter =
        std::dynamic_pointer_cast<jit_reg_spill_begin_emitter>(parent_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(reg_spill_begin_emitter, "Failed to obtain reg_spill_begin emitter");
    m_abi_reg_spiller = reg_spill_begin_emitter->m_abi_reg_spiller;
}

void jit_reg_spill_end_emitter::validate_arguments(const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Out regs should be empty for reg_spill_end emitter");
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == m_abi_reg_spiller->get_num_spilled_regs(),
                              "Invalid number of in regs for reg_spill_end emitter");
}

void jit_reg_spill_end_emitter::emit_code_impl(const std::vector<size_t>& in,
                                               const std::vector<size_t>& out,
                                               const std::vector<size_t>& pool_vec_idxs,
                                               const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    m_abi_reg_spiller->postamble();
}

}  // namespace ov::intel_cpu
