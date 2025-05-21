// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_binary_call_emitter.hpp"

#include "emitters/plugin/x64/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_binary_call_emitter::jit_binary_call_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                 std::set<snippets::Reg> live_regs)
    : jit_emitter(h, isa),
      m_regs_to_spill(std::move(live_regs)) {}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& in,
                                                    const std::vector<size_t>& out) const {
    std::vector<size_t> mem_ptr_idxs = in;
    mem_ptr_idxs.insert(mem_ptr_idxs.end(), out.begin(), out.end());
    init_binary_call_regs(num_binary_args, mem_ptr_idxs);
}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& used_gpr_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(sizeof(abi_param_regs) / sizeof(*abi_param_regs) >= num_binary_args,
                              "Requested number of runtime arguments is not supported");
    // This regs will be corrupted, since we'll use them to pass runtime args
    for (size_t i = 0; i < num_binary_args; i++) {
        m_regs_to_spill.emplace(snippets::RegType::gpr, abi_param_regs[i]);
    }
    // Note: aux_gpr idx must be non-empty because aux_gprs_count() returns 1 for this emitter
    OV_CPU_JIT_EMITTER_ASSERT(aux_gprs_count() >= 1, "Invalid aux_gpr count");
    m_call_address_reg = Reg64(static_cast<int>(aux_gpr_idxs.back()));
    aux_gpr_idxs.pop_back();
    bool spill_required = false;
    m_callee_saved_reg = Reg64(static_cast<int>(get_callee_saved_aux_gpr(aux_gpr_idxs, used_gpr_idxs, spill_required)));
    if (spill_required) {
        m_regs_to_spill.emplace(snippets::RegType::gpr, m_callee_saved_reg.getIdx());
    }
    m_regs_initialized = true;
}

const Xbyak::Reg64& jit_binary_call_emitter::get_call_address_reg() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "You should call init_binary_call_regs() before using this method");
    return m_call_address_reg;
}
const Xbyak::Reg64& jit_binary_call_emitter::get_callee_saved_reg() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "You should call init_binary_call_regs() before using this method");
    return m_callee_saved_reg;
}

const std::set<snippets::Reg>& jit_binary_call_emitter::get_regs_to_spill() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "You should call init_binary_call_regs() before using this method");
    return m_regs_to_spill;
}

}  // namespace ov::intel_cpu
