// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_binary_call_emitter.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <algorithm>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64 {

jit_binary_call_emitter::jit_binary_call_emitter(jit_generator* h, cpu_isa_t isa, std::set<snippets::Reg> live_regs)
    : jit_emitter(h, isa),
      m_regs_to_spill(std::move(live_regs)) {}

jit_binary_call_emitter::~jit_binary_call_emitter() {
    OPENVINO_DEBUG_ASSERT(
        !m_stack_preserved,
        "Stack preservation mismatch: emit_stack_preserve was called but emit_stack_restore was not called");
}

const std::set<snippets::Reg>& jit_binary_call_emitter::get_regs_to_spill() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");
    return m_regs_to_spill;
}

const Xbyak_aarch64::XReg& jit_binary_call_emitter::get_call_address_reg() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");
    return m_call_address_reg;
}

const Xbyak_aarch64::XReg& jit_binary_call_emitter::get_callee_saved_reg() const {
    OV_CPU_JIT_EMITTER_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");
    return m_callee_saved_reg;
}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& used_gpr_idxs) const {
    if (m_regs_initialized) {
        return;
    }

    // ARM64 AAPCS: X0-X7 are parameter/result registers, X8-X18 are temporary registers
    // X19-X28 are callee-saved registers, X29 is frame pointer, X30 is link register, X31 is stack pointer

    std::vector<size_t> all_used_idxs = used_gpr_idxs;

    // Ensure we have enough registers for binary call arguments
    OV_CPU_JIT_EMITTER_ASSERT(num_binary_args <= 8, "ARM64 ABI supports maximum 8 parameter registers (X0-X7)");

    // Add ABI parameter registers to used list (X0-X7 based on num_binary_args)
    for (size_t i = 0; i < num_binary_args; i++) {
        all_used_idxs.push_back(i);
        if (m_regs_to_spill.find({snippets::RegType::gpr, i}) == m_regs_to_spill.end()) {
            m_regs_to_spill.emplace(snippets::RegType::gpr, i);
        }
    }

    // Add special registers that should not be allocated
    static const std::vector<size_t> reserved_regs = {
        18,  // Platform register (should not be used)
        29,  // Frame pointer (FP)
        30,  // Link register (LR)
        31   // Stack pointer (SP)
    };
    all_used_idxs.insert(all_used_idxs.end(), reserved_regs.begin(), reserved_regs.end());

    // Helper to find first available register in range [start, end)
    auto find_available = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            if (std::find(all_used_idxs.begin(), all_used_idxs.end(), i) == all_used_idxs.end()) {
                return i;
            }
        }
        return SIZE_MAX;
    };

    // Allocate call address register from temporary registers (X8-X18, excluding X18)
    auto call_reg = find_available(8, 18);
    OV_CPU_JIT_EMITTER_ASSERT(call_reg != SIZE_MAX, "No available temporary register for call address");
    m_call_address_reg = Xbyak_aarch64::XReg(static_cast<int>(call_reg));
    all_used_idxs.push_back(call_reg);
    if (m_regs_to_spill.find({snippets::RegType::gpr, call_reg}) == m_regs_to_spill.end()) {
        m_regs_to_spill.emplace(snippets::RegType::gpr, call_reg);
    }

    // Allocate callee-saved register (X19-X28) for stack alignment
    auto callee_reg = find_available(19, 29);
    OV_CPU_JIT_EMITTER_ASSERT(callee_reg != SIZE_MAX, "No available callee-saved register");
    m_callee_saved_reg = Xbyak_aarch64::XReg(static_cast<int>(callee_reg));
    if (m_regs_to_spill.find({snippets::RegType::gpr, callee_reg}) == m_regs_to_spill.end()) {
        m_regs_to_spill.emplace(snippets::RegType::gpr, callee_reg);
    }

    m_regs_initialized = true;
}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& in,
                                                    const std::vector<size_t>& out) const {
    std::vector<size_t> used_gpr_idxs = in;
    used_gpr_idxs.insert(used_gpr_idxs.end(), out.begin(), out.end());
    init_binary_call_regs(num_binary_args, used_gpr_idxs);
}

void jit_binary_call_emitter::emit_stack_preserve(size_t stack_size) const {
    OV_CPU_JIT_EMITTER_ASSERT(!m_stack_preserved, "emit_stack_preserve called twice without emit_stack_restore");

    // ARM64 requires 16-byte stack alignment
    stack_size = ov::intel_cpu::rnd_up(stack_size, sp_alignment);

    if (stack_size > 0) {
        h->sub(h->sp, h->sp, stack_size);
    }

    m_stack_preserved = true;
}

void jit_binary_call_emitter::emit_stack_restore(size_t stack_size) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_stack_preserved, "emit_stack_restore called without corresponding emit_stack_preserve");

    // ARM64 requires 16-byte stack alignment
    stack_size = ov::intel_cpu::rnd_up(stack_size, sp_alignment);

    if (stack_size > 0) {
        h->add(h->sp, h->sp, stack_size);
    }

    m_stack_preserved = false;
}

}  // namespace ov::intel_cpu::aarch64
