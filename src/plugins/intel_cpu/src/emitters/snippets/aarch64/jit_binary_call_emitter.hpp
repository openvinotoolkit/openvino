// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <set>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "snippets/emitter.hpp"

namespace ov::intel_cpu::aarch64 {

/**
 * @brief Base class for binary call emitters on ARM64. Provides proper ABI compliance for function calls
 * by managing register spilling, stack alignment, and calling conventions according to ARM64 AAPCS.
 * This follows the same pattern as the x64 jit_binary_call_emitter but adapted for ARM64 architecture.
 */
class jit_binary_call_emitter : public jit_emitter {
public:
    jit_binary_call_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                            dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                            std::set<snippets::Reg> live_regs);

    // Need at least one register to store the callable address
    size_t aux_gprs_count() const {
        return 1;
    }

protected:
    /**
     * @brief Returns a set of snippets::Reg that should be spilled in the derived emitter.
     * This set includes live_regs passed in constructor, plus callee-saved regs and regs for ABI params.
     */
    const std::set<snippets::Reg>& get_regs_to_spill() const;

    /**
     * @brief Returns a GPR that can be used to store the address of the callable.
     */
    const Xbyak_aarch64::XReg& get_call_address_reg() const;

    /**
     * @brief Returns a callee-saved GPR that can be used for stack alignment before the call.
     */
    const Xbyak_aarch64::XReg& get_callee_saved_reg() const;

    /**
     * @brief Initializes registers for binary call emission according to ARM64 AAPCS.
     * @param num_binary_args - the number of arguments of the binary that will be called
     * @param used_gpr_idxs - indices of registers that must be preserved during aux reg allocation
     */
    void init_binary_call_regs(size_t num_binary_args, const std::vector<size_t>& used_gpr_idxs) const;

    /**
     * @brief Overload that takes separate input/output register vectors.
     */
    void init_binary_call_regs(size_t num_binary_args,
                               const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const;

    /**
     * @brief Emit proper stack management for ARM64 function calls.
     * @param stack_size - size of stack space to reserve (must be 16-byte aligned)
     */
    void emit_stack_preserve(size_t stack_size) const;

    /**
     * @brief Restore stack after ARM64 function call.
     * @param stack_size - size of stack space to restore
     */
    void emit_stack_restore(size_t stack_size) const;

private:
    // All mutable because init_binary_call_regs() is called from const emit_impl()
    mutable std::set<snippets::Reg> m_regs_to_spill;
    mutable Xbyak_aarch64::XReg m_callee_saved_reg{31};  // Initialize to invalid reg
    mutable Xbyak_aarch64::XReg m_call_address_reg{31};  // Initialize to invalid reg
    mutable bool m_regs_initialized = false;
};

}  // namespace ov::intel_cpu::aarch64
