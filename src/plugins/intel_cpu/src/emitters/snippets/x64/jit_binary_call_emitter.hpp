// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov::intel_cpu {
/**
 * @brief Base class for binary call emitters. Its main function is to allocate 2 auxiliary registers needed for binary
 * call emission: one is any gpr to store callable address, the second one is a callee-saved reg to organize rsp
 * alignment before the call. It also creates a set of registers to spill that can be passed directly to
 * EmitABIRegSpills.
 */
class jit_binary_call_emitter : public jit_emitter {
public:
    jit_binary_call_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            std::set<snippets::Reg> live_regs);
    // Note: we need at least one register to allocate a gpr to store the callable address
    size_t aux_gprs_count() const override {
        return 1;
    }

protected:
    /**
     * @brief Returns a set of snippets::Reg that should be spilled in the derived emitter. This set includes live_regs
     * passed in constructor, plus a callee-saved reg and regs for ABI params. This method can be used only after
     * init_binary_call_regs(...)
     */
    const std::set<snippets::Reg>& get_regs_to_spill() const;
    /**
     * @brief Returns a gpr that can be used to store the address of the callable. This method can be used only after
     * init_binary_call_regs(...)
     */
    const Xbyak::Reg64& get_call_address_reg() const;
    /**
     * @brief Returns a callee-saved gpr that can be used align rsp before the call instruction. This method can be used
     * only after init_binary_call_regs(...)
     */
    const Xbyak::Reg64& get_callee_saved_reg() const;
    /**
     * @brief Initializes registers that can be then obtained via get_regs_to_spill(), get_call_address_reg() or
     * get_callee_saved_reg().
     * @param num_binary_args - the number of arguments of the binary that will be called
     * @param used_gpr_idxs - indices of registers that must be preserved during aux reg allocation, usually in/out
     * memory pointers
     */
    void init_binary_call_regs(size_t num_binary_args, const std::vector<size_t>& used_gpr_idxs) const;
    /**
     * @brief Initializes registers that can be then obtained via get_regs_to_spill(), get_call_address_reg() or
     * get_callee_saved_reg().
     * @param num_binary_args - the number of arguments of the binary that will be called
     * @param in - indices of input registers that must be preserved during aux reg allocation
     * @param out - indices of output registers that must be preserved during aux reg allocation
     */
    void init_binary_call_regs(size_t num_binary_args,
                               const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const;

private:
    // Note: init_regs() can be called only from emit_impl, since it needs initialized regs
    // init_impl is a constant method, so all these fields have to be mutable
    mutable std::set<snippets::Reg> m_regs_to_spill{};
    mutable Xbyak::Reg64 m_callee_saved_reg;
    mutable Xbyak::Reg64 m_call_address_reg;
    mutable bool m_regs_initialized = false;
};

}  // namespace ov::intel_cpu
