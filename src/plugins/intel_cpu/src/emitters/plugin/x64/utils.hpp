// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "snippets/emitter.hpp"

namespace ov::intel_cpu {

std::set<size_t> get_callee_saved_reg_idxs();
/**
 * @brief Chooses a callee-saved gpr from the provided pool of registers (`available_gprs`) and removes it from the
 * pool. If there are no callee-saved regs in the pool, chooses any register that is not in the `used_gprs`. Raises
 * exception if it fails to do so.
 * @arg available_gprs - pool of available registers
 * @arg used_gprs - registers that are already in use and should not be selected
 * @arg spill_required - reference to a bool flag that will be set to `true` if spill is required, i.e. the register was
 * not selected from the pool
 * @return reg_idx - idx of callee-saved gpr
 */
size_t get_callee_saved_aux_gpr(std::vector<size_t>& available_gprs,
                                const std::vector<size_t>& used_gprs,
                                bool& spill_required);

// The class emit register spills for the possible call of external binary code
class EmitABIRegSpills {
public:
    EmitABIRegSpills(dnnl::impl::cpu::x64::jit_generator* h);
    ~EmitABIRegSpills();
    [[nodiscard]] size_t get_num_spilled_regs() const {
        return m_regs_to_spill.size();
    }
    /**
     * @brief Spills registers to stack
     * @arg live_regs - set of registers to spill (optional). All registers will be spilled if live_regs is not
     * provided.
     */
    void preamble(const std::set<snippets::Reg>& live_regs = {});
    /**
     * @brief Restores registers previously spilled in preamble(live_regs) call.
     */
    void postamble();

    void rsp_align(size_t callee_saved_gpr_idx);
    void rsp_restore();

private:
    EmitABIRegSpills() = default;
    static dnnl::impl::cpu::x64::cpu_isa_t get_isa();
    dnnl::impl::cpu::x64::jit_generator* h{nullptr};
    const dnnl::impl::cpu::x64::cpu_isa_t isa{dnnl::impl::cpu::x64::cpu_isa_t::isa_undef};
    std::vector<Xbyak::Reg> m_regs_to_spill;
    Xbyak::Reg m_rsp_align_reg;
    uint32_t m_bytes_to_spill = 0;

    bool spill_status = true;
    bool rsp_status = true;
};

}  // namespace ov::intel_cpu
