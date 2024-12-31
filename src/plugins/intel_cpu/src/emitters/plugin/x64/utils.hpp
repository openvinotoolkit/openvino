// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "snippets/emitter.hpp"

namespace ov {
namespace intel_cpu {

// The class emit register spills for the possible call of external binary code
class EmitABIRegSpills {
public:
    EmitABIRegSpills(dnnl::impl::cpu::x64::jit_generator* h);
    ~EmitABIRegSpills();
    size_t get_num_spilled_regs() const {
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

    // align stack on 16-byte and allocate shadow space as ABI reqiures
    // callee is responsible to save and restore `rbx`. `rbx` must not be changed after call callee.
    void rsp_align();
    void rsp_restore();

private:
    EmitABIRegSpills() = default;
    static dnnl::impl::cpu::x64::cpu_isa_t get_isa();
    dnnl::impl::cpu::x64::jit_generator* h{nullptr};
    const dnnl::impl::cpu::x64::cpu_isa_t isa{dnnl::impl::cpu::x64::cpu_isa_t::isa_undef};
    std::vector<Xbyak::Reg> m_regs_to_spill;
    uint32_t m_bytes_to_spill = 0;

    bool spill_status = true;
    bool rsp_status = true;
};

}  // namespace intel_cpu
}  // namespace ov
