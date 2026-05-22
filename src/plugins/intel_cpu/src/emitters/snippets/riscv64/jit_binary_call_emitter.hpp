// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <set>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "snippets/emitter.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_binary_call_emitter : public jit_emitter {
public:
    jit_binary_call_emitter(jit_generator_t* h, cpu_isa_t isa, std::set<snippets::Reg> live_regs);

    size_t aux_gprs_count() const override {
        return 1;
    }

protected:
    void init_binary_call_regs(size_t num_binary_args, const std::vector<size_t>& used_gpr_idxs) const;
    void init_binary_call_regs(size_t num_binary_args,
                               const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const;

    void binary_call_preamble() const;
    void binary_call_postamble() const;

    const Xbyak_riscv::Reg& get_call_address_reg() const;

private:
    std::vector<size_t> get_gpr_regs_to_spill() const;
    std::vector<size_t> get_vec_regs_to_spill() const;

    mutable std::set<snippets::Reg> m_live_regs;
    mutable Xbyak_riscv::Reg m_call_address_reg{Xbyak_riscv::zero};
    mutable bool m_regs_initialized = false;
};

}  // namespace ov::intel_cpu::riscv64
