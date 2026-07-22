// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::utils {

jit_aux_gpr_holder::jit_aux_gpr_holder(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       std::vector<size_t>& pool_gpr_idxs,
                                       const std::vector<size_t>& used_gpr_idxs)
    : m_h(host),
      m_pool_gpr_idxs(pool_gpr_idxs) {
    if (m_pool_gpr_idxs.empty()) {
        // choose an available caller-saved reg not in used set
        m_reg = ov::intel_cpu::riscv64::utils::get_aux_gpr(used_gpr_idxs);
        m_preserved = true;
        // Maintain 16-byte alignment; reserve 16 bytes and save at 0
        m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -16);
        m_h->sd(m_reg, Xbyak_riscv::sp, 0);
    } else {
        m_reg = Xbyak_riscv::Reg(static_cast<int>(m_pool_gpr_idxs.back()));
        m_pool_gpr_idxs.pop_back();
    }
}

jit_aux_gpr_holder::~jit_aux_gpr_holder() {
    if (m_preserved) {
        m_h->ld(m_reg, Xbyak_riscv::sp, 0);
        m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, 16);
    } else {
        m_pool_gpr_idxs.push_back(static_cast<size_t>(m_reg.getIdx()));
    }
}

Xbyak_riscv::Reg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    // RISC-V reserved registers to avoid: x0(zero), x1(ra), x2(sp), x3(gp), x4(tp), x8(s0/fp)
    // Also avoid a0, a1 which are used for ABI parameters
    const std::set<size_t> reserved_regs = {0, 1, 2, 3, 4, 8, 10, 11};

    // Start with temporary registers t0-t6 (x5-x7, x28-x31)
    const std::vector<size_t> temp_regs = {5, 6, 7, 28, 29, 30, 31};

    for (size_t reg_idx : temp_regs) {
        if (std::find(used_gpr_idxs.begin(), used_gpr_idxs.end(), reg_idx) == used_gpr_idxs.end()) {
            return Xbyak_riscv::Reg(static_cast<int>(reg_idx));
        }
    }

    // If no temporary registers available, try saved registers s1-s11 (x9, x18-x27)
    const std::vector<size_t> saved_regs = {9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    for (size_t reg_idx : saved_regs) {
        if (std::find(used_gpr_idxs.begin(), used_gpr_idxs.end(), reg_idx) == used_gpr_idxs.end()) {
            return Xbyak_riscv::Reg(static_cast<int>(reg_idx));
        }
    }

    OPENVINO_THROW("No available auxiliary GPR registers");
}

}  // namespace ov::intel_cpu::riscv64::utils
