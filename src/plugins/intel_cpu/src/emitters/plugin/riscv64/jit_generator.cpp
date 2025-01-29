// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_generator.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

void jit_generator::preamble() {
    // TODO: FP gpr ?
    addi(sp, sp, -num_abi_save_gpr_regs * xlen);
    int imm = 0;
    for (const auto& gpr : abi_save_gpr_regs) {
        sw(gpr, sp, imm);
        imm += 4;
    }
}

void jit_generator::postamble() {
    int imm = 0;
    for (const auto& gpr : abi_save_gpr_regs) {
        lw(gpr, sp, imm);
        imm += 4;
    }
    addi(sp, sp, num_abi_save_gpr_regs * xlen);
    ret();
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov