// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_generator.hpp"

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

void jit_generator::preamble() {
    const int frame_size = rnd_up((num_abi_save_gpr_regs + 1) * xlen + num_abi_save_fp_gpr_regs * flen, sp_aligment);
    addi(sp, sp, -frame_size);
    int imm = 0;
    for (const auto& gpr : abi_save_gpr_regs) {
        sd(gpr, sp, imm);
        imm += xlen;
    }
    for (const auto& fp_gpr : abi_save_fp_gpr_regs) {
        fsd(fp_gpr, sp, imm);
        imm += flen;
    }
    sd(ra, sp, imm);
}

void jit_generator::postamble() {
    const int frame_size = rnd_up((num_abi_save_gpr_regs + 1) * xlen + num_abi_save_fp_gpr_regs * flen, sp_aligment);
    int imm = 0;
    for (const auto& gpr : abi_save_gpr_regs) {
        ld(gpr, sp, imm);
        imm += xlen;
    }
    for (const auto& fp_gpr : abi_save_fp_gpr_regs) {
        fld(fp_gpr, sp, imm);
        imm += flen;
    }
    ld(ra, sp, imm);

    addi(sp, sp, frame_size);

    ret();
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov