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

Xbyak_riscv::LMUL jit_generator::float2lmul(const float lmul) const {
    if (lmul == 0.125f) return LMUL::mf8;
    if (lmul == 0.25f) return LMUL::mf4;
    if (lmul == 0.5f) return LMUL::mf2;
    if (lmul == 1.f) return LMUL::m1;
    if (lmul == 2.f) return LMUL::m2;
    if (lmul == 4.f) return LMUL::m4;
    if (lmul == 8.f) return LMUL::m8;
    OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(lmul));
}

float jit_generator::lmul2float(const LMUL lmul) const {
    switch (lmul) {
    case LMUL::mf8: return 0.125f;
    case LMUL::mf4: return 0.25f;
    case LMUL::mf2: return 0.5f;
    case LMUL::m1: return 1;
    case LMUL::m2: return 2;
    case LMUL::m4: return 4;
    case LMUL::m8: return 8;
    default: {
        OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(static_cast<uint32_t>(lmul)));
    }
    }
}

Xbyak_riscv::SEW jit_generator::bytes2sew(const size_t sew) const {
    switch(sew) {
    case 1lu: return SEW::e8;
    case 2lu: return SEW::e16;
    case 4lu: return SEW::e32;
    case 8lu: return SEW::e64;
    default: {
        OPENVINO_THROW(std::string("not supported sew: ") + std::to_string(sew));
    }
    }
}

size_t jit_generator::sew2bytes(const Xbyak_riscv::SEW sew) const {
    switch(sew) {
    case SEW::e8: return 1lu;
    case SEW::e16: return 2lu;
    case SEW::e32: return 4lu;
    case SEW::e64: return 8lu;
    default: {
        OPENVINO_THROW(std::string("not supported sew: ") + std::to_string(static_cast<uint32_t>(sew)));
    }
    }
}


}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov