// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "openvino/core/except.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using namespace Xbyak_riscv;

void jit_generator_t::preamble() {
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

void jit_generator_t::postamble() {
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

void jit_generator_t::uni_li(const Reg& rd, size_t value) {
    // We have to decompose pseudo-instruction `li` into several small instructions because
    // the immediate field in RISC-V instructions is limited to 12 or 20 bits:
    // - use `lui` (Load Upper Immediate) for high bits.
    // - use `addi` (Load Upper Immediate) for low bits.
    // - use `slli` to combine parts.

    // Check that value is 32-bit value
    if (static_cast<uint64_t>(static_cast<int64_t>(value << 32) >> 32) == value) {
        const auto value32 = static_cast<uint32_t>(value);
        if (value32 == 0) {
            mv(rd, zero);
            return;
        }

        // Add 0x800 to cancel out the signed extension of ADDIW.
        const auto upper_20 = (value32 + 0x800) >> 12 & 0xFFFFF;
        int32_t lower_12 = static_cast<int32_t>(value32) & 0xFFF;
        // Convert to signed 12-bit
        if (lower_12 > 2047) {
            lower_12 -= 4096;
        }
        if (lower_12 < -2048) {
            lower_12 += 4096;
        }

        if (upper_20 != 0) {
            lui(rd, upper_20);
        }

        if (lower_12 != 0) {
            auto src = upper_20 == 0 ? zero : rd;
            addi(rd, src, lower_12);
        }

        return;
    }

    auto trailing_zero = [](uint64_t value) {
        uint32_t bits = 0;
        if (value == 0) {
            return bits;
        }
        while ((value & 1) == 0) {
            bits++;
            value >>= 1;
        }
        return bits;
    };

    auto lower_12 = static_cast<int32_t>(static_cast<int64_t>(value << 52) >> 52);
    // Convert to signed 12-bit
    if (lower_12 > 2047) {
        lower_12 -= 4096;
    }
    if (lower_12 < -2048) {
        lower_12 += 4096;
    }

    // Add 0x800 to cancel out the signed extension of ADDI.
    uint64_t upper_52 = (value + 0x800) >> 12;
    const uint32_t shift = 12 + trailing_zero(upper_52);
    upper_52 = static_cast<uint64_t>((static_cast<int64_t>(upper_52 >> (shift - 12)) << shift) >> shift);

    uni_li(rd, upper_52);
    slli(rd, rd, shift);
    if (lower_12 != 0) {
        addi(rd, rd, lower_12);
    }
}

void jit_generator_t::vfneg_vv(const Xbyak_riscv::VReg& vd, const Xbyak_riscv::VReg& vs, Xbyak_riscv::VM vm) {
    vfsgnjn_vv(vd, vs, vs, vm);
}

Xbyak_riscv::LMUL jit_generator_t::float2lmul(const float lmul) {
    if (lmul == 0.125F) {
        return LMUL::mf8;
    }
    if (lmul == 0.25F) {
        return LMUL::mf4;
    }
    if (lmul == 0.5F) {
        return LMUL::mf2;
    }
    if (lmul == 1.F) {
        return LMUL::m1;
    }
    if (lmul == 2.F) {
        return LMUL::m2;
    }
    if (lmul == 4.F) {
        return LMUL::m4;
    }
    if (lmul == 8.F) {
        return LMUL::m8;
    }
    OPENVINO_THROW(std::string("not supported vector length multiplier: ") + std::to_string(lmul));
}

float jit_generator_t::lmul2float(const LMUL lmul) {
    switch (lmul) {
    case LMUL::mf8:
        return 0.125F;
    case LMUL::mf4:
        return 0.25F;
    case LMUL::mf2:
        return 0.5F;
    case LMUL::m1:
        return 1;
    case LMUL::m2:
        return 2;
    case LMUL::m4:
        return 4;
    case LMUL::m8:
        return 8;
    default: {
        OPENVINO_THROW(std::string("not supported vector length multiplier: ") +
                       std::to_string(static_cast<uint32_t>(lmul)));
    }
    }
}

Xbyak_riscv::SEW jit_generator_t::bytes2sew(const size_t sew) {
    switch (sew) {
    case 1LU:
        return SEW::e8;
    case 2LU:
        return SEW::e16;
    case 4LU:
        return SEW::e32;
    case 8LU:
        return SEW::e64;
    default: {
        OPENVINO_THROW(std::string("not supported sew: ") + std::to_string(sew));
    }
    }
}

size_t jit_generator_t::sew2bytes(const Xbyak_riscv::SEW sew) {
    switch (sew) {
    case SEW::e8:
        return 1LU;
    case SEW::e16:
        return 2LU;
    case SEW::e32:
        return 4LU;
    case SEW::e64:
        return 8LU;
    default: {
        OPENVINO_THROW(std::string("not supported sew: ") + std::to_string(static_cast<uint32_t>(sew)));
    }
    }
}

}  // namespace ov::intel_cpu::riscv64
