// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_context_helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64::utils {

size_t get_vlen_bytes() {
    return Xbyak_riscv::CPU().getVlen() / 8;
}

void sub_sp(jit_generator_t& h, size_t bytes) {
    // addi supports signed 12-bit immediates [-2048..2047].
    while (bytes > 0) {
        const auto chunk = static_cast<int32_t>(std::min<size_t>(bytes, 2048));
        h.addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -chunk);
        bytes -= chunk;
    }
}

void add_sp(jit_generator_t& h, size_t bytes) {
    while (bytes > 0) {
        const auto chunk = static_cast<int32_t>(std::min<size_t>(bytes, 2047));
        h.addi(Xbyak_riscv::sp, Xbyak_riscv::sp, chunk);
        bytes -= chunk;
    }
}

void save_vector_state(jit_generator_t& h,
                       const Xbyak_riscv::Reg& vl_gpr,
                       const Xbyak_riscv::Reg& vtype_gpr,
                       size_t vl_offset,
                       size_t vtype_offset) {
    h.csrr(vl_gpr, Xbyak_riscv::CSR::vl);
    h.csrr(vtype_gpr, Xbyak_riscv::CSR::vtype);
    h.sd(vl_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vl_offset));
    h.sd(vtype_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vtype_offset));
}

void restore_vector_state(jit_generator_t& h,
                          const Xbyak_riscv::Reg& vl_gpr,
                          const Xbyak_riscv::Reg& vtype_gpr,
                          size_t vl_offset,
                          size_t vtype_offset) {
    h.ld(vl_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vl_offset));
    h.ld(vtype_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vtype_offset));
    h.vsetvl(Xbyak_riscv::zero, vl_gpr, vtype_gpr);
}

void save_vregs(jit_generator_t& h,
                const Xbyak_riscv::Reg& vlen_gpr,
                const Xbyak_riscv::Reg& ptr_gpr,
                size_t stack_offset,
                const std::vector<size_t>& vreg_idxs) {
    const auto vlen = get_vlen_bytes();
    h.uni_li(vlen_gpr, vlen);
    h.vsetvli(Xbyak_riscv::zero, vlen_gpr, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

    h.addi(ptr_gpr, Xbyak_riscv::sp, static_cast<int32_t>(stack_offset));
    for (const auto& idx : vreg_idxs) {
        h.vse8_v(Xbyak_riscv::VReg(static_cast<int>(idx)), ptr_gpr);
        h.add(ptr_gpr, ptr_gpr, vlen_gpr);
    }
}

void restore_vregs(jit_generator_t& h,
                   const Xbyak_riscv::Reg& vlen_gpr,
                   const Xbyak_riscv::Reg& ptr_gpr,
                   size_t stack_offset,
                   const std::vector<size_t>& vreg_idxs) {
    const auto vlen = get_vlen_bytes();
    h.uni_li(vlen_gpr, vlen);
    h.vsetvli(Xbyak_riscv::zero, vlen_gpr, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

    h.addi(ptr_gpr, Xbyak_riscv::sp, static_cast<int32_t>(stack_offset));
    for (const auto& idx : vreg_idxs) {
        h.vle8_v(Xbyak_riscv::VReg(static_cast<int>(idx)), ptr_gpr);
        h.add(ptr_gpr, ptr_gpr, vlen_gpr);
    }
}

}  // namespace ov::intel_cpu::riscv64::utils
