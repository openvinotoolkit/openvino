// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#    include "debug_capabilities.hpp"

#    include <algorithm>
#    include <cstddef>
#    include <cstdint>

#    include "emitters/plugin/common/debug_utils.hpp"
#    include "nodes/kernels/riscv64/jit_generator.hpp"
#    include "utils/general_utils.h"
#    include "xbyak_riscv/xbyak_riscv.hpp"
#    include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;

namespace {

constexpr size_t sp_alignment = 16;
constexpr size_t gpr_len = 8;
constexpr int sp_idx = Xbyak_riscv::sp.getIdx();
constexpr int gpr_cnt = 32;
constexpr int vec_cnt = 32;

inline uintptr_t to_uintptr(const char* ptr) {
    return reinterpret_cast<uintptr_t>(ptr);
}

constexpr uintptr_t to_uintptr(std::nullptr_t) {
    return 0U;
}

size_t vlen_bytes() {
    return Xbyak_riscv::CPU().getVlen() / 8;
}

size_t gpr_save_bytes() {
    return (gpr_cnt - 1) * gpr_len;
}

size_t saved_vl_offset() {
    return gpr_save_bytes();
}

size_t saved_vtype_offset() {
    return gpr_save_bytes() + gpr_len;
}

size_t vec_save_offset() {
    return gpr_save_bytes() + 2 * gpr_len;
}

size_t ctx_save_bytes() {
    return ov::intel_cpu::rnd_up(vec_save_offset() + vec_cnt * vlen_bytes(), sp_alignment);
}

void sub_sp(jit_generator_t& h, size_t bytes) {
    // addi supports signed 12-bit immediates [-2048..2047]
    while (bytes > 0) {
        const int32_t chunk = static_cast<int32_t>(std::min<size_t>(bytes, 2048));
        h.addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -chunk);
        bytes -= chunk;
    }
}

void add_sp(jit_generator_t& h, size_t bytes) {
    while (bytes > 0) {
        const int32_t chunk = static_cast<int32_t>(std::min<size_t>(bytes, 2047));
        h.addi(Xbyak_riscv::sp, Xbyak_riscv::sp, chunk);
        bytes -= chunk;
    }
}

void save_gprs(jit_generator_t& h) {
    size_t off = 0;
    for (int idx = 0; idx < gpr_cnt; ++idx) {
        if (idx == sp_idx) {
            continue;
        }
        h.sd(Xbyak_riscv::Reg(idx), Xbyak_riscv::sp, static_cast<int32_t>(off));
        off += gpr_len;
    }
}

void restore_gprs(jit_generator_t& h) {
    size_t off = 0;
    for (int idx = 0; idx < gpr_cnt; ++idx) {
        if (idx == sp_idx) {
            continue;
        }
        h.ld(Xbyak_riscv::Reg(idx), Xbyak_riscv::sp, static_cast<int32_t>(off));
        off += gpr_len;
    }
}

void save_vector_state(jit_generator_t& h, const Xbyak_riscv::Reg& tmp0, const Xbyak_riscv::Reg& tmp1) {
    h.csrr(tmp0, Xbyak_riscv::CSR::vl);
    h.csrr(tmp1, Xbyak_riscv::CSR::vtype);
    h.sd(tmp0, Xbyak_riscv::sp, static_cast<int32_t>(saved_vl_offset()));
    h.sd(tmp1, Xbyak_riscv::sp, static_cast<int32_t>(saved_vtype_offset()));
}

void restore_vector_state(jit_generator_t& h, const Xbyak_riscv::Reg& tmp0, const Xbyak_riscv::Reg& tmp1) {
    h.ld(tmp0, Xbyak_riscv::sp, static_cast<int32_t>(saved_vl_offset()));
    h.ld(tmp1, Xbyak_riscv::sp, static_cast<int32_t>(saved_vtype_offset()));
    h.vsetvl(Xbyak_riscv::zero, tmp0, tmp1);
}

void save_vregs(jit_generator_t& h, const Xbyak_riscv::Reg& vlen_gpr, const Xbyak_riscv::Reg& ptr_gpr) {
    const auto vlen = vlen_bytes();
    h.uni_li(vlen_gpr, vlen);
    h.vsetvli(Xbyak_riscv::zero, vlen_gpr, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

    h.addi(ptr_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vec_save_offset()));
    for (int idx = 0; idx < vec_cnt; ++idx) {
        h.vse8_v(Xbyak_riscv::VReg(idx), ptr_gpr);
        h.add(ptr_gpr, ptr_gpr, vlen_gpr);
    }
}

void restore_vregs(jit_generator_t& h, const Xbyak_riscv::Reg& vlen_gpr, const Xbyak_riscv::Reg& ptr_gpr) {
    const auto vlen = vlen_bytes();
    h.uni_li(vlen_gpr, vlen);
    h.vsetvli(Xbyak_riscv::zero, vlen_gpr, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

    h.addi(ptr_gpr, Xbyak_riscv::sp, static_cast<int32_t>(vec_save_offset()));
    for (int idx = 0; idx < vec_cnt; ++idx) {
        h.vle8_v(Xbyak_riscv::VReg(idx), ptr_gpr);
        h.add(ptr_gpr, ptr_gpr, vlen_gpr);
    }
}

}  // namespace

void RegPrinter::preamble(jit_generator_t& h) {
    sub_sp(h, ctx_save_bytes());
    save_gprs(h);
    save_vector_state(h, Xbyak_riscv::t0, Xbyak_riscv::t1);
    save_vregs(h, Xbyak_riscv::t0, Xbyak_riscv::t1);
}

void RegPrinter::postamble(jit_generator_t& h) {
    restore_vregs(h, Xbyak_riscv::t0, Xbyak_riscv::t1);
    restore_vector_state(h, Xbyak_riscv::t0, Xbyak_riscv::t1);
    restore_gprs(h);
    add_sp(h, ctx_save_bytes());
}

template <typename PRC_T>
void RegPrinter::print_vmm(jit_generator_t& h, const Xbyak_riscv::VReg& vmm, const char* name) {
    preamble(h);

    const auto vlen = vlen_bytes();
    const size_t stack_bytes = ov::intel_cpu::rnd_up(vlen, sp_alignment);
    sub_sp(h, stack_bytes);

    h.uni_li(Xbyak_riscv::t0, vlen);
    h.vsetvli(Xbyak_riscv::zero, Xbyak_riscv::t0, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);
    h.vse8_v(vmm, Xbyak_riscv::sp);

    h.uni_li(Xbyak_riscv::a0, name ? to_uintptr(name) : to_uintptr(nullptr));
    h.uni_li(Xbyak_riscv::a1, to_uintptr(vmm.toString()));
    h.mv(Xbyak_riscv::a2, Xbyak_riscv::sp);
    h.uni_li(Xbyak_riscv::a3, vlen);

    auto printer = &ov::intel_cpu::debug_utils::print_vmm_prc_runtime<PRC_T>;
    h.uni_li(Xbyak_riscv::t0, reinterpret_cast<size_t>(printer));
    h.jalr(Xbyak_riscv::t0);

    add_sp(h, stack_bytes);
    postamble(h);
}

template <typename PRC_T>
void RegPrinter::print_reg(jit_generator_t& h, const Xbyak_riscv::Reg& reg, const char* name) {
    preamble(h);

    const size_t stack_bytes = ov::intel_cpu::rnd_up(gpr_len, sp_alignment);
    sub_sp(h, stack_bytes);
    h.sd(reg, Xbyak_riscv::sp, 0);

    h.uni_li(Xbyak_riscv::a0, name ? to_uintptr(name) : to_uintptr(nullptr));
    h.uni_li(Xbyak_riscv::a1, to_uintptr(reg.toString()));
    h.mv(Xbyak_riscv::a2, Xbyak_riscv::sp);

    auto printer = &ov::intel_cpu::debug_utils::print_reg_prc<PRC_T>;
    h.uni_li(Xbyak_riscv::t0, reinterpret_cast<size_t>(printer));
    h.jalr(Xbyak_riscv::t0);

    add_sp(h, stack_bytes);
    postamble(h);
}

template void RegPrinter::print<float, Xbyak_riscv::VReg>(jit_generator_t& h, Xbyak_riscv::VReg reg, const char* name);
template void RegPrinter::print<int, Xbyak_riscv::VReg>(jit_generator_t& h, Xbyak_riscv::VReg reg, const char* name);
template void RegPrinter::print<float, Xbyak_riscv::Reg>(jit_generator_t& h, Xbyak_riscv::Reg reg, const char* name);
template void RegPrinter::print<int, Xbyak_riscv::Reg>(jit_generator_t& h, Xbyak_riscv::Reg reg, const char* name);

}  // namespace ov::intel_cpu::riscv64

#endif  // CPU_DEBUG_CAPS
