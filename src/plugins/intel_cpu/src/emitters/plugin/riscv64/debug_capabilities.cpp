// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#    include "debug_capabilities.hpp"

#    include <cstddef>
#    include <cstdint>
#    include <numeric>
#    include <vector>

#    include "emitters/plugin/common/debug_utils.hpp"
#    include "emitters/plugin/riscv64/jit_context_helpers.hpp"
#    include "nodes/kernels/riscv64/jit_generator.hpp"
#    include "utils/general_utils.h"
#    include "xbyak_riscv/xbyak_riscv.hpp"

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

size_t gpr_save_bytes() {
    return (gpr_cnt - 1) * gpr_len;
}

size_t saved_vl_offset() {
    return gpr_save_bytes();
}

size_t saved_vtype_offset() {
    return gpr_save_bytes() + gpr_len;
}

size_t vregs_save_offset() {
    return gpr_save_bytes() + 2 * gpr_len;
}

size_t ctx_save_bytes() {
    return ov::intel_cpu::rnd_up(vregs_save_offset() + vec_cnt * utils::get_vlen_bytes(), sp_alignment);
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

std::vector<size_t> get_all_vreg_idxs() {
    std::vector<size_t> idxs(vec_cnt);
    std::iota(idxs.begin(), idxs.end(), size_t{0});
    return idxs;
}

}  // namespace

void RegPrinter::preamble(jit_generator_t& h) {
    utils::sub_sp(h, ctx_save_bytes());
    save_gprs(h);
    utils::save_vector_state(h, Xbyak_riscv::t0, Xbyak_riscv::t1, saved_vl_offset(), saved_vtype_offset());
    utils::save_vregs(h, Xbyak_riscv::t0, Xbyak_riscv::t1, vregs_save_offset(), get_all_vreg_idxs());
}

void RegPrinter::postamble(jit_generator_t& h) {
    utils::restore_vregs(h, Xbyak_riscv::t0, Xbyak_riscv::t1, vregs_save_offset(), get_all_vreg_idxs());
    utils::restore_vector_state(h, Xbyak_riscv::t0, Xbyak_riscv::t1, saved_vl_offset(), saved_vtype_offset());
    restore_gprs(h);
    utils::add_sp(h, ctx_save_bytes());
}

template <typename PRC_T>
void RegPrinter::print_vmm(jit_generator_t& h, const Xbyak_riscv::VReg& vmm, const char* name) {
    preamble(h);

    const auto vlen = utils::get_vlen_bytes();
    const size_t stack_bytes = ov::intel_cpu::rnd_up(vlen, sp_alignment);
    utils::sub_sp(h, stack_bytes);

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

    utils::add_sp(h, stack_bytes);
    postamble(h);
}

template <typename PRC_T>
void RegPrinter::print_reg(jit_generator_t& h, const Xbyak_riscv::Reg& reg, const char* name) {
    preamble(h);

    const size_t stack_bytes = ov::intel_cpu::rnd_up(gpr_len, sp_alignment);
    utils::sub_sp(h, stack_bytes);
    h.sd(reg, Xbyak_riscv::sp, 0);

    h.uni_li(Xbyak_riscv::a0, name ? to_uintptr(name) : to_uintptr(nullptr));
    h.uni_li(Xbyak_riscv::a1, to_uintptr(reg.toString()));
    h.mv(Xbyak_riscv::a2, Xbyak_riscv::sp);

    auto printer = &ov::intel_cpu::debug_utils::print_reg_prc<PRC_T>;
    h.uni_li(Xbyak_riscv::t0, reinterpret_cast<size_t>(printer));
    h.jalr(Xbyak_riscv::t0);

    utils::add_sp(h, stack_bytes);
    postamble(h);
}

template void RegPrinter::print<float, Xbyak_riscv::VReg>(jit_generator_t& h, Xbyak_riscv::VReg reg, const char* name);
template void RegPrinter::print<int, Xbyak_riscv::VReg>(jit_generator_t& h, Xbyak_riscv::VReg reg, const char* name);
template void RegPrinter::print<float, Xbyak_riscv::Reg>(jit_generator_t& h, Xbyak_riscv::Reg reg, const char* name);
template void RegPrinter::print<int, Xbyak_riscv::Reg>(jit_generator_t& h, Xbyak_riscv::Reg reg, const char* name);

}  // namespace ov::intel_cpu::riscv64

#endif  // CPU_DEBUG_CAPS
