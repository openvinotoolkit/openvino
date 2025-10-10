// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#    include "debug_capabilities.hpp"

#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#    include <array>
#    include <climits>
#    include <cstddef>
#    include <cstdint>
#    include <type_traits>

#    include "cpu/aarch64/jit_generator.hpp"
#    include "emitters/plugin/common/debug_utils.hpp"
#    include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::aarch64;

namespace {
constexpr std::array<const char*, RegPrinter::vec_cnt> vreg_names = {
    "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",  "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"};

constexpr std::array<const char*, RegPrinter::reg_cnt> xreg_names = {
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",  "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30"};

constexpr std::array<const char*, RegPrinter::reg_cnt> wreg_names = {
    "w0",  "w1",  "w2",  "w3",  "w4",  "w5",  "w6",  "w7",  "w8",  "w9",  "w10", "w11", "w12", "w13", "w14", "w15",
    "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30"};

inline const char* get_original_name(const VReg& reg) {
    return vreg_names[reg.getIdx()];
}

inline const char* get_original_name(const XReg& reg) {
    return xreg_names[reg.getIdx()];
}

inline const char* get_original_name(const WReg& reg) {
    return wreg_names[reg.getIdx()];
}

inline uintptr_t to_uintptr(const char* ptr) {
    return reinterpret_cast<uintptr_t>(ptr);
}

constexpr uintptr_t to_uintptr(std::nullptr_t) {
    return 0U;
}

}  // namespace

template <typename PRC_T, typename REG_T, typename PrinterFunc>
void RegPrinter::print_reg_common(jit_generator_t& h, const REG_T& reg, const char* name, PrinterFunc printer) {
    preamble(h);

    const size_t reg_bytes = reg.getBit() / CHAR_BIT;
    const size_t stack_bytes = ov::intel_cpu::rnd_up(reg_bytes, sp_alignment);

    h.sub(h.sp, h.sp, stack_bytes);

    if constexpr (std::is_same_v<REG_T, VReg>) {
        h.str(QReg(reg.getIdx()), ptr(h.sp));
    } else if constexpr (std::is_same_v<REG_T, XReg>) {
        h.str(XReg(reg.getIdx()), ptr(h.sp));
    } else if constexpr (std::is_same_v<REG_T, WReg>) {
        h.str(WReg(reg.getIdx()), ptr(h.sp));
    } else {
        static_assert(std::is_same_v<REG_T, VReg> || std::is_same_v<REG_T, XReg> || std::is_same_v<REG_T, WReg>,
                      "Unsupported register type");
    }

    h.mov(abi_param3, h.sp);
    h.mov(abi_param2, to_uintptr(get_original_name(reg)));
    h.mov(abi_param1, name ? to_uintptr(name) : to_uintptr(nullptr));

    h.mov(h.X_TMP_0, reinterpret_cast<uintptr_t>(printer));
    h.blr(h.X_TMP_0);

    h.add(h.sp, h.sp, stack_bytes);

    postamble(h);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_vmm(jit_generator_t& h, const REG_T& vmm, const char* name) {
    auto printer = &ov::intel_cpu::debug_utils::print_vmm_prc<PRC_T, RegPrinter::vec_len>;
    print_reg_common<PRC_T>(h, vmm, name, printer);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_reg(jit_generator_t& h, const REG_T& reg, const char* name) {
    auto printer = &ov::intel_cpu::debug_utils::print_reg_prc<PRC_T>;
    print_reg_common<PRC_T>(h, reg, name, printer);
}

void RegPrinter::preamble(jit_generator_t& h) {
    save_reg(h);
    save_vmm(h);
}

void RegPrinter::postamble(jit_generator_t& h) {
    restore_vmm(h);
    restore_reg(h);
}

void RegPrinter::save_reg(jit_generator_t& h) {
    const size_t total_size = ov::intel_cpu::rnd_up(reg_len * reg_cnt, sp_alignment);
    h.sub(h.sp, h.sp, total_size);

    size_t offset = 0;
    for (size_t idx = 0; idx + 1 < reg_cnt; idx += 2) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.stp(XReg(idx), XReg(idx + 1), ptr(h.sp, current_offset));
        offset += 2 * reg_len;
    }
    if (reg_cnt % 2 != 0) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.str(XReg(reg_cnt - 1), ptr(h.sp, current_offset));
    }
}

void RegPrinter::restore_reg(jit_generator_t& h) {
    size_t offset = 0;
    for (size_t idx = 0; idx + 1 < reg_cnt; idx += 2) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.ldp(XReg(idx), XReg(idx + 1), ptr(h.sp, current_offset));
        offset += 2 * reg_len;
    }
    if (reg_cnt % 2 != 0) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.ldr(XReg(reg_cnt - 1), ptr(h.sp, current_offset));
    }

    const size_t total_size = ov::intel_cpu::rnd_up(reg_len * reg_cnt, sp_alignment);
    h.add(h.sp, h.sp, total_size);
}

void RegPrinter::save_vmm(jit_generator_t& h) {
    const size_t total_size = ov::intel_cpu::rnd_up(vec_len * vec_cnt, sp_alignment);
    h.sub(h.sp, h.sp, total_size);

    size_t offset = 0;
    for (size_t idx = 0; idx < vec_cnt; idx += 2) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.stp(QReg(idx), QReg(idx + 1), ptr(h.sp, current_offset));
        offset += 2 * vec_len;
    }
}

void RegPrinter::restore_vmm(jit_generator_t& h) {
    size_t offset = 0;
    for (size_t idx = 0; idx < vec_cnt; idx += 2) {
        const auto current_offset = static_cast<int32_t>(offset);
        h.ldp(QReg(idx), QReg(idx + 1), ptr(h.sp, current_offset));
        offset += 2 * vec_len;
    }

    const size_t total_size = ov::intel_cpu::rnd_up(vec_len * vec_cnt, sp_alignment);
    h.add(h.sp, h.sp, total_size);
}

template void RegPrinter::print<float, VReg>(jit_generator_t& h, VReg reg, const char* name);
template void RegPrinter::print<int, VReg>(jit_generator_t& h, VReg reg, const char* name);
template void RegPrinter::print<float, XReg>(jit_generator_t& h, XReg reg, const char* name);
template void RegPrinter::print<int, XReg>(jit_generator_t& h, XReg reg, const char* name);
template void RegPrinter::print<float, WReg>(jit_generator_t& h, WReg reg, const char* name);
template void RegPrinter::print<int, WReg>(jit_generator_t& h, WReg reg, const char* name);

}  // namespace ov::intel_cpu::aarch64

#endif  // CPU_DEBUG_CAPS
