// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h>

#include <type_traits>

#ifdef CPU_DEBUG_CAPS

#    include "cpu/aarch64/jit_generator.hpp"

namespace ov::intel_cpu::aarch64 {

// Usage
// 1. Include this header in AArch64 JIT kernels where register printing is needed.
// 2. Call RegPrinter::print<PRC_T>(h, reg) or RegPrinter::print<PRC_T>(h, reg, "name").
//    The name argument must be a string literal (not a local char* variable) because
//    the pointer is captured at JIT-emit time and dereferenced at JIT-run time.
//
//    Example 1:
//    Invocation: RegPrinter::print<float>(*this, vmm_val, "vmm_val");
//    Console:    vmm_val | v0: {1.5, 2.0, 3.0, 4.0}
//
//    Example 2:
//    Invocation: RegPrinter::print<float>(*this, vmm_val);
//    Console:    v0: {1.5, 2.0, 3.0, 4.0}
//
//    Example 3:
//    Invocation: RegPrinter::print<int>(*this, vmm_idx, "vmm_idx");
//    Console:    vmm_idx | v1: {5, 6, 0, 2}
//
//    Example 4:
//    Invocation: RegPrinter::print<int>(*this, reg_work_amount, "reg_work_amount");
//    Console:    reg_work_amount | x9: 8
//
//    Example 5:
//    Invocation: RegPrinter::print<int>(*this, reg_work_amount);
//    Console:    x9: 8
//
// Supported combinations (VReg prints vec_len/sizeof(PRC_T) elements; XReg/WReg print 1):
//           fp32        int32       int8        u8
// VReg       Yes         Yes         No          No
// XReg       Yes         Yes         No          No
// WReg       Yes         Yes         No          No
class RegPrinter {
public:
    using jit_generator_t = dnnl::impl::cpu::aarch64::jit_generator_t;

    template <typename PRC_T, typename REG_T>
    static void print(jit_generator_t& h, REG_T reg, const char* name = nullptr) {
        if constexpr (std::is_base_of_v<Xbyak_aarch64::VRegVec, REG_T>) {
            print_vmm<PRC_T>(h, reg, name);
        } else {
            print_reg<PRC_T>(h, reg, name);
        }
    }

    static constexpr size_t reg_len = 8;
    static constexpr size_t reg_cnt = 31;
    static constexpr size_t vec_len = 16;
    static constexpr size_t vec_cnt = 32;
    static constexpr size_t sp_alignment = 16;

private:
    RegPrinter() = default;

    template <typename PRC_T, typename REG_T, typename PrinterFunc>
    static void print_reg_common(jit_generator_t& h, const REG_T& reg, const char* name, PrinterFunc printer);

    template <typename PRC_T, typename REG_T>
    static void print_vmm(jit_generator_t& h, const REG_T& vmm, const char* name);

    template <typename PRC_T, typename REG_T>
    static void print_reg(jit_generator_t& h, const REG_T& reg, const char* name);

    static void preamble(jit_generator_t& h);
    static void postamble(jit_generator_t& h);
    static void save_vmm(jit_generator_t& h);
    static void restore_vmm(jit_generator_t& h);
    static void save_reg(jit_generator_t& h);
    static void restore_reg(jit_generator_t& h);
};

}  // namespace ov::intel_cpu::aarch64

#endif  // CPU_DEBUG_CAPS
