// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h>

#include <type_traits>

#ifdef CPU_DEBUG_CAPS

#    include "cpu/aarch64/jit_generator.hpp"

namespace ov::intel_cpu::aarch64 {

// Usage
// 1. Include this header in AArch64 JIT kernels when register printing is required.
// 2. Invoke RegPrinter::print method with a supported register/precision combination.
//    Examples:
//      RegPrinter::print<float>(*this, vmm_val, "vmm_val");
//      RegPrinter::print<int>(*this, vmm_idx);
//      RegPrinter::print<int>(*this, reg_work_amount, "reg_work_amount");
//
// Supported combinations:
//           fp32        int32       int8        u8
// VReg       Yes         Yes        No          No
// XReg       Yes         Yes        No          No
// WReg       Yes         Yes        No          No
class RegPrinter {
public:
    using jit_generator_t = dnnl::impl::cpu::aarch64::jit_generator;

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

    template <typename PRC_T, typename REG_T>
    static void print_vmm(jit_generator_t& h, const REG_T& vmm, const char* name);

    template <typename PRC_T, typename REG_T>
    static void print_reg(jit_generator_t& h, const REG_T& reg, const char* name);

    template <typename PRC_T, size_t vlen>
    static void print_vmm_prc(const char* name, const char* ori_name, PRC_T* ptr);

    template <typename T>
    static void print_reg_prc(const char* name, const char* ori_name, T* ptr);

    static void preamble(jit_generator_t& h);
    static void postamble(jit_generator_t& h);
    static void save_vmm(jit_generator_t& h);
    static void restore_vmm(jit_generator_t& h);
    static void save_reg(jit_generator_t& h);
    static void restore_reg(jit_generator_t& h);
};

}  // namespace ov::intel_cpu::aarch64

#endif  // CPU_DEBUG_CAPS
