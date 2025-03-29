// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef CPU_DEBUG_CAPS

#    include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu {

// Usage
// 1. Include this headfile where JIT kennels of CPU plugin are implemented for Register printing
// 2. Invoke RegPrinter::print method. Here are some examples. Note that user friendly register name
//    will be printed, if it has been set. Current implementation doesn't buffer the name. So if you
//    choose to set a name for the register, do not use local variable to pass the name, just pass a
//    direct string to the interface like examples. While Original Xbyak register name will always be
//    printed.
//    Example 1:
//    Invocation: RegPrinter::print<float>(*this, vmm_val, "vmm_val");
//    Console:    vmm_val | ymm0: {30, 20, 25, 29, 24, 31, 27, 23}
//
//    Example 2:
//    Invocation: RegPrinter::print<float>(*this, vmm_val);
//    Console:    ymm0: {30, 20, 25, 29, 24, 31, 27, 23}
//
//    Example 3:
//    Invocation: RegPrinter::print<int>(*this, vmm_idx, "vmm_idx");
//    Console:    vmm_idx | ymm1: {5, 6, 0, 2, 0, 6, 6, 6}
//
//    Example 4:
//    Invocation: RegPrinter::print<int>(*this, reg_work_amount, "reg_work_amount");
//    Console:    reg_work_amount | r13: 8
//
//    Example 5:
//    Invocation: RegPrinter::print<int>(*this, reg_work_amount);
//    Console:    r13: 8
//
//    Example 6:
//    Invocation: RegPrinter::print<float>(*this, reg_tmp_64, "reg_tmp_64");
//    Console:    reg_tmp_64 | r15: 1
//
// Parameter
// The following combinations of Register types and precisions are supported.
//          fp32        int32       int8        u8
// Xmm      Yes         Yes         No          No
// Ymm      Yes         Yes         No          No
// Zmm      Yes         Yes         No          No
// Reg64    Yes         Yes         No          No
// Reg32    Yes         Yes         No          No
// Reg16    No          No          Yes         Yes
// Reg8     No          No          Yes         Yes

class RegPrinter {
public:
    using jit_generator = dnnl::impl::cpu::x64::jit_generator;
    template <typename PRC_T, typename REG_T, std::enable_if_t<std::is_base_of_v<Xbyak::Xmm, REG_T>, int> = 0>
    static void print(jit_generator& h, REG_T reg, const char* name = nullptr) {
        print_vmm<PRC_T, REG_T>(h, reg, name);
    }
    template <typename PRC_T, typename REG_T, std::enable_if_t<!std::is_base_of_v<Xbyak::Xmm, REG_T>, int> = 0>
    static void print(jit_generator& h, REG_T reg, const char* name = nullptr) {
        print_reg<PRC_T, REG_T>(h, reg, name);
    }

private:
    RegPrinter() {}
    template <typename PRC_T, typename REG_T>
    static void print_vmm(jit_generator& h, REG_T vmm, const char* name);
    template <typename PRC_T, typename REG_T>
    static void print_reg(jit_generator& h, REG_T reg, const char* name);
    template <typename PRC_T, size_t vlen>
    static void print_vmm_prc(const char* name, const char* ori_name, PRC_T* ptr);
    template <typename T>
    static void print_reg_prc(const char* name, const char* ori_name, T* val);
    static void preamble(jit_generator& h);
    static void postamble(jit_generator& h);
    template <typename T>
    static void save_vmm(jit_generator& h);
    template <typename T>
    static void restore_vmm(jit_generator& h);
    static void save_reg(jit_generator& h);
    static void restore_reg(jit_generator& h);
    static void align_rsp(jit_generator& h);
    static void restore_rsp(jit_generator& h);
    static constexpr size_t reg_len = 8;
    static constexpr size_t reg_cnt = 16;
};

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
