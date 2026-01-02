// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "xbyak_riscv/xbyak_riscv.hpp"

#ifdef CPU_DEBUG_CAPS

#    include "nodes/kernels/riscv64/jit_generator.hpp"

namespace ov::intel_cpu::riscv64 {

// Usage
// 1. Include this header in RISC-V JIT kernels if register printing is required.
// 2. Invoke the RegPrinter::print method with a supported register/precision combination.
//    Examples:
//      RegPrinter::print<float>(*h, vreg_val, "vreg_val");
//      RegPrinter::print<int>(*h, gpr_val, "gpr_val");
//
// Supported combinations:
//           fp32        int32
// VReg       Yes         Yes
// Reg        Yes         Yes
class RegPrinter {
public:
    using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;

    template <typename PRC_T, typename REG_T>
    static void print(jit_generator_t& h, REG_T reg, const char* name = nullptr) {
        if constexpr (std::is_same_v<REG_T, Xbyak_riscv::VReg>) {
            print_vmm<PRC_T>(h, reg, name);
        } else {
            print_reg<PRC_T>(h, reg, name);
        }
    }

private:
    RegPrinter() = default;

    template <typename PRC_T>
    static void print_vmm(jit_generator_t& h, const Xbyak_riscv::VReg& vmm, const char* name);

    template <typename PRC_T>
    static void print_reg(jit_generator_t& h, const Xbyak_riscv::Reg& reg, const char* name);

    static void preamble(jit_generator_t& h);
    static void postamble(jit_generator_t& h);
};

}  // namespace ov::intel_cpu::riscv64

#endif  // CPU_DEBUG_CAPS
