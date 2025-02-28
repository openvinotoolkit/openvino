// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_util.hpp"

#include "openvino/core/except.hpp"


namespace ov {
namespace intel_cpu {
namespace riscv64 {

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return #jit_name; } \
    const char *source_file() const override { return __FILE__; }

// RISCV-64 specific registers mapping
// reg    | ABI Name | descripion             | saved by
// =====================================================
//                32 integer registers
// x0     | zero     | Always zero            |   ---
// x1     | ra       | Return address         |  Caller
// x2     | sp       | Stack pointer          |  Callee
// x3     | gp       | Global pointer         |   ---
// x4     | tp       | Thread poiner          |   ---
// x5     | t0       | Temp / Alt `ra`        |  Caller
// x6-7   | t1-2     | Temporaries            |  Caller
// x8     | s0 / fp  | Saved reg / frame ptr  |  Callee
// x9     | s1       | Saved register         |  Callee
// x10-11 | a0-1     | Func args / return val |  Caller
// x12-17 | a2-7     | Function arguments     |  Caller
// x18-27 | s2-11    | Saved registers        |  Callee
// x28-31 | t3-6     | Temporaries            |  Caller
// =====================================================
//             32 floating-point registers
// f0-7   | ft0-7    | FP Temporaries         |  Caller
// f8-9   | fs0-1    | FP Saved registers     |  Callee
// f10-11 | fa0-1    | FP args / return val   |  Caller
// f12-17 | fa2-7    | FP Function arguments  |  Caller
// f18-27 | fs2-11   | FP Saved registers     |  Callee
// f28-31 | ft8-11   | FP Temporaries         |  Caller

class jit_generator : public Xbyak_riscv::CodeGenerator {
public:
    jit_generator(size_t maxSize = Xbyak_riscv::DEFAULT_MAX_CODE_SIZE,
                  void *userPtr = Xbyak_riscv::DontSetProtectRWE,
                  Xbyak_riscv::Allocator *allocator = 0)
        : Xbyak_riscv::CodeGenerator(maxSize, userPtr, allocator) {}
    virtual ~jit_generator() {}

    const uint8_t *jit_ker() const {
        OPENVINO_ASSERT(jit_ker_,"jit_ker_ is nullable");
        return jit_ker_;
    }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual bool create_kernel() {
        generate();
        jit_ker_ = getCode();
        return jit_ker_;
    }

    void preamble();
    void postamble();

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_riscv::Label &label) {
        Xbyak_riscv::CodeGenerator::L(label);
    }

    jit_generator(const jit_generator &) = delete;
    jit_generator &operator=(const jit_generator &) = delete;

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    // Callee-saved registers
    static constexpr Xbyak_riscv::Reg abi_save_gpr_regs[] = {Xbyak_riscv::s0, Xbyak_riscv::s1, Xbyak_riscv::s2, Xbyak_riscv::s3,
                                                             Xbyak_riscv::s4, Xbyak_riscv::s5, Xbyak_riscv::s6, Xbyak_riscv::s7,
                                                             Xbyak_riscv::s8, Xbyak_riscv::s9, Xbyak_riscv::s10, Xbyak_riscv::s11};
    static constexpr Xbyak_riscv::FReg abi_save_fp_gpr_regs[] = {Xbyak_riscv::fs0, Xbyak_riscv::fs1, Xbyak_riscv::fs2, Xbyak_riscv::fs3,
                                                                 Xbyak_riscv::fs4, Xbyak_riscv::fs5, Xbyak_riscv::fs6, Xbyak_riscv::fs7,
                                                                 Xbyak_riscv::fs8, Xbyak_riscv::fs9, Xbyak_riscv::fs10, Xbyak_riscv::fs11};
    // ABI-arguments registers
    static constexpr Xbyak_riscv::Reg abi_param_regs[] = {Xbyak_riscv::a0, Xbyak_riscv::a1, Xbyak_riscv::a2, Xbyak_riscv::a3,
                                                          Xbyak_riscv::a4, Xbyak_riscv::a5, Xbyak_riscv::a6, Xbyak_riscv::a7};

    // load size_t value to GPR safely
    void uni_li(const Xbyak_riscv::Reg& rd, size_t value);

    // negative pseudo-instruction
    void vfneg_vv(const Xbyak_riscv::VReg& vd, const Xbyak_riscv::VReg& vs, Xbyak_riscv::VM vm = Xbyak_riscv::VM::unmasked);

    static Xbyak_riscv::LMUL float2lmul(const float lmul);
    static Xbyak_riscv::SEW bytes2sew(const size_t bytes);
    static float lmul2float(const Xbyak_riscv::LMUL lmul);
    static size_t sew2bytes(const Xbyak_riscv::SEW sew);

protected:
    virtual void generate() = 0;

    virtual const uint8_t* getCodeAddress() const {
        return CodeGenerator::getCode();
    }

    const uint8_t *jit_ker_ = nullptr;

    // In the standard RISC-V calling convention, the stack pointer is always kept 16-byte aligned
    const size_t sp_aligment = 16;
    // FP GP register count
    const size_t fp_gpr_count = 32;
    // GP register count
    const size_t gpr_count = 32;
    // Vector register count
    const size_t vec_count = 32;
    // integer gpr byte size
    const size_t xlen = Xbyak_riscv::CPU::getInstance().getXlen() / 8;
    // fp gpr byte size
    const size_t flen = Xbyak_riscv::CPU::getInstance().getFlen() / 8;
    // vector register byte size
    const size_t vlen = Xbyak_riscv::CPU::getInstance().getVlen() / 8;

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
    const size_t num_abi_save_fp_gpr_regs
            = sizeof(abi_save_fp_gpr_regs) / sizeof(abi_save_fp_gpr_regs[0]);
    const size_t num_abi_param_regs
            = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);

private:
    const uint8_t* getCode() {
        ready();
        if (!is_initialized()) return nullptr;
        return getCodeAddress();
    }

    static inline bool is_initialized() {
        // At the moment, Xbyak_riscv does not have GetError()
        // so that return dummy result.
        return true;
    }
};

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
