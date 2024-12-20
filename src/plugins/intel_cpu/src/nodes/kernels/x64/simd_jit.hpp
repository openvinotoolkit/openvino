// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "cpu/x64/jit_generator.hpp"

namespace ov {
namespace intel_cpu {

#ifdef _WIN32
#    define abi_param_regs_num 4
#else
#    define abi_param_regs_num 6
#endif
// first few regs contains input arguments passed in through stack
constexpr Xbyak::Operand::Code abi_x86_64_regs[] = {
#ifdef _WIN32
    // args passed in register
    Xbyak::Operand::RCX,
    Xbyak::Operand::RDX,
    Xbyak::Operand::R8,
    Xbyak::Operand::R9,

    // regs for local variables
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R10,
    Xbyak::Operand::R11,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15
#else
    // args passed in register
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
    Xbyak::Operand::RDX,
    Xbyak::Operand::RCX,
    Xbyak::Operand::R8,
    Xbyak::Operand::R9,

    // regs for local variables
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R10,
    Xbyak::Operand::R11,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15
#endif
};

//*****************************************
// RegCmp & compare operator overload to support if_ statement:
//    - if_(rax > rbx)
//    - if_(rax == rbx)
//    - if_(rax <= 8)
// Note lhs & rhs of binary operator can only be register or imm32, not a general expression
struct RegCmp {
    struct RegImm {
        int32_t id; /* imm32 or regid */
        bool is_imm32;
        RegImm() = default;
        RegImm(const Xbyak::Reg64& reg) : id(reg.getIdx()), is_imm32(false) {}
        RegImm(int imm32) : id(imm32), is_imm32(true) {}
    };
    Xbyak::Reg64 lhs;
    RegImm rhs;
    std::string op;
    RegCmp(std::string _op, const RegImm& _lhs, const RegImm& _rhs) {
        if (_lhs.is_imm32) {
            lhs = Xbyak::Reg64(_rhs.id);
            rhs = _lhs;
            op = _op;
            // revert op
            if (op == ">")
                op = "<";
            else if (op == "<")
                op = ">";
            else if (op == "<=")
                op = ">=";
            else if (op == ">=")
                op = "<=";
        } else {
            lhs = Xbyak::Reg64(_lhs.id);
            rhs = _rhs;
            op = _op;
        }
    }
};

inline RegCmp operator==(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp("==", lhs, rhs);
}
inline RegCmp operator==(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp("==", lhs, rhs);
}
inline RegCmp operator!=(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp("!=", lhs, rhs);
}
inline RegCmp operator!=(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp("!=", lhs, rhs);
}
inline RegCmp operator>=(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp(">=", lhs, rhs);
}
inline RegCmp operator>=(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp(">=", lhs, rhs);
}
inline RegCmp operator<=(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp("<=", lhs, rhs);
}
inline RegCmp operator<=(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp("<=", lhs, rhs);
}
inline RegCmp operator>(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp(">", lhs, rhs);
}
inline RegCmp operator>(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp(">", lhs, rhs);
}
inline RegCmp operator<(const RegCmp::RegImm& lhs, const Xbyak::Reg64& rhs) {
    return RegCmp("<", lhs, rhs);
}
inline RegCmp operator<(const Xbyak::Reg64& lhs, const RegCmp::RegImm& rhs) {
    return RegCmp("<", lhs, rhs);
}

class SIMDJit : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(SIMDJit);
    const bool use_avx512;

    SIMDJit(const char* name)
        : dnnl::impl::cpu::x64::jit_generator(name),
          use_avx512(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        mov(rax, rsp);
        preamble();
    }

    void generate() override{};

    // add an int64_t return value
    template <typename... kernel_args_t>
    int64_t operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = int64_t (*)(const kernel_args_t... args);
        auto* fptr = (jit_kernel_func_t)jit_ker();
        return (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    void finalize(Xbyak::Reg64 return_value = Xbyak::Reg64(Xbyak::Operand::RAX)) {
        if (return_value.getIdx() != rax.getIdx())
            mov(rax, return_value);
        postamble();
        create_kernel();
    }

    Xbyak::Reg64 get_sreg(int i, bool is_arg = false) {
        if (i < abi_param_regs_num)
            return Xbyak::Reg64(abi_x86_64_regs[i]);
        if (i >= static_cast<int>(sizeof(abi_x86_64_regs) / sizeof(abi_x86_64_regs[0])))
            throw std::runtime_error(std::string("try to allocate invalid scalar register #") + std::to_string(i));

        auto r = Xbyak::Reg64(abi_x86_64_regs[i]);
        if (is_arg)
            mov(r, ptr[rax + (i - abi_param_regs_num + 1) * 8]);  // load from stack
        return r;
    }

    Xbyak::Xmm Vmm(int id) {
        if (use_avx512) {
            if (id >= 32)
                throw std::runtime_error(std::string("try to use invalid zmm register: #") + std::to_string(id));
            return Xbyak::Zmm(id);
        } else {
            if (id >= 16)
                throw std::runtime_error(std::string("try to use invalid ymm register: #") + std::to_string(id));
            return Xbyak::Ymm(id);
        }
    }

    // simd_xxx helpers have meaning similar to x86 intrinsics
    // it's more well-known than raw instruction can it also can be
    // made cross-platform(avx2/avx512/neon/...)

    void simd_set1_epi32(Xbyak::Xmm vmm, int32_t imm32) {
        // this set1 is not performance critical
        mov(dword[rsp - 4], imm32);
        vpbroadcastd(vmm, dword[rsp - 4]);
    }
    void simd_and(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        if (use_avx512) {
            vpandd(c, a, b);
        } else {
            vpand(c, a, b);
        }
    }
    void simd_srli_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpsrld(vdst, vsrc, imm8);
    }
    void simd_srai_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpsrad(vdst, vsrc, imm8);
    }
    void simd_slli_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpslld(vdst, vsrc, imm8);
    }
    void simd_setzero_ps(Xbyak::Xmm vmm) {
        if (use_avx512) {
            vpxord(vmm, vmm, vmm);
        } else {
            vpxor(vmm, vmm, vmm);
        }
    }
    void simd_loadu_ps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vmovups(vmm, addr);
    }
    // load packed half into packed single
    void simd_loadu_phps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vcvtph2ps(vmm, addr);
    }
    void simd_load_epu8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vpmovzxbd(vmm, addr);
    }
    void simd_load_epi8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vpmovsxbd(vmm, addr);
    }
    void simd_storeu_ps(const Xbyak::Address& addr, Xbyak::Xmm vmm) {
        vmovups(addr, vmm);
    }
    void simd_fmadd_ps(Xbyak::Xmm c, Xbyak::Xmm a, const Xbyak::Operand& b) {
        vfmadd231ps(c, a, b);
    }
    void simd_sub_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vsubps(c, a, b);
    }
    void simd_add_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vaddps(c, a, b);
    }
    void simd_mul_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vmulps(c, a, b);
    }
    void simd_broadcast_ss(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vbroadcastss(vmm, addr);
    }
    void simd_cvtepi32_ps(Xbyak::Xmm vmm_dst, Xbyak::Xmm vmm_src) {
        vcvtdq2ps(vmm_dst, vmm_src);
    }

    //***********************************************
    // for_loop(idx, start, stop, step, loop_body) performs following:
    //    for(int idx=start; idx + step <= stop; idx+=step) {
    //       loop_body();
    //    }
    template <typename Fn, typename START, typename STEP>
    void for_loop(Xbyak::Reg64 idx, START start, Xbyak::Reg64 stop, STEP step, const Fn& loop_body) {
        Xbyak::Label loop, exit;
        mov(idx, start);

        align(64, false);
        L(loop);
        add(idx, step);
        cmp(idx, stop);
        jg(exit, T_NEAR);
        sub(idx, step);

        loop_body();
        add(idx, step);

        jmp(loop, T_NEAR);
        L(exit);
        // at exit, idx is pointing to tail
        sub(idx, step);
    }

    //***********************************************
    // while_(rax > 0, loop_body) performs following:
    //    while(rax > 0) {
    //       loop_body();
    //    }
    template <typename Fn>
    void while_(RegCmp regcmp, const Fn& loop_body) {
        Xbyak::Label loop, exit;

        align(64, false);
        L(loop);

        if (regcmp.rhs.is_imm32)
            cmp(regcmp.lhs, regcmp.rhs.id);
        else
            cmp(regcmp.lhs, Xbyak::Reg64(regcmp.rhs.id));
        if (regcmp.op == "==")
            jne(exit, T_NEAR);  // if not equal (ZF=0).
        if (regcmp.op == "!=")
            je(exit, T_NEAR);  // if equal (ZF=1).
        if (regcmp.op == ">")
            jle(exit, T_NEAR);  // if less or equal (ZF=1 or SF≠ OF).
        if (regcmp.op == ">=")
            jl(exit, T_NEAR);  // if less (SF≠ OF).
        if (regcmp.op == "<")
            jge(exit, T_NEAR);  // if greater or equal (SF=OF).
        if (regcmp.op == "<=")
            jg(exit, T_NEAR);  // if greater (ZF=0 and SF=OF).

        loop_body();

        jmp(loop, T_NEAR);
        L(exit);
    }

    template <typename Fn>
    void do_while_(RegCmp regcmp, const Fn& loop_body) {
        Xbyak::Label loop;

        align(64, false);
        L(loop);

        loop_body();

        if (regcmp.rhs.is_imm32)
            cmp(regcmp.lhs, regcmp.rhs.id);
        else
            cmp(regcmp.lhs, Xbyak::Reg64(regcmp.rhs.id));
        if (regcmp.op == "==")
            je(loop, T_NEAR);
        if (regcmp.op == "!=")
            jne(loop, T_NEAR);
        if (regcmp.op == ">")
            jg(loop, T_NEAR);
        if (regcmp.op == ">=")
            jge(loop, T_NEAR);
        if (regcmp.op == "<")
            jl(loop, T_NEAR);
        if (regcmp.op == "<=")
            jle(loop, T_NEAR);
    }

    //***********************************************
    // if (reg >= imm32, then_body, else_body)
    void if_(RegCmp regcmp, const std::function<void()>& then_body, const std::function<void()>& else_body = {}) {
        Xbyak::Label if_else, if_exit;

        if (regcmp.rhs.is_imm32)
            cmp(regcmp.lhs, regcmp.rhs.id);
        else
            cmp(regcmp.lhs, Xbyak::Reg64(regcmp.rhs.id));
        if (regcmp.op == "==")
            jne(if_else, T_NEAR);  // if not equal (ZF=0).
        if (regcmp.op == "!=")
            je(if_else, T_NEAR);  // if equal (ZF=1).
        if (regcmp.op == ">")
            jle(if_else, T_NEAR);  // if less or equal (ZF=1 or SF≠ OF).
        if (regcmp.op == ">=")
            jl(if_else, T_NEAR);  // if less (SF≠ OF).
        if (regcmp.op == "<")
            jge(if_else, T_NEAR);  // if greater or equal (SF=OF).
        if (regcmp.op == "<=")
            jg(if_else, T_NEAR);  // if greater (ZF=0 and SF=OF).

        then_body();

        if (else_body)
            jmp(if_exit, T_NEAR);

        L(if_else);

        if (else_body)
            else_body();

        L(if_exit);
    }

    template <typename DT>
    static int vmm_width() {
        return (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 512 : 256) / (sizeof(DT) * 8);
    }
};

}  // namespace intel_cpu
}  // namespace ov
