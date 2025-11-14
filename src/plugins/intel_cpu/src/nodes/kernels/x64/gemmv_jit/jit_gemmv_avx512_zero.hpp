// Minimal AVX-512 JIT kernel for sanity: write zeros to y for one M-block
#pragma once

#include "gemmv_ukernel.hpp"
#include "xbyak/xbyak.h"

namespace ov::intel_cpu::x64::gemmv_jit {

class JitGemmvAvx512Zero : public GemmvKernel, private Xbyak::CodeGenerator {
public:
    struct CallArgs {
        float* y;
    };
    JitGemmvAvx512Zero() : Xbyak::CodeGenerator(4096) {
        using namespace Xbyak;
        setDefaultJmpNEAR(true);
        // Save callee-saved we might touch
        push(r12); push(r13); push(r14); push(r15);
        Reg64 reg_args = rdi;
        Reg64 reg_y = r13;
        mov(reg_y, ptr[reg_args + offsetof(CallArgs, y)]);
        Zmm z = zmm0;
        vxorps(z, z, z);
        vmovups(ptr[reg_y], z);
        pop(r15); pop(r14); pop(r13); pop(r12);
        ret();
        fn_ = getCode<fn_t>();
    }
    void operator()(const gemmv_ukr_params_t* p) const override {
        CallArgs a{static_cast<float*>(p->y)};
        fn_(&a);
    }
    const char* name() const override { return "jit_avx512_zero"; }
private:
    using fn_t = void(*)(const CallArgs*);
    fn_t fn_{};
};

} // namespace ov::intel_cpu::x64::gemmv_jit
