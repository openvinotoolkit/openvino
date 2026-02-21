// Minimal AVX-512 JIT kernel for sanity: write zeros to y for one M-block
#pragma once

#include "gemmv_ukernel.hpp"
#include "openvino/core/except.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

class JitGemmvAvx512Zero : public GemmvKernel {
public:
    struct CallArgs {
        float* y;
    };

    JitGemmvAvx512Zero();

    void operator()(const gemmv_ukr_params_t* p) const override {
        CallArgs a{static_cast<float*>(p->y)};
        fn_(&a);
    }
    const char* name() const override { return "jit_avx512_zero"; }
private:
    using fn_t = void(*)(const CallArgs*);
    class kernel_t : public dnnl::impl::cpu::x64::jit_generator_t {
    public:
        kernel_t();
        fn_t get() const { return fn_; }
    protected:
        const char* name() const override { return "jit_gemmv_avx512_zero"; }
        const char* source_file() const override { return __FILE__; }
        void generate() override;
    private:
        fn_t fn_ = nullptr;
    };

    fn_t fn_{};
};

inline JitGemmvAvx512Zero::kernel_t::kernel_t()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_gemmv_avx512_zero",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_gemmv_avx512_zero kernel");
    }
    fn_ = reinterpret_cast<fn_t>(jit_ker());
}

inline void JitGemmvAvx512Zero::kernel_t::generate() {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif
    push(r12); push(r13); push(r14); push(r15);
    Reg64 reg_args = rdi;
    Reg64 reg_y = r13;
    mov(reg_y, ptr[reg_args + offsetof(CallArgs, y)]);
    Zmm z = zmm0;
    vxorps(z, z, z);
    vmovups(ptr[reg_y], z);
    pop(r15); pop(r14); pop(r13); pop(r12);
    ret();
}

inline JitGemmvAvx512Zero::JitGemmvAvx512Zero() {
    static kernel_t kernel;
    fn_ = kernel.get();
    OPENVINO_ASSERT(fn_ != nullptr, "jit_gemmv_avx512_zero kernel pointer is null");
}

} // namespace ov::intel_cpu::x64::gemmv_jit
