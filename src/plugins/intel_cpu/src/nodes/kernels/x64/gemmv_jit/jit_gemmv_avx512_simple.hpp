// AVX-512 simple GEMM-v: Y = sum_k u8(W)*X (no scales/zp/bias), M_blk=16
#pragma once

#include "gemmv_ukernel.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

struct gemmv_avx512_simple_call_args {
    const float* x;
    const uint8_t* wq;
    const float* svec;
    const float* bvec;
    const float* zcomp;
    float sum_x;
    float* y;
    int K;
    int gran;
    int w_is_u8;
    int M_tail;
    int mode;
    int skip_mul;
    int x_const_one;
    int no_fma;
    int pack_only;
};

class jit_gemmv_avx512_simple_kernel : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    using fn_t = void(*)(const gemmv_avx512_simple_call_args*);

    jit_gemmv_avx512_simple_kernel();

    fn_t get() const { return fn_; }

protected:
    const char* name() const override { return "jit_gemmv_avx512_simple_kernel"; }
    const char* source_file() const override { return __FILE__; }
    void generate() override;

private:
    fn_t fn_ = nullptr;
};

class JitGemmvAvx512Simple : public GemmvKernel {
public:
    JitGemmvAvx512Simple();
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "jit_avx512_simple"; }
private:
    using fn_t = jit_gemmv_avx512_simple_kernel::fn_t;
    fn_t fn_{};
};

} // namespace ov::intel_cpu::x64::gemmv_jit
