// AVX-512 simple GEMM-v: Y = sum_k u8(W)*X (no scales/zp/bias), M_blk=16
#pragma once

#include "gemmv_ukernel.hpp"
#include "xbyak/xbyak.h"

namespace ov::intel_cpu::x64::gemmv_jit {

class JitGemmvAvx512Simple : public GemmvKernel, private Xbyak::CodeGenerator {
public:
    struct CallArgs {
        const float* x;           // K
        const uint8_t* wq;        // packed K*16 bytes for this M-block
        // Pre-expanded vectors for this M-block (length 16)
        const float* svec;        // scales vector (or all 1.0f)
        const float* bvec;        // bias vector (or nullptr)
        const float* zcomp;       // s*zp*sum_x precomputed (or nullptr)
        float sum_x;              // for zp compensation
        float* y;                 // output 16 floats
        int K;
        int gran;                 // quant_granularity_t
        int w_is_u8;              // 1=u8,0=i8
        int M_tail;               // 0=full, otherwise valid lanes (1..15)
        int mode;                 // 0: no s/bias/zp; 1: +s; 2: +s+bias; 3: +s+bias+zp
        int skip_mul;             // 1: skip vmulps (debug)
        int x_const_one;          // 1: use X=1.0f broadcast (debug)
        int no_fma;               // 1: replace FMA with MUL+ADD (debug)
        int pack_only;            // 1: C += W (debug: test unpack only)
    };
    JitGemmvAvx512Simple();
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "jit_avx512_simple"; }
private:
    using fn_t = void(*)(const CallArgs*);
    fn_t fn_{};
};

} // namespace ov::intel_cpu::x64::gemmv_jit
