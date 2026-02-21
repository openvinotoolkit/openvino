// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

#include <memory>
#include <vector>

#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

struct gemmv_avx512_fp32_call_args {
    const float* x;
    const uint8_t* wq;
    const float* scales;
    const int32_t* zps;
    const float* bias;
    float* y;
    int K;
    int M_tail;
    int ld_w_bytes;
    int gran;
    int w_nbits;
    int w_unsigned;
    int k_step_bytes;
    int accumulate;
    float sum_x;
    int fuse_gate;
    float gate;
    int act_kind;
    int dbg_enable;
    int dbg_k;
    float* dbg_q;
    float* dbg_qs;
    const uint8_t* x_q8 = nullptr;
    float s_x = 1.f;
    int32_t zp_x = 128;
    int32_t sum_x_q = 0;
    int use_vnni = 0;
    int need_sumw = 0;
};

class jit_gemmv_avx512_fp32_kernel : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    using fn_t = void(*)(const gemmv_avx512_fp32_call_args*);

    jit_gemmv_avx512_fp32_kernel();

    fn_t get() const { return fn_; }

protected:
    const char* name() const override { return "jit_gemmv_avx512_fp32_kernel"; }
    const char* source_file() const override { return __FILE__; }
    void generate() override;

private:
    fn_t fn_ = nullptr;
};

// AVX-512 FP32 GEMM-v (int8/u8 -> fp32), M block = 16 lanes (zmm)
class JitGemmvAvx512Fp32 : public GemmvKernel {
public:
    JitGemmvAvx512Fp32();
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "jit_avx512_fp32"; }

private:
    using fn_t = jit_gemmv_avx512_fp32_kernel::fn_t;
    fn_t fn_ = nullptr;
};

} // namespace ov::intel_cpu::x64::gemmv_jit
