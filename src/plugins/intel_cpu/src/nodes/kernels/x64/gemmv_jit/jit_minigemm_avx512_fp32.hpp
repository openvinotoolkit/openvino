// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// AVX-512 FP32 mini-GEMM (small N) for int8/u8 -> fp32, M block = 16
// Skeleton supports per_tensor and per_channel; per_group falls back to ref.
class JitMiniGemmAvx512Fp32 : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    struct CallArgs {
        // Up to 4 columns of X and Y
        const float* x0; const float* x1; const float* x2; const float* x3;
        float* y0; float* y1; float* y2; float* y3;
        int n_cols;                // number of active columns (1..4)

        const uint8_t* wq;         // packed weights block base (K * M_step bytes)
        const float* scales;       // per-tensor(1) or per-channel(16) float
        const int32_t* zps;        // nullable; per-tensor(1) or per-channel(16)
        const float* bias;         // nullable; per-tensor(1) or per-channel(16)
        int K;                     // K length
        int M_tail;                // tail lanes (0/full)
        int gran;                  // quant_granularity_t
        int is_u8;                 // 1: u8; 0: i8  (ignored if w_nbits==4)
        int w_nbits;               // 8 or 4
        int w_unsigned;            // 1 if unsigned (u8/u4), 0 if signed (i8/i4)
        int k_step_bytes;          // 16 for 8-bit, 8 for 4-bit
        float sum_x0, sum_x1, sum_x2, sum_x3; // per-column sum(X)
        // Optional MoE gate scales per column (1..4); if unset, wrapper fills 1.f
        float gate0, gate1, gate2, gate3;
        // Optional activation kind: 0=none, 1=ReLU (placeholder for SiLU/GELU)
        int act_kind;
    };

    JitMiniGemmAvx512Fp32();

    using fn_t = void(*)(const CallArgs*);
    fn_t get() const { return fn_; }

protected:
    const char* name() const override { return "jit_minigemm_avx512_fp32"; }
    const char* source_file() const override { return __FILE__; }
    void generate() override;

private:
    fn_t fn_ = nullptr;
};

// Helper to run JIT mini-GEMM for i8/u8 if available; returns false if unsupported
bool run_minigemm_jit_i8u8_fp32(const float* x, int K, int N,
                                const uint8_t* wq_packed, int M, int ld_w_bytes,
                                const float* scales, const int32_t* zps,
                                float* y, const float* bias,
                                quant_granularity_t gran, bool is_u8);

// Generalized helper to run JIT mini-GEMM for i8/u8/i4/u4; returns false if unsupported
bool run_minigemm_jit_q_fp32(const float* x, int K, int N,
                             const uint8_t* wq_packed, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran, w_dtype_t wtype, int group_size);

} // namespace ov::intel_cpu::x64::gemmv_jit
