// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// Convenience entry for int8/u8 weights and fp32 input/output
void run_gemmv_i8_fp32(const float* x, int K,
                       const uint8_t* wq_packed, int M, int ld_w_bytes,
                       const float* scales, const int32_t* zps,
                       float* y, const float* bias,
                       quant_granularity_t gran, bool accumulate,
                       bool is_u8);

// Extended API: executes and returns kernel name via out-parameter
void run_gemmv_i8_fp32_ex(const float* x, int K,
                          const uint8_t* wq_packed, int M, int ld_w_bytes,
                          const float* scales, const int32_t* zps,
                          float* y, const float* bias,
                          quant_granularity_t gran, bool accumulate,
                          bool is_u8, const char** kernel_name);

// Generic entry for i8/u8/i4/u4 weights
void run_gemmv_q_fp32_ex(const float* x, int K,
                         const uint8_t* wq_packed, int M, int ld_w_bytes,
                         const float* scales, const int32_t* zps,
                         float* y, const float* bias,
                         quant_granularity_t gran, int group_size, bool accumulate,
                         w_dtype_t wtype, const char** kernel_name);

// Multi-thread wrapper across M; threads<=1 falls back to single-thread path.
void run_gemmv_q_fp32_mt(const float* x, int K,
                         const uint8_t* wq_packed, int M, int ld_w_bytes,
                         const float* scales, const int32_t* zps,
                         float* y, const float* bias,
                         quant_granularity_t gran, int group_size, bool accumulate,
                         w_dtype_t wtype, int threads);

// Mini-GEMM (small N) dispatch: tries AVX-512 JIT for i8/u8 (per_tensor/per_channel), otherwise ref
void run_minigemm_q_fp32_ex(const float* x, int K, int N,
                            const uint8_t* wq_packed, int M, int ld_w_bytes,
                            const float* scales, const int32_t* zps,
                            float* y, const float* bias,
                            quant_granularity_t gran, int group_size,
                            w_dtype_t wtype, const char** kernel_name);

// Experimental: AVX-512 VNNI GEMV for i8 (W) x u8 (X) with per-tensor quant.
// Expects W packed via pack_w_i8_k4_m16 (K grouped by 4, 64B per group per M-block).
// Returns true if VNNI path executed; false if unsupported.
bool run_gemmv_vnni_q8s8_ex(const float* x_fp32, int K,
                            const uint8_t* wq_k4, int M, int ld_w_gbytes,
                            const float* scales, const int32_t* zps,
                            float* y, const float* bias,
                            quant_granularity_t gran,
                            int dbg_block = -1, int32_t* dbg_acc = nullptr, int32_t* dbg_sumw = nullptr,
                            const int32_t* sumW_precomp = nullptr);

struct gemmv_profile_snapshot {
    double quant_ns = 0.0;
    double kernel_ns = 0.0;
    double total_ns = 0.0;
};

gemmv_profile_snapshot get_last_gemmv_profile();
void set_gemmv_profile_override(bool enable);

} // namespace ov::intel_cpu::x64::gemmv_jit
