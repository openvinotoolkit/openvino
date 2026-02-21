// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "gemmv_ukernel.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

struct amx_lane_meta_t {
    const float* scales = nullptr;
    const int32_t* zps = nullptr;
    const float* bias = nullptr;
};

// AMX INT8 GEMV (u8 X, s8 W) -> fp32 Y. Returns true if executed; false if unsupported at runtime.
// Optional sumW_precomp: if provided, length must be M (padded to block of 16 if needed)
bool run_gemmv_amx_i8u8_fp32(const float* x_fp32, int K,
                             const uint8_t* wq, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran, int group_size,
                             const int32_t* sumW_precomp = nullptr,
                             const amx_lane_meta_t* lane_meta = nullptr,
                             const uint8_t* x_q8 = nullptr,
                             float x_scale = 1.f,
                             int32_t x_zp = 128,
                             int32_t sum_x_q = 0);

// AMX BF16 GEMV: W (bf16 prepacked K64x16), X fp32 -> bf16 panel; accum fp32; per-tensor bias only
bool run_gemmv_amx_bf16_fp32(const float* x_fp32, int K,
                             const uint16_t* w_bf16_k64, int M, int ld_w_kbytes,
                             float* y, const float* bias);

// Minimal AMX palette load/release self-test (diagnostics)
bool run_amx_tile_probe();

} // namespace ov::intel_cpu::x64::gemmv_jit
