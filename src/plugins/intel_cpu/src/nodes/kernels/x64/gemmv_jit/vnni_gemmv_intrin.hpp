// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ov::intel_cpu::x64::gemmv_jit {

// AVX-512 VNNI intrinsic GEMV implementation (u8 X, s8 W) -> fp32 Y
// Expects W packed via pack_w_i8_k4_m16 (64 bytes per K-group per M-block of 16 rows).
// Returns true if compiled with required ISA and executed; false otherwise.
// Optional sumW_precomp: if provided, length must be M (padded to block of 16)
bool run_gemmv_vnni_intrin_i8u8_fp32(const uint8_t* xq, int K,
                                     const uint8_t* wq_k4, int M, int ld_w_gbytes,
                                     float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                                     float* y, float bias, const int32_t* sumW_precomp = nullptr);

// K64-repack variant: weights in K-major groups of 64 rows, each row 16 bytes (one per M-lane).
// Matches layout produced by repack_interleave_m16_to_k64_m16. Supports per-tensor scale/zp and optional precomputed sumW.
bool run_gemmv_vnni_intrin_i8u8_fp32_k64(const uint8_t* xq, int K,
                                         const uint8_t* wq_k64, int M, int ld_w_kbytes,
                                         float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                                         float* y, float bias, const int32_t* sumW_precomp = nullptr);

// No-repack variant: weights in interleave_m16 layout (16 bytes per k-step per M-block)
// Supports per-tensor/per-channel/per-group scales/zps and optional bias vector; computes
// y[M] = (X_q*u8)·(W_q*s8) with vpdpbusd + integer compensation, then scales by s_w*s_x and adds bias.
// Returns true if ISA supported and executed; false otherwise.
bool run_gemmv_vnni_intrin_i8u8_fp32_norepack(const uint8_t* xq, int K,
                                              const uint8_t* wq_m16k, int M, int ld_w_bytes,
                                              const float* scales, const int32_t* zps, const float* bias,
                                              float s_x, int32_t zp_x, int32_t sum_x_q,
                                              float* y, int M_tail,
                                              int gran, int group_size);

} // namespace ov::intel_cpu::x64::gemmv_jit
