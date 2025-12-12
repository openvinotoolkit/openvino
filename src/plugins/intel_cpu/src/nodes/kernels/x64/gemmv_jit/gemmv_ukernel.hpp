// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

namespace ov::intel_cpu::x64::gemmv_jit {

// Note: This API models a GEMM-v microkernel (N=1), computing Y[M] = W[MxK] * X[K] (+bias)

enum class a_dtype_t { fp32, bf16 };
enum class w_dtype_t { i8, u8, i4, u4 };
enum class quant_granularity_t { per_tensor, per_channel, per_group };
enum class w_layout_t {
    interleave_m16 = 0,
    k4_m16,
    k64_m16,
    k64_tile_i8,
    k64_tile_bf16
};

struct gemmv_ukr_params_t {
    // Input vector X[K]
    const void* x = nullptr;
    int K = 0;

    // Quantized, prepacked weights for a single M-blocked panel
    // Layout for int8/i8: for each k in [0..K), 16 bytes of {w[m..m+15,k]} (M_blk=16)
    const uint8_t* wq = nullptr;
    int ld_w_bytes = 0; // stride between M-blocks in bytes (usually K * M_blk)
    const int32_t* sumW_precomp = nullptr; // optional precomputed per-row sumW (size >= padded M)

    // Quantization metadata
    const float* scales = nullptr;     // length depends on granularity
    const int32_t* zps = nullptr;      // nullable
    quant_granularity_t gran = quant_granularity_t::per_tensor;
    int group_size = 0;               // for per_group granularity: number of rows per group (along M)

    // Optional per-lane metadata (length >= padded M of the panel)
    const float* lane_scales = nullptr;
    const float* lane_bias = nullptr;
    const int32_t* lane_zps = nullptr;

    // Output Y[M]
    void* y = nullptr;                 // fp32
    float* bias = nullptr;             // optional, fp32 per-channel or per-tensor
    int M = 0;                         // total rows to process
    bool accumulate = false;           // if true: y += gemmv; otherwise overwrite
    int m_base = 0;                    // global M offset used for per_group metadata addressing in subrange calls

    // Optional small-batch parameter for mini-GEMM reference/JIT paths
    // When >1, callers may compute Y[M x N_expert] by either running GEMV N_expert times
    // or using a dedicated mini-GEMM kernel that reuses W decode across columns.
    int N_expert = 1;                  // default 1 (classic GEMV)

    // Tail (M not divisible by block)
    int M_tail = 0;                    // valid lanes in the last M-block (0 means full)

    // Types
    a_dtype_t a_type = a_dtype_t::fp32;
    w_dtype_t w_type = w_dtype_t::i8;
    w_layout_t w_layout = w_layout_t::interleave_m16;

    // Optional debug (INT4 capture)
    int dbg_enable = 0;            // 1 to enable capture in JIT
    int dbg_k = 0;                 // capture at this k index
    float* dbg_q = nullptr;        // [M_blk] pre-scale decoded q (fp32)
    float* dbg_qs = nullptr;       // [M_blk] q*s (fp32)

    // MoE epilogue (optional)
    // For GEMV (N=1): single gate scale. For mini-GEMM JIT, per-column gates passed via CallArgs.
    bool fuse_gate = false;
    float gate_scale = 1.f;
    int act_kind = 0;              // 0: none, 1: ReLU (placeholder for SiLU/GELU)

    // Optional quantized-X (u8) fast path (VNNI/AMX): when provided and supported,
    // implementation may use int8-dot kernels with s32 accumulation.
    const uint8_t* x_q8 = nullptr; // if null, implementation may quantize x on the fly or fallback
    float x_scale = 1.f;           // per-tensor scale for X (u8 path)
    int32_t x_zp = 128;            // per-tensor zero-point for X (u8 path)
    int32_t sum_x_q = 0;           // sum of quantized X entries (for zp compensation)
};

// Reference mini-GEMM (small-N) using packed quantized W and fp32 X
// Computes Y[M x N] = dequant(Wq[MxK]) * X[KxN] + (bias - s*zp*sum_x_col)
// - w_type selects int8/uint8/int4/uint4 decode path
// - gran selects per_tensor or per_channel scaling/bias/zp mapping
// - ld_w_bytes is the stride between M-blocks in bytes (K*16 for 8-bit, K*8 for 4-bit)
// Layout conventions:
//   * Wq is already packed by pack_w_i8_interleave_m16 / pack_w_i4_interleave_m16
//   * X is column-major by K (i.e., X_col(n)[k] at x[n*K + k])
//   * Y is column-major by M (i.e., Y_col(n)[m] at y[n*M + m])
void run_minigemm_ref_q_fp32(const float* x, int K, int N,
                             const uint8_t* wq_packed, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran,
                             w_dtype_t wtype,
                             bool accumulate,
                             int group_size = 0);

// Opaque compiled JIT kernel for AVX-512/AVX2/AMX flavors.
class GemmvKernel {
public:
    virtual ~GemmvKernel() = default;
    virtual void operator()(const gemmv_ukr_params_t* p) const = 0;
    virtual const char* name() const = 0;
};

// Factory returns best available kernel for current CPU/features and params.
// May fall back to a portable reference implementation.
GemmvKernel* create_gemmv_kernel(const gemmv_ukr_params_t& proto);

// Simple int8 prepack helper: packs row-major W[MxK] -> (M_blk x K) with M-major interleave
// so that for fixed k, bytes {W[m..m+M_blk-1, k]} are contiguous.
// Returns buffer size in bytes written to dst. M is padded up to M_blk (16).
size_t pack_w_i8_interleave_m16(uint8_t* dst,
                                const uint8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                                int M_blk = 16);

// Pack int4/u4 (values in low 4 bits of each src byte for convenience) from row-major [M,K]
// into M-major interleaved blocks of M_blk=16. For each k, 16 nibbles are packed into 8 bytes:
// byte i holds { W[m0+2*i, k] as low nibble, W[m0+2*i+1, k] as high nibble }.
// Returns bytes written; M is padded up to M_blk. ld_src_bytes is the stride of the row-major source in bytes.
size_t pack_w_i4_interleave_m16(uint8_t* dst,
                                const int8_t* src_q4, int M, int K, ptrdiff_t ld_src_bytes,
                                int M_blk = 16);

// Pack i8 weights for VNNI GEMV (N=1): group K by 4, contiguous 4 bytes per row.
// Layout per M-block (M_blk=16): for g in [0..K/4), for m in [0..M_blk): emit bytes {w[m,k+0],w[m,k+1],w[m,k+2],w[m,k+3]}.
// Returns bytes written; M is padded to M_blk. ld_src_bytes is stride in bytes between elements of K within a row.
size_t pack_w_i8_k4_m16(uint8_t* dst,
                        const int8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                        int M_blk = 16);

// Pack i8 weights for AMX GEMV (N=1) with K blocked by 64: per M-block (M_blk=16),
// for each K-group of 64, and for each row m in [0..M_blk): emit 64 bytes
// {w[m, k0+0], ..., w[m, k0+63]} (zeros for k>=K or m>=M). Returns bytes written.
size_t pack_w_i8_k64_m16(uint8_t* dst,
                         const int8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                         int M_blk = 16, int K_blk = 64);

// Repack from interleaved-by-M M_blk=16 layout (current GEMV pack) to K64 layout expected by AMX.
// Source layout: for each block base m0, for k in [0..K): 16 bytes of {W[m0..m0+15, k]}.
// Produces blocks of 64-bytes per row per K-group of 64 in dst.
size_t repack_interleave_m16_to_k64_m16(uint8_t* dst,
                                        const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                        int M_blk = 16, int K_blk = 64,
                                        int32_t* sumW_out = nullptr);

// AMX tile-friendly repack (K64 x 16), writing rows of 16 contiguous bytes per k
// Layout: for each K64 block and for each k in [0..63], emit 16 bytes = W[m0..m0+15, k]
// This matches _tile_loadd(..., stride=16) with cfg.rows[2]=64, cfg.colsb[2]=16.
size_t repack_interleave_m16_to_k64_m16_tile(uint8_t* dst,
                                             const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                             int M_blk = 16, int K_blk = 64,
                                             int32_t* sumW_out = nullptr);

// Repack from interleave_m16 to K4 layout expected by VNNI GEMV (4 bytes per row per K-group).
// dst size: M_pad * K_grp * 4 bytes. ld_w_bytes_interleave is stride (K*M_blk) of interleave layout.
size_t repack_interleave_m16_to_k4_m16(uint8_t* dst,
                                       const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                       int M_blk = 16,
                                       int32_t* sumW_out = nullptr);

// Convert interleave_m16 int8 weights to BF16 and repack into AMX tile-friendly K64x16 layout.
// dst_bf16 holds rows of 16 bf16 lanes for each k in K64 group: size = M_pad * K_grp * K_blk * 16 * 2 bytes.
// Only per-tensor (scale, zp) supported by this helper.
size_t repack_interleave_m16_to_k64_m16_bf16(uint16_t* dst_bf16,
                                             const uint8_t* src_interleave_i8, int M, int K, int ld_w_bytes_interleave,
                                             float scale, int32_t zp,
                                             int M_blk = 16, int K_blk = 64);

} // namespace ov::intel_cpu::x64::gemmv_jit
