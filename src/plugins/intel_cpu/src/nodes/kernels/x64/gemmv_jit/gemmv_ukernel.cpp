// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gemmv_ukernel.hpp"
#include "jit_gemmv_avx512_fp32.hpp"
#include "jit_gemmv_avx2_fp32.hpp"
#include "jit_gemmv_avx512_zero.hpp"
#include "jit_gemmv_avx512_simple.hpp"
#include "jit_gemmv_avx512_vnni_s32.hpp"
#include "jit_gemmv_amx_kernels.hpp"
#include "gemmv_force_isa.hpp"

#include <memory>
#include <new>
#include <cstring>
#include <vector>

// Use oneDNN ISA traits for runtime feature detection
#include "cpu/x64/cpu_isa_traits.hpp"
#include "xbyak/xbyak_util.h"

namespace ov::intel_cpu::x64::gemmv_jit {

using dnnl::impl::cpu::x64::cpu_isa_t;
using dnnl::impl::cpu::x64::get_max_cpu_isa;
using dnnl::impl::cpu::x64::is_subset;

// Simple shared-kernel proxy to allow caching compiled code across calls
namespace {
class SharedKernelProxy : public GemmvKernel {
public:
    explicit SharedKernelProxy(const GemmvKernel* target) : tgt_(target) {}
    void operator()(const gemmv_ukr_params_t* p) const override { (*tgt_)(p); }
    const char* name() const override { return tgt_->name(); }
private:
    const GemmvKernel* tgt_;
};
} // anonymous namespace

// Factory
GemmvKernel* create_gemmv_kernel(const gemmv_ukr_params_t& proto) {
    Xbyak::util::Cpu cpu;
    const bool has_avx512f  = cpu.has(Xbyak::util::Cpu::tAVX512F);
    const bool has_avx2     = cpu.has(Xbyak::util::Cpu::tAVX2);
    const bool has_vnni     = cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) && cpu.has(Xbyak::util::Cpu::tAVX512BW);
    const bool has_amx_int8 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::amx_int8);
    const bool has_amx_bf16 = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::amx_bf16);

    const gemmv_force_isa_t force_mode = get_gemmv_force_isa();
    auto make_avx512 = [](){
        static JitGemmvAvx512Fp32* shared = nullptr;
        if (!shared) shared = new JitGemmvAvx512Fp32();
        return (GemmvKernel*)new SharedKernelProxy(shared);
    };
    auto make_avx2 = [](){
        // No dedicated AVX2 JIT yet: fall back to portable REF (vectorized C++)
        return (GemmvKernel*)new RefGemmvFp32();
    };
    auto make_amx_int8 = [](){
        static JitGemmvAmxInt8Kernel* shared = nullptr;
        if (!shared) shared = new JitGemmvAmxInt8Kernel();
        return (GemmvKernel*)new SharedKernelProxy(shared);
    };
    auto make_amx_bf16 = [](){
        static JitGemmvAmxBf16Kernel* shared = nullptr;
        if (!shared) shared = new JitGemmvAmxBf16Kernel();
        return (GemmvKernel*)new SharedKernelProxy(shared);
    };
    auto make_vnni = []() {
        static JitGemmvAvx512VnniKernel* shared = nullptr;
        if (!shared) shared = new JitGemmvAvx512VnniKernel();
        return (GemmvKernel*)new SharedKernelProxy(shared);
    };

    if (force_mode == gemmv_force_isa_t::avx512_fp32) {
        if (has_avx512f) return make_avx512();
        if (has_avx2)    return make_avx2();
        return new RefGemmvFp32();
    }
    if (force_mode == gemmv_force_isa_t::avx2_fp32) return make_avx2();
    if (force_mode == gemmv_force_isa_t::ref_fp32) {
        return new RefGemmvFp32();
    }
    if (force_mode == gemmv_force_isa_t::amx_int8) {
        // When forced, instantiate AMX kernel regardless of mayiuse() to let the caller
        // handle XCR0/XFD enablement (ensure_amx_thread_state).
        if (proto.w_layout == w_layout_t::k64_tile_i8 &&
            proto.w_type == w_dtype_t::i8 && proto.a_type == a_dtype_t::fp32) {
            return make_amx_int8();
        }
        return new RefGemmvFp32();
    }
    if (force_mode == gemmv_force_isa_t::amx_bf16) {
        if (has_amx_bf16 && proto.w_layout == w_layout_t::k64_tile_bf16 &&
            proto.a_type == a_dtype_t::fp32) {
            return make_amx_bf16();
        }
        return new RefGemmvFp32();
    }
    if (force_mode == gemmv_force_isa_t::vnni) {
        if (has_vnni &&
            proto.w_layout == w_layout_t::k4_m16 &&
            proto.w_type == w_dtype_t::i8 &&
            proto.a_type == a_dtype_t::fp32 &&
            proto.x_q8 != nullptr) {
            return make_vnni();
        }
        if (has_avx512f) return make_avx512();
        if (has_avx2) return make_avx2();
        return new RefGemmvFp32();
    }

    // AUTO (default): prefer AMX (when layout ready), then AVX-512, then AVX2, else REF
    if (has_amx_int8 &&
        proto.w_layout == w_layout_t::k64_tile_i8 &&
        proto.w_type == w_dtype_t::i8 &&
        proto.a_type == a_dtype_t::fp32) {
        return make_amx_int8();
    }
    if (has_amx_bf16 &&
        proto.w_layout == w_layout_t::k64_tile_bf16 &&
        proto.a_type == a_dtype_t::fp32) {
        return make_amx_bf16();
    }
    if (has_vnni &&
        proto.w_layout == w_layout_t::k4_m16 &&
        proto.w_type == w_dtype_t::i8 &&
        proto.a_type == a_dtype_t::fp32 &&
        proto.x_q8 != nullptr) {
        return make_vnni();
    }
    if (has_avx512f) return make_avx512();
    if (has_avx2)    return make_avx2();
    return new RefGemmvFp32();
}

// Simple pack: row-major src (MxK) -> dst interleaved by M_blk=16 per k
size_t pack_w_i8_interleave_m16(uint8_t* dst,
                                const uint8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                                int M_blk) {
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K);
    // layout: for each block base m0, for k in [0..K): write M_blk bytes: W[m0..m0+M_blk-1, k]
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        for (int k = 0; k < K; ++k) {
            for (int im = 0; im < M_blk; ++im) {
                const int m = m0 + im;
                uint8_t val = 0;
                if (m < M) {
                    const uint8_t* row = src_wq + m * ld_src_bytes;
                    val = row[k];
                } else {
                    val = 0; // pad
                }
                *out++ = val;
            }
        }
    }
    return bytes_written;
}

size_t pack_w_i4_interleave_m16(uint8_t* dst,
                                const int8_t* src_q4, int M, int K, ptrdiff_t ld_src_bytes,
                                int M_blk) {
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K) / 2;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < M_blk / 2; ++i) {
                const int m_even = m0 + 2 * i;
                const int m_odd  = m0 + 2 * i + 1;
                uint8_t lo_n = 0, hi_n = 0;
                if (m_even < M) {
                    // Treat src_q4 values as already quantized 4-bit (two's complement for i4 or 0..15 for u4)
                    int8_t v = *(const int8_t*)((const uint8_t*)src_q4 + m_even * ld_src_bytes + k);
                    lo_n = static_cast<uint8_t>(v) & 0x0F;
                }
                if (m_odd < M) {
                    int8_t v = *(const int8_t*)((const uint8_t*)src_q4 + m_odd * ld_src_bytes + k);
                    hi_n = (static_cast<uint8_t>(v) & 0x0F) << 4;
                }
                *out++ = static_cast<uint8_t>(lo_n | hi_n);
            }
        }
    }
    return bytes_written;
}

size_t pack_w_i8_k4_m16(uint8_t* dst,
                        const int8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                        int M_blk) {
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + 3) / 4;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K_grp) * 4;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * 4;
            for (int im = 0; im < M_blk; ++im) {
                const int m = m0 + im;
                uint8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                if (m < M) {
                    const uint8_t* row = (const uint8_t*)src_wq + m * ld_src_bytes;
                    if (k0 + 0 < K) b0 = (uint8_t)row[k0 + 0];
                    if (k0 + 1 < K) b1 = (uint8_t)row[k0 + 1];
                    if (k0 + 2 < K) b2 = (uint8_t)row[k0 + 2];
                    if (k0 + 3 < K) b3 = (uint8_t)row[k0 + 3];
                }
                *out++ = b0; *out++ = b1; *out++ = b2; *out++ = b3;
            }
        }
    }
    return bytes_written;
}

size_t pack_w_i8_k64_m16(uint8_t* dst,
                         const int8_t* src_wq, int M, int K, ptrdiff_t ld_src_bytes,
                         int M_blk, int K_blk) {
    // K-major 64x16 blocks: for each M-block and each K-group of 64,
    // emit 64 rows, each row holds 16 bytes (one per M-lane)
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + K_blk - 1) / K_blk;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K_grp) * (size_t)K_blk * (size_t)M_blk;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * K_blk;
            for (int kb = 0; kb < K_blk; ++kb) {
                const int k = k0 + kb;
                for (int im = 0; im < M_blk; ++im) {
                    uint8_t b = 0;
                    const int m = m0 + im;
                    if (m < M && k < K) {
                        const uint8_t* row = (const uint8_t*)src_wq + m * ld_src_bytes;
                        b = row[k];
                    }
                    *out++ = b;
                }
            }
        }
    }
    return bytes_written;
}

size_t repack_interleave_m16_to_k64_m16(uint8_t* dst,
                                        const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                        int M_blk, int K_blk,
                                        int32_t* sumW_out) {
    // Output layout: for each K64 group g and for each sub-group of 4 (kg),
    // emit a 64-byte Bgrp tile such that bytes for each lane (m) are contiguous 4 bytes [k..k+3],
    // matching dpbusd expected layout. That eliminates runtime shuffles.
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + K_blk - 1) / K_blk;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K_grp) * (size_t)K_blk * (size_t)M_blk;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        const uint8_t* wblk = src_interleave + (size_t)(m0 / M_blk) * (size_t)ld_w_bytes_interleave;
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * K_blk;
            // within K64 block, iterate 4-wide groups -> 16 Bgrp tiles per group
            for (int kg = 0; kg < K_blk; kg += 4) {
                const int k_base = k0 + kg;
                for (int im = 0; im < M_blk; ++im) {
                    const int m = m0 + im;
                    uint8_t b0=0,b1=0,b2=0,b3=0;
                    if (m < M) {
                        const uint8_t* wk0 = (k_base + 0 < K) ? (wblk + (size_t)(k_base + 0) * M_blk) : nullptr;
                        const uint8_t* wk1 = (k_base + 1 < K) ? (wblk + (size_t)(k_base + 1) * M_blk) : nullptr;
                        const uint8_t* wk2 = (k_base + 2 < K) ? (wblk + (size_t)(k_base + 2) * M_blk) : nullptr;
                        const uint8_t* wk3 = (k_base + 3 < K) ? (wblk + (size_t)(k_base + 3) * M_blk) : nullptr;
                        if (wk0) b0 = wk0[im];
                        if (wk1) b1 = wk1[im];
                        if (wk2) b2 = wk2[im];
                        if (wk3) b3 = wk3[im];
                    }
                    // Write 4 bytes for lane im within this 4-k group
                    *out++ = b0; *out++ = b1; *out++ = b2; *out++ = b3;
                    if (sumW_out) sumW_out[m0 + im] += (int32_t)(int8_t)b0 + (int32_t)(int8_t)b1 + (int32_t)(int8_t)b2 + (int32_t)(int8_t)b3;
                }
            }
        }
    }
    return bytes_written;
}

size_t repack_interleave_m16_to_k64_m16_tile(uint8_t* dst,
                                             const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                             int M_blk,
                                             int K_blk,
                                             int32_t* sumW_out) {
    // Output layout: row-major by M (rows) then K within each K64 block.
    // Each row stores contiguous K_blk bytes so tileloadd with colsb=64 consumes one row.
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + K_blk - 1) / K_blk;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K_grp) * K_blk * M_blk;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        const uint8_t* wblk = src_interleave + (size_t)(m0 / M_blk) * (size_t)ld_w_bytes_interleave;
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * K_blk;
            for (int kb = 0; kb < K_blk; ++kb) {
                const int k = k0 + kb;
                for (int im = 0; im < M_blk; ++im) {
                    uint8_t val = 0;
                    const int m = m0 + im;
                    if (m < M && k < K) {
                        const uint8_t* wk = wblk + (size_t)k * M_blk;
                        val = wk[im];
                    }
                    *out++ = val;
                    if (sumW_out && k < K) {
                        sumW_out[m0 + im] += static_cast<int32_t>(static_cast<int8_t>(val));
                    }
                }
            }
        }
    }
    return bytes_written;
}

size_t repack_interleave_m16_to_k4_m16(uint8_t* dst,
                                       const uint8_t* src_interleave, int M, int K, int ld_w_bytes_interleave,
                                       int M_blk,
                                       int32_t* sumW_out) {
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + 3) / 4;
    size_t bytes_written = static_cast<size_t>(M_pad) * static_cast<size_t>(K_grp) * 4;
    uint8_t* out = dst;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        const uint8_t* wblk = src_interleave + (size_t)(m0 / M_blk) * (size_t)ld_w_bytes_interleave;
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * 4;
            for (int im = 0; im < M_blk; ++im) {
                uint8_t b0=0,b1=0,b2=0,b3=0;
                if (m0 + im < M) {
                    const uint8_t* wk0 = (k0 + 0 < K) ? (wblk + (size_t)(k0 + 0) * M_blk) : nullptr;
                    const uint8_t* wk1 = (k0 + 1 < K) ? (wblk + (size_t)(k0 + 1) * M_blk) : nullptr;
                    const uint8_t* wk2 = (k0 + 2 < K) ? (wblk + (size_t)(k0 + 2) * M_blk) : nullptr;
                    const uint8_t* wk3 = (k0 + 3 < K) ? (wblk + (size_t)(k0 + 3) * M_blk) : nullptr;
                    if (wk0) b0 = wk0[im];
                    if (wk1) b1 = wk1[im];
                    if (wk2) b2 = wk2[im];
                    if (wk3) b3 = wk3[im];
                }
                *out++ = b0; *out++ = b1; *out++ = b2; *out++ = b3;
                if (sumW_out) {
                    sumW_out[m0 + im] += (int32_t)(int8_t)b0 + (int32_t)(int8_t)b1 + (int32_t)(int8_t)b2 + (int32_t)(int8_t)b3;
                }
            }
        }
    }
    return bytes_written;
}

static inline uint16_t f32_to_bf16(float v) {
    union { uint32_t u32; float f; } u; u.f = v;
    // round-to-nearest-even: add 0x7FFF + LSB of truncated part
    uint32_t x = u.u32;
    uint32_t lsb = (x >> 16) & 1U;
    x += 0x7FFF + lsb;
    return (uint16_t)(x >> 16);
}

size_t repack_interleave_m16_to_k64_m16_bf16(uint16_t* dst_bf16,
                                             const uint8_t* src_interleave_i8, int M, int K, int ld_w_bytes_interleave,
                                             float scale, int32_t zp,
                                             int M_blk, int K_blk) {
    const int M_pad = ((M + M_blk - 1) / M_blk) * M_blk;
    const int K_grp = (K + K_blk - 1) / K_blk;
    size_t elems = (size_t)M_pad * (size_t)K_grp * (size_t)K_blk * (size_t)M_blk;
    uint16_t* out = dst_bf16;
    for (int m0 = 0; m0 < M_pad; m0 += M_blk) {
        const uint8_t* wblk = src_interleave_i8 + (size_t)(m0 / M_blk) * (size_t)ld_w_bytes_interleave;
        for (int g = 0; g < K_grp; ++g) {
            const int k0 = g * K_blk;
            for (int kb = 0; kb < K_blk; ++kb) {
                const int k = k0 + kb;
                for (int im = 0; im < M_blk; ++im) {
                    float wr = 0.f;
                    if (im + m0 < M && k < K) {
                        const uint8_t* wk = wblk + (size_t)k * M_blk;
                        int8_t q = (int8_t)wk[im];
                        wr = scale * ((float)q - (float)zp);
                    }
                    *out++ = f32_to_bf16(wr);
                }
            }
        }
    }
    return elems * sizeof(uint16_t);
}

// Helpers mirroring REF path accessors
static inline float ref_get_scale(const float* s, int idx, quant_granularity_t gran) {
    return gran == quant_granularity_t::per_tensor ? s[0] : s[idx];
}
static inline int32_t ref_get_zp(const int32_t* z, int idx, quant_granularity_t gran) {
    if (!z) return 0;
    return gran == quant_granularity_t::per_tensor ? z[0] : z[idx];
}
static inline float ref_get_bias(const float* b, int idx, quant_granularity_t gran) {
    if (!b) return 0.f;
    return gran == quant_granularity_t::per_tensor ? b[0] : b[idx];
}

void run_minigemm_ref_q_fp32(const float* x, int K, int N,
                             const uint8_t* wq_packed, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran,
                             w_dtype_t wtype,
                             bool accumulate,
                             int group_size) {
    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;

    // Precompute sum_x per column if zp compensation is used
    std::vector<float> sumx;
    sumx.resize(N);
    for (int n = 0; n < N; ++n) {
        float sx = 0.f;
        if (zps) {
            const float* xn = x + (size_t)n * K;
            for (int k = 0; k < K; ++k) sx += xn[k];
        }
        sumx[n] = sx;
    }

    auto do_block = [&](int bi, int valid) {
        const uint8_t* wblk = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
        const int base = (gran == quant_granularity_t::per_tensor) ? 0 : bi * M_blk; // per_group handled lane-wise below
        for (int n = 0; n < N; ++n) {
            float acc[M_blk] = {0};
            float* yn = y + (size_t)n * (size_t)M + (size_t)bi * M_blk;
            if (accumulate) {
                for (int m = 0; m < valid; ++m) acc[m] = yn[m];
            }
            if (wtype == w_dtype_t::i4 || wtype == w_dtype_t::u4) {
                for (int k = 0; k < K; ++k) {
                    const float xk = x[(size_t)n * K + k];
                    const uint8_t* bp = wblk + (size_t)k * (M_blk / 2);
                    for (int m = 0; m < valid; ++m) {
                        const int idx = m >> 1;
                        const uint8_t b = bp[idx];
                        uint8_t nib = (m & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                        int32_t q = (wtype == w_dtype_t::u4) ? (int32_t)nib : (int32_t)((nib ^ 0x8) - 0x8);
                        const float s = (gran == quant_granularity_t::per_group)
                            ? scales[((bi*M_blk + m) / (group_size > 0 ? group_size : M_blk))]
                            : ref_get_scale(scales, base + m, gran);
                        acc[m] += (float)q * s * xk;
                    }
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    const float xk = x[(size_t)n * K + k];
                    const uint8_t* wk = wblk + (size_t)k * M_blk;
                    for (int m = 0; m < valid; ++m) {
                        int32_t q = (int32_t)wk[m];
                        if (wtype == w_dtype_t::i8) q = (int32_t)(int8_t)wk[m];
                        const float s = (gran == quant_granularity_t::per_group)
                            ? scales[((bi*M_blk + m) / (group_size > 0 ? group_size : M_blk))]
                            : ref_get_scale(scales, base + m, gran);
                        acc[m] += (float)q * s * xk;
                    }
                }
            }
            // Epilogue per output lane
            for (int m = 0; m < valid; ++m) {
                const int g = ((bi*M_blk + m) / (group_size > 0 ? group_size : M_blk));
                const float s = (gran == quant_granularity_t::per_group)
                    ? scales[g]
                    : ref_get_scale(scales, base + m, gran);
                const float b = (gran == quant_granularity_t::per_group)
                    ? (bias ? bias[g] : 0.f)
                    : ref_get_bias(bias, base + m, gran);
                const float z = (float)((gran == quant_granularity_t::per_group)
                    ? (zps ? zps[g] : 0)
                    : ref_get_zp(zps, base + m, gran));
                acc[m] += b - s * z * sumx[n];
            }
            for (int m = 0; m < valid; ++m) yn[m] = acc[m];
        }
    };

    for (int bi = 0; bi < M_full; ++bi) do_block(bi, M_blk);
    if (M_tail) do_block(M_full, M_tail);
}

} // namespace ov::intel_cpu::x64::gemmv_jit
