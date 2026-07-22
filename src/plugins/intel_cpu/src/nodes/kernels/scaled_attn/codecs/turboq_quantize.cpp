// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboq_quantize.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "common_quantize.hpp"
#include "cpu_parallel.hpp"
#include "nodes/kernels/simd/simd.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "turboq_rotation.hpp"
#include "turboq_tables.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Env-var switch (checked once at static init).
//   OV_TURBOQ_NORM_CORRECTION   store corrected norm so ||dequant|| == ||src||
static const bool g_turboq_norm_correction = std::getenv("OV_TURBOQ_NORM_CORRECTION") != nullptr;

// Fused norm + sign-flip + normalize. Templated on input precision T.
// Reads src as T, writes f32 unit*signs into dst, returns the L2 norm.
template <typename T>
inline float turboq_norm_signflip(const T* src, const float* signs, float* dst, int dim) {
    constexpr int W = simd::f32::width;
    simd::f32 acc0, acc1;
    int i = 0;
    for (; i + 2 * W - 1 < dim; i += 2 * W) {
        auto v0 = simd::load<simd::f32>(src + i);
        auto v1 = simd::load<simd::f32>(src + i + W);
        acc0 = fmadd(v0, v0, acc0);
        acc1 = fmadd(v1, v1, acc1);
    }
    for (; i + W - 1 < dim; i += W) {
        auto v0 = simd::load<simd::f32>(src + i);
        acc0 = fmadd(v0, v0, acc0);
    }
    const float norm = std::sqrt(reduce(acc0 + acc1));
    const float inv_norm = (norm < 1e-30F) ? 0.0F : 1.0F / norm;

    auto vn = simd::f32(inv_norm);
    for (i = 0; i + W - 1 < dim; i += W) {
        store(simd::load<simd::f32>(src + i) * vn * simd::load<simd::f32>(signs + i), dst + i);
    }
    return norm;
}

// Dispatch helper: bf16 / f16 / f32 — single pass, returns norm.
static inline float dispatch_norm_signflip(const void* src,
                                           const float* signs,
                                           float* dst,
                                           int dim,
                                           ov::element::Type precision) {
    if (precision == ov::element::bf16) {
        return turboq_norm_signflip(static_cast<const ov::bfloat16*>(src), signs, dst, dim);
    }
    if (precision == ov::element::f16) {
        return turboq_norm_signflip(static_cast<const ov::float16*>(src), signs, dst, dim);
    }
    return turboq_norm_signflip(static_cast<const float*>(src), signs, dst, dim);
}

void turboq_quantize_head(const void* src,
                          void* dst,
                          float* out_norm,
                          int head_dim,
                          int bits,
                          ov::element::Type src_precision,
                          float* ws,
                          const float* signs) {
    assert((bits == 3 || bits == 4) && "bits must be 3 or 4");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");
    assert(ws != nullptr && "per-thread scratch required");
    assert(signs != nullptr && "WHT signs required");

    const int dim = head_dim;

    // Step 1: norm + sign-flip + normalize into ws.
    float* buf = ws;
    const float norm = dispatch_norm_signflip(src, signs, buf, dim, src_precision);

    // Step 2: WHT in-place. Unscaled butterfly; |buf|² = dim after.
    turboq_wht_inplace(buf, dim);

    // Step 3: quantize + pack directly to output.
    auto* out = static_cast<uint8_t*>(dst);
    const float* boundaries = turboq_boundaries(bits, dim);
    const float* codebook = turboq_codebook(bits, dim);
    const int n_bnd = (bits == 4) ? 15 : 7;
    float recon_sq = 0.0F;

    if (bits == 4) {
        for (int i = 0; i < dim; i += 2) {
            uint8_t lo = scalar_quantize(buf[i], boundaries, n_bnd);
            uint8_t hi = scalar_quantize(buf[i + 1], boundaries, n_bnd);
            recon_sq += codebook[lo] * codebook[lo] + codebook[hi] * codebook[hi];
            out[i / 2] = static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
        }
    } else {
        for (int g = 0; g < dim / 8; g++) {
            const float* r = buf + g * 8;
            uint8_t i0 = scalar_quantize(r[0], boundaries, n_bnd);
            uint8_t i1 = scalar_quantize(r[1], boundaries, n_bnd);
            uint8_t i2 = scalar_quantize(r[2], boundaries, n_bnd);
            uint8_t i3 = scalar_quantize(r[3], boundaries, n_bnd);
            uint8_t i4 = scalar_quantize(r[4], boundaries, n_bnd);
            uint8_t i5 = scalar_quantize(r[5], boundaries, n_bnd);
            uint8_t i6 = scalar_quantize(r[6], boundaries, n_bnd);
            uint8_t i7 = scalar_quantize(r[7], boundaries, n_bnd);
            recon_sq += codebook[i0] * codebook[i0] + codebook[i1] * codebook[i1] + codebook[i2] * codebook[i2] +
                        codebook[i3] * codebook[i3] + codebook[i4] * codebook[i4] + codebook[i5] * codebook[i5] +
                        codebook[i6] * codebook[i6] + codebook[i7] * codebook[i7];
            uint8_t* d = out + g * 3;
            d[0] = static_cast<uint8_t>((i0 & 7) | ((i1 & 7) << 3) | ((i2 & 3) << 6));
            d[1] = static_cast<uint8_t>(((i2 >> 2) & 1) | ((i3 & 7) << 1) | ((i4 & 7) << 4) | ((i5 & 1) << 7));
            d[2] = static_cast<uint8_t>(((i5 >> 1) & 3) | ((i6 & 7) << 2) | ((i7 & 7) << 5));
        }
    }

    // Step 4: store norm (optionally corrected so ||dequant|| == ||src||).
    float stored_norm = norm;
    if (g_turboq_norm_correction) {
        float recon_norm = std::sqrt(recon_sq);
        float sqrt_dim = std::sqrt(static_cast<float>(dim));
        if (recon_norm > 1e-30F) {
            stored_norm = norm * sqrt_dim / recon_norm;
        }
    }
    *out_norm = stored_norm;
}

void turboq_quantize(const ov::intel_cpu::PlainTensor& cur,
                     ov::intel_cpu::PlainTensor& dst,
                     ov::intel_cpu::PlainTensor& meta_data,
                     size_t L0,
                     int bits,
                     const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                     ov::Extensions::Cpu::StridedData<float> ws,
                     const ov::intel_cpu::PlainTensor& signs) {
    const auto B = cur.size(0);
    const auto H = cur.size(1);
    const auto L1 = cur.size(2);
    const auto S = cur.size(3);
    const auto prec = cur.get_precision();
    const float* signs_ptr = signs.ptr<float>();
    cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
        float* tws = ws[parallel_get_thread_num()];
        for (size_t l = 0; l < L1; l++) {
            const void* src = cur.ptr_v(b, h, l);
            auto* d = dst.ptr<uint8_t>(b, h, L0 + l);
            auto* out_norm = meta_data.ptr<float>(b, h, L0 + l);
            turboq_quantize_head(src, d, out_norm, static_cast<int>(S), bits, prec, tws, signs_ptr);
        }
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
