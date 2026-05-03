// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// TurboQuant encode: per-head quantize + pack for bits ∈ {3, 4}.
// Output layout: [packed indices | norm_fp32].

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common_quantize.hpp"
#include "nodes/kernels/simd/simd.hpp"
#include "openvino/core/type/element_type.hpp"
#include "turboq_rotation.hpp"
#include "turboq_tables.h"

namespace ov::Extensions::Cpu::XARCH {

// Env-var switches (checked once at static init).
//   OV_TURBOQ_NORM_CORRECTION   store corrected norm so ||dequant|| == ||src||
//   OV_TURBOQ_FUSED_QUANTIZE    use fused path (norm + WHT + pack, single pass)
static const bool g_turboq_norm_correction = std::getenv("OV_TURBOQ_NORM_CORRECTION") != nullptr;
static const bool g_turboq_fused_quantize = std::getenv("OV_TURBOQ_FUSED_QUANTIZE") != nullptr;

// ---------------------------------------------------------------------------
// Bit packing.
// ---------------------------------------------------------------------------

// Pack 4-bit indices into dim/2 bytes. Two indices per byte, low nibble first.
static inline void pack_4bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int i = 0; i < dim / 2; i++) {
        packed[i] = static_cast<uint8_t>((indices[2 * i + 1] << 4) | (indices[2 * i] & 0x0F));
    }
}

// Pack 3-bit indices into dim*3/8 bytes. Groups of 8 indices (8*3=24 bits=3 bytes).
static inline void pack_3bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int g = 0; g < dim / 8; g++) {
        const uint8_t* idx = indices + g * 8;
        uint8_t* dst = packed + g * 3;
        dst[0] = static_cast<uint8_t>((idx[0] & 0x07) | ((idx[1] & 0x07) << 3) | ((idx[2] & 0x03) << 6));
        dst[1] = static_cast<uint8_t>(((idx[2] >> 2) & 0x01) | ((idx[3] & 0x07) << 1) | ((idx[4] & 0x07) << 4) |
                                      ((idx[5] & 0x01) << 7));
        dst[2] = static_cast<uint8_t>(((idx[5] >> 1) & 0x03) | ((idx[6] & 0x07) << 2) | ((idx[7] & 0x07) << 5));
    }
}

// Pack 2-bit indices into dim/4 bytes. 4 indices per byte, LSB first.
static inline void pack_2bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int i = 0; i < dim / 4; i++) {
        packed[i] = static_cast<uint8_t>((indices[4 * i] & 0x03) | ((indices[4 * i + 1] & 0x03) << 2) |
                                         ((indices[4 * i + 2] & 0x03) << 4) | ((indices[4 * i + 3] & 0x03) << 6));
    }
}

// ---------------------------------------------------------------------------
// Fused quantize: norm → sign-flip + normalize → WHT in-place → quantize+pack.
// Single stack buffer; zero heap allocations. Equivalent to the 3-step path.
// ---------------------------------------------------------------------------
// NOTE: the exported entry points below have external linkage; this header
// must only be included by the cross-compiled mha_kv_cache_codec.cpp TU
// (compiled once per ISA with XARCH remapped to AVX512F / AVX2 / ANY).
// Including it elsewhere would cause ODR violations.
static inline void turboq_quantize_head_fused(const float* src, void* dst, float* out_norm, int head_dim, int bits) {
    assert((bits == 3 || bits == 4) && "bits must be 3 or 4");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");
    assert(head_dim <= 512 && "head_dim must be <= 512 for stack buffer");

    const int dim = head_dim;

    // Step 1: SIMD norm.
    constexpr int W = simd::f32::width;
    simd::f32 acc0, acc1;
    {
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
    }
    float norm_sq = reduce(acc0 + acc1);
    float norm = std::sqrt(norm_sq);
    float inv_norm = (norm < 1e-30F) ? 0.0F : 1.0F / norm;

    // Step 2: Fused sign-flip + normalize into single buffer, then WHT in-place.
    float buf[512];
    const float* signs = turboq_get_wht_signs(dim);
    {
        auto vn = simd::f32(inv_norm);
        for (int i = 0; i + W - 1 < dim; i += W) {
            store(simd::load<simd::f32>(src + i) * vn * simd::load<simd::f32>(signs + i), buf + i);
        }
    }
    turboq_wht_inplace(buf, dim);

    // Step 3: Fused quantize + pack directly to output.
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

    // Step 4: Store norm (optionally corrected so ||dequant|| == ||src||) to meta-data slot.
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

// ---------------------------------------------------------------------------
// Reference quantize: normalize → rotate → scalar quantize → pack.
// Uses heap allocations (readable path). Fused variant above avoids them.
// ---------------------------------------------------------------------------
// Entry point: external linkage (resolves against cross-compile dispatcher).
void turboq_quantize_head(const void* src,
                          void* dst,
                          float* out_norm,
                          int head_dim,
                          int bits,
                          ov::element::Type src_precision);

}  // namespace ov::Extensions::Cpu::XARCH
