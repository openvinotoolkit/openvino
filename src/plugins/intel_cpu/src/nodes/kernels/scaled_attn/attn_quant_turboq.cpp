// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "attn_quant_turboq.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "codecs.hpp"
#include "common.hpp"
#include "nodes/kernels/simd/simd.hpp"
#include "openvino/core/parallel.hpp"
#include "polar_codecs.hpp"
#include "polarq_tables.h"
#include "softmax_kernel.hpp"
#include "turboq_codecs.hpp"
#include "turboq_rotation.hpp"
#include "turboq_tables.h"

namespace ov::Extensions::Cpu::XARCH {

// M_PI is not standard C++ and MSVC does not define it without _USE_MATH_DEFINES.
static constexpr float TURBOQ_PI_F = 3.14159265358979323846F;

// ---------------------------------------------------------------------------
// Debug / experimental env-var switches (checked once at static-init time).
// ---------------------------------------------------------------------------
// Store corrected norm so ||dequant|| == ||original||.  ~0.5% effect at 4-bit,
// ~1.7% at 3-bit, ~6.4% at 2-bit (QJL Lloyd-Max component).
static const bool g_norm_correction = std::getenv("OV_TURBOQ_NORM_CORRECTION") != nullptr;
// Use fused (SIMD norm + WHT in-place + pack) quantize path instead of the
// reference scalar path.  Same output, fewer allocations.
static const bool g_fused_quantize = std::getenv("OV_TURBOQ_FUSED_QUANTIZE") != nullptr;
// Skip Q rotation in the TBQ read path (debug only — breaks dot-product
// correctness, useful for isolating rotation overhead in profiles).
static const bool g_skip_rotation = std::getenv("OV_TURBOQ_SKIP_ROTATION") != nullptr;

// ---------------------------------------------------------------------------
// Scalar quantize — branchless linear scan over boundaries.
// ---------------------------------------------------------------------------

static inline uint8_t turboq_scalar_quantize(float x, const float* boundaries, int n_boundaries) {
    // @todo: benchmark linear vs binary for n=7 and n=15
    int idx = 0;
    for (int i = 0; i < n_boundaries; i++) {
        idx += (x > boundaries[i]) ? 1 : 0;
    }
    return static_cast<uint8_t>(idx);
}

// ---------------------------------------------------------------------------
// Bit packing / unpacking.
// ---------------------------------------------------------------------------

// Pack 4-bit indices into dim/2 bytes.  Two indices per byte, low nibble first.
static void pack_4bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int i = 0; i < dim / 2; i++) {
        packed[i] = static_cast<uint8_t>((indices[2 * i + 1] << 4) | (indices[2 * i] & 0x0F));
    }
}

// Pack 3-bit indices into dim*3/8 bytes.
// Groups of 8 indices (8 * 3 = 24 bits = 3 bytes).
static void pack_3bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int g = 0; g < dim / 8; g++) {
        const uint8_t* idx = indices + g * 8;
        uint8_t* dst = packed + g * 3;

        dst[0] = static_cast<uint8_t>((idx[0] & 0x07) | ((idx[1] & 0x07) << 3) | ((idx[2] & 0x03) << 6));
        dst[1] = static_cast<uint8_t>(((idx[2] >> 2) & 0x01) | ((idx[3] & 0x07) << 1) | ((idx[4] & 0x07) << 4) |
                                      ((idx[5] & 0x01) << 7));
        dst[2] = static_cast<uint8_t>(((idx[5] >> 1) & 0x03) | ((idx[6] & 0x07) << 2) | ((idx[7] & 0x07) << 5));
    }
}

// Pack 2-bit indices into dim/4 bytes.  4 indices per byte, LSB first.
static void pack_2bit(const uint8_t* indices, uint8_t* packed, int dim) {
    for (int i = 0; i < dim / 4; i++) {
        packed[i] = static_cast<uint8_t>((indices[4 * i] & 0x03) | ((indices[4 * i + 1] & 0x03) << 2) |
                                         ((indices[4 * i + 2] & 0x03) << 4) | ((indices[4 * i + 3] & 0x03) << 6));
    }
}

// ---------------------------------------------------------------------------
// Per-head quantize.
// ---------------------------------------------------------------------------

// Compute L2 norm and normalize src to unit vector in f32.
// Dispatches on src_precision to handle bf16/f16→f32 conversion during the load.
template <typename T>
static void normalize_to_unit(const void* src_raw, float* unit, int dim, float& norm) {
    constexpr int W = simd::f32::width;
    const auto* src = static_cast<const T*>(src_raw);

    // SIMD norm: sum of squares.
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
    norm = std::sqrt(reduce(acc0 + acc1));

    // Normalize to unit vector.
    if (norm < 1e-30F) {
        std::memset(unit, 0, dim * sizeof(float));
    } else {
        auto vn = simd::f32(1.0F / norm);
        for (i = 0; i + W - 1 < dim; i += W) {
            store(simd::load<simd::f32>(src + i) * vn, unit + i);
        }
    }
}

static void dispatch_normalize(const void* src, float* unit, int dim, float& norm, ov::element::Type precision) {
    if (precision == ov::element::bf16) {
        normalize_to_unit<ov::bfloat16>(src, unit, dim, norm);
    } else if (precision == ov::element::f16) {
        normalize_to_unit<ov::float16>(src, unit, dim, norm);
    } else {
        normalize_to_unit<float>(src, unit, dim, norm);
    }
}

static void turboq_quantize_head_fused(const float* src, void* dst, int head_dim, int bits);

void turboq_quantize_head(const void* src, void* dst, int head_dim, int bits, ov::element::Type src_precision) {
    if (g_fused_quantize && src_precision == ov::element::f32) {
        turboq_quantize_head_fused(static_cast<const float*>(src), dst, head_dim, bits);
        return;
    }
    assert((bits == 3 || bits == 4) && "bits must be 3 or 4");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");

    const int dim = head_dim;
    const float sqrt_dim = std::sqrt(static_cast<float>(dim));

    // 1-2. Compute L2 norm and normalize to unit vector (handles bf16/f16→f32).
    std::vector<float> unit(dim);
    float norm;
    dispatch_normalize(src, unit.data(), dim, norm, src_precision);

    // 3. Rotate: rotated = R(unit) * sqrt(dim).
    std::vector<float> rotated(dim);
    turboq_rotate_forward(unit.data(), rotated.data(), dim);
    for (int i = 0; i < dim; i++) {
        rotated[i] *= sqrt_dim;
    }

    // 4. Quantize each element to a codebook index.
    std::vector<uint8_t> indices(dim);
    const float* boundaries = turboq_boundaries(bits, dim);
    int n_boundaries = (bits == 4) ? 15 : 7;
    for (int i = 0; i < dim; i++) {
        indices[i] = turboq_scalar_quantize(rotated[i], boundaries, n_boundaries);
    }

    // 5. Pack indices.
    auto* out = static_cast<uint8_t*>(dst);
    if (bits == 4) {
        pack_4bit(indices.data(), out, dim);
    } else {
        pack_3bit(indices.data(), out, dim);
    }

    // 6. Store norm (optionally corrected so ||dequant|| == ||src||).
    float stored_norm = norm;
    if (g_norm_correction) {
        const float* codebook = turboq_codebook(bits, dim);
        float recon_sq = 0.0F;
        for (int i = 0; i < dim; i++) {
            float c = codebook[indices[i]];
            recon_sq += c * c;
        }
        float recon_norm = std::sqrt(recon_sq);
        if (recon_norm > 1e-30F) {
            stored_norm = norm * sqrt_dim / recon_norm;
        }
    }
    auto data_bytes = static_cast<size_t>((dim * bits + 7) / 8);
    std::memcpy(out + data_bytes, &stored_norm, sizeof(stored_norm));
}

// Fused quantize: norm → scale → rotate → quantize+pack → store.
// Zero heap allocations; two stack buffers (WHT requires src != dst).
// SIMD norm computation (AVX-512/AVX2), fused normalize+scale before rotation,
// fused quantize+pack after rotation (no intermediate indices array).
// Fused quantize: norm → sign-flip+normalize → WHT in-place → quantize+pack → store.
// Single stack buffer. Zero heap allocations.
//
// Math: we want R(unit)*sqrt(dim) where R = H·diag(signs)/sqrt(dim).
//   R(unit)*sqrt(dim) = H·diag(signs)·unit/sqrt(dim) * sqrt(dim) = H·diag(signs)·unit
// So the sqrt(dim) factors cancel. The fused pipeline is:
//   buf[i] = src[i]/||src|| * signs[i]      (normalize + sign-flip)
//   wht_inplace(buf)                         (unnormalized Hadamard)
//   quantize+pack buf → out                  (fused, no indices array)
//   store norm
static void turboq_quantize_head_fused(const float* src, void* dst, int head_dim, int bits) {
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
    // buf = diag(signs) · src / ||src||, then wht_inplace(buf) gives H·diag(signs)·unit.
    // This equals R(unit)*sqrt(dim) because the /sqrt(dim) in WHT cancels with *sqrt(dim).
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
            uint8_t lo = turboq_scalar_quantize(buf[i], boundaries, n_bnd);
            uint8_t hi = turboq_scalar_quantize(buf[i + 1], boundaries, n_bnd);
            recon_sq += codebook[lo] * codebook[lo] + codebook[hi] * codebook[hi];
            out[i / 2] = static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
        }
    } else {
        for (int g = 0; g < dim / 8; g++) {
            const float* r = buf + g * 8;
            uint8_t i0 = turboq_scalar_quantize(r[0], boundaries, n_bnd);
            uint8_t i1 = turboq_scalar_quantize(r[1], boundaries, n_bnd);
            uint8_t i2 = turboq_scalar_quantize(r[2], boundaries, n_bnd);
            uint8_t i3 = turboq_scalar_quantize(r[3], boundaries, n_bnd);
            uint8_t i4 = turboq_scalar_quantize(r[4], boundaries, n_bnd);
            uint8_t i5 = turboq_scalar_quantize(r[5], boundaries, n_bnd);
            uint8_t i6 = turboq_scalar_quantize(r[6], boundaries, n_bnd);
            uint8_t i7 = turboq_scalar_quantize(r[7], boundaries, n_bnd);
            recon_sq += codebook[i0] * codebook[i0] + codebook[i1] * codebook[i1] + codebook[i2] * codebook[i2] +
                        codebook[i3] * codebook[i3] + codebook[i4] * codebook[i4] + codebook[i5] * codebook[i5] +
                        codebook[i6] * codebook[i6] + codebook[i7] * codebook[i7];
            uint8_t* d = out + g * 3;
            d[0] = static_cast<uint8_t>((i0 & 7) | ((i1 & 7) << 3) | ((i2 & 3) << 6));
            d[1] = static_cast<uint8_t>(((i2 >> 2) & 1) | ((i3 & 7) << 1) | ((i4 & 7) << 4) | ((i5 & 1) << 7));
            d[2] = static_cast<uint8_t>(((i5 >> 1) & 3) | ((i6 & 7) << 2) | ((i7 & 7) << 5));
        }
    }

    // Step 4: Store norm (optionally corrected so ||dequant|| == ||src||).
    float stored_norm = norm;
    if (g_norm_correction) {
        float recon_norm = std::sqrt(recon_sq);
        float sqrt_dim = std::sqrt(static_cast<float>(dim));
        if (recon_norm > 1e-30F) {
            stored_norm = norm * sqrt_dim / recon_norm;
        }
    }
    auto data_bytes = static_cast<size_t>((dim * bits + 7) / 8);
    std::memcpy(out + data_bytes, &stored_norm, sizeof(stored_norm));
}

// ---------------------------------------------------------------------------
// QJL per-head quantize: (b-1)-bit Lloyd-Max + 1-bit sign correction.
// Output layout: [index_bytes | sign_bytes(dim/8) | gamma_fp32(4) | norm_fp32(4)]
// ---------------------------------------------------------------------------

void turboq_quantize_head_qjl(const void* src, void* dst, int head_dim, int lm_bits, ov::element::Type src_precision) {
    assert((lm_bits == 2 || lm_bits == 3) && "lm_bits must be 2 (tbq3+qjl) or 3 (tbq4+qjl)");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");

    const int dim = head_dim;
    const float sqrt_dim = std::sqrt(static_cast<float>(dim));

    // 1-2. Compute L2 norm and normalize to unit vector (handles bf16/f16→f32).
    std::vector<float> unit(dim);
    float norm;
    dispatch_normalize(src, unit.data(), dim, norm, src_precision);

    // 3. Rotate: rotated = WHT(signs, unit) * sqrt(dim).
    //    (Replaced dense Haar Q*x with O(n log n) WHT+signs.)
    std::vector<float> rotated(dim);
    const float* wht_signs = turboq_get_wht_signs(dim);
    turboq_wht_forward(wht_signs, unit.data(), rotated.data(), dim);
    for (int i = 0; i < dim; i++) {
        rotated[i] *= sqrt_dim;
    }

    // 4. Lloyd-Max quantize with (b-1) bits.
    std::vector<uint8_t> indices(dim);
    const float* codebook = turboq_codebook(lm_bits, dim);
    const float* boundaries = turboq_boundaries(lm_bits, dim);
    int n_boundaries = (1 << lm_bits) - 1;
    for (int i = 0; i < dim; i++) {
        indices[i] = turboq_scalar_quantize(rotated[i], boundaries, n_boundaries);
    }

    // 5. Compute residual and gamma = ||residual||_2.
    std::vector<float> residual(dim);
    float gamma_sq = 0.0F;
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - codebook[indices[i]];
        gamma_sq += residual[i] * residual[i];
    }
    float gamma = std::sqrt(gamma_sq);

    // 6. Project residual through QJL projection, extract sign bits.
    std::vector<float> projected(dim);
    turboq_qjl_project_forward(residual.data(), projected.data(), dim);

    std::vector<uint8_t> sign_bytes(dim / 8, 0);
    for (int i = 0; i < dim; i++) {
        if (projected[i] >= 0.0F) {
            sign_bytes[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
        }
    }

    // 7. Store norm (optionally corrected).
    float stored_norm = norm;
    if (g_norm_correction) {
        float recon_sq = 0.0F;
        for (int i = 0; i < dim; i++) {
            float c = codebook[indices[i]];
            recon_sq += c * c;
        }
        float recon_norm = std::sqrt(recon_sq);
        if (recon_norm > 1e-30F) {
            stored_norm = norm * sqrt_dim / recon_norm;
        }
    }

    // 8. Pack output: [indices | signs | gamma | norm]
    //    Norm is always the last 4 bytes — same convention as standard TBQ.
    auto* out = static_cast<uint8_t*>(dst);
    auto index_bytes = static_cast<size_t>((dim * lm_bits + 7) / 8);
    if (lm_bits == 3) {
        pack_3bit(indices.data(), out, dim);
    } else {
        pack_2bit(indices.data(), out, dim);
    }
    std::memcpy(out + index_bytes, sign_bytes.data(), sign_bytes.size());
    std::memcpy(out + index_bytes + sign_bytes.size(), &gamma, sizeof(gamma));
    std::memcpy(out + index_bytes + sign_bytes.size() + 4, &stored_norm, sizeof(stored_norm));
}

// ---------------------------------------------------------------------------
// Generic codec dot product and weighted accumulation templates.
// These are the shared inner SIMD loops used by all codec types (TBQ, Polar,
// u8, u4). Metadata reading (norm, scale/zp) is handled by callers.
// ---------------------------------------------------------------------------

// Lightweight strided pointer — wraps (base, stride) for head-strided arrays.
// Usage: data[head][offset]
template <typename T>
struct StridedData {
    T* data;
    size_t stride;

    T* operator[](size_t i) const {
        return data + i * stride;
    }
};

// Decode W elements at element index j from packed data.
// Computes bit position from j and codec::bits, handles sub-byte offsets.
// For SIMD paths where (bits * W) % 8 == 0, bit offset is effectively 0.
// Bit offset is used instead of byte offset to unify logic for scalar reference case.
template <typename Codec>
static inline auto decode_at(const uint8_t* data, int j, const Codec& codec) {
    int b = j * Codec::bits;
    return codec.decode(data + b / 8, b % 8);
}

// Generic dot product: sum(codec.decode(data[j]) * q[j]) for j in [0, dim).
// Returns raw dot product — caller applies outer scale (norm*inv_sqrt_dim for TBQ,
// group_scale for u8/u4).
// QT: query element type (float, float16, bfloat16). simd::load handles conversion.
template <typename QT, typename Codec>
static inline float codec_dot(const uint8_t* data, const QT* q, int dim, const Codec& codec) {
    constexpr int W = simd::f32::width;
    simd::f32 dot0, dot1, dot2, dot3;
    int j = 0;
    for (; j + 4 * W - 1 < dim; j += 4 * W) {
        dot0 = fmadd(simd::load<simd::f32>(q + j), decode_at(data, j, codec), dot0);
        dot1 = fmadd(simd::load<simd::f32>(q + j + W), decode_at(data, j + W, codec), dot1);
        dot2 = fmadd(simd::load<simd::f32>(q + j + 2 * W), decode_at(data, j + 2 * W, codec), dot2);
        dot3 = fmadd(simd::load<simd::f32>(q + j + 3 * W), decode_at(data, j + 3 * W, codec), dot3);
    }
    for (; j + W - 1 < dim; j += W) {
        dot0 = fmadd(simd::load<simd::f32>(q + j), decode_at(data, j, codec), dot0);
    }
    return reduce((dot0 + dot1) + (dot2 + dot3));
}

// Generic weighted V accumulation: for each element j, decode once and accumulate
// into all n_heads output buffers with weight * outer_scale.
// Decodes 4*W elements per main-loop iteration, applies to all heads (GQA pattern).
template <typename Codec>
static inline void codec_weighted_accum(const uint8_t* data,
                                        int dim,
                                        const Codec& codec,
                                        float outer_scale,
                                        StridedData<const float> weights,
                                        StridedData<float> accum,
                                        int n_heads) {
    constexpr int W = simd::f32::width;
    int j = 0;
    for (; j + 4 * W - 1 < dim; j += 4 * W) {
        auto v0 = decode_at(data, j, codec);
        auto v1 = decode_at(data, j + W, codec);
        auto v2 = decode_at(data, j + 2 * W, codec);
        auto v3 = decode_at(data, j + 3 * W, codec);
        for (int h = 0; h < n_heads; h++) {
            auto vw = simd::f32(weights[h][0] * outer_scale);
            float* out = accum[h] + j;
            store(fmadd(vw, v0, simd::load<simd::f32>(out)), out);
            store(fmadd(vw, v1, simd::load<simd::f32>(out + W)), out + W);
            store(fmadd(vw, v2, simd::load<simd::f32>(out + 2 * W)), out + 2 * W);
            store(fmadd(vw, v3, simd::load<simd::f32>(out + 3 * W)), out + 3 * W);
        }
    }
    for (; j + W - 1 < dim; j += W) {
        auto v0 = decode_at(data, j, codec);
        for (int h = 0; h < n_heads; h++) {
            auto vw = simd::f32(weights[h][0] * outer_scale);
            float* out = accum[h] + j;
            store(fmadd(vw, v0, simd::load<simd::f32>(out)), out);
        }
    }
}

// ---------------------------------------------------------------------------
// QJL sign-bit helpers.
// ---------------------------------------------------------------------------

// Expand packed sign bits to ±1.0 float vector.
// Expand packed sign bits at byte pointer to +/-1.0 float vector.
// Used by SignCodec::decode.
template <simd::isa i = simd::active_isa>
static inline simd::vec<float, i> expand_signs_at(const uint8_t* p) {
    simd::vec<float, i> neg(-1.0F), pos(1.0F);
    if constexpr (i == simd::isa::avx512) {
        simd::mask<i> mask{read_as<uint16_t>(p)};
        return select(mask, neg, pos);
    } else if constexpr (i == simd::isa::avx2) {
        // Expand 8 packed bits to per-lane mask: broadcast byte, AND with per-lane bit,
        // compare equal → all-ones where bit is set.
        static constexpr int32_t bit_masks_data[] = {1, 2, 4, 8, 16, 32, 64, 128};
        auto bit_masks = simd::load<simd::vec<int32_t, i>>(bit_masks_data);
        auto bits = simd::vec<int32_t, i>{static_cast<int32_t>(*p)};
        auto mask = (bits & bit_masks) == bit_masks;
        return select(mask, neg, pos);
    } else {
        bool s = (*p & 1) != 0;
        return {s ? 1.0F : -1.0F};
    }
}

// Sign codec: expands packed 1-bit sign data to +/-1.0. Compatible with codec_dot.
struct SignCodec {
    static constexpr int bits = 1;

    static inline simd::f32 decode(const uint8_t* base, int /*bit_offset*/) {
        return expand_signs_at(base);
    }
};

// Legacy wrapper: expand signs by element index.
// Reads simd::f32::width consecutive bits starting at signs[j/8], bit j%8.
template <simd::isa i = simd::active_isa>
static inline simd::vec<float, i> expand_signs(const uint8_t* signs, int j) {
    if constexpr (i == simd::isa::scalar) {
        bool s = ((signs[j / 8] >> (j % 8)) & 1) != 0;
        return {s ? 1.0F : -1.0F};
    } else {
        return expand_signs_at<i>(signs + j / 8);
    }
}

// QJL-aware Q·K dot: base codebook dot + sign correction.
// Record layout: [index_bytes | sign_bytes | gamma_fp32 | norm_fp32].
// q_packed layout: [rotated_q (dim) | projected_q (dim)].
// QJL QK dot: base codebook dot + sign correction.
// record_bytes is the full record size (includes indices + signs + gamma + norm).
// Offsets: norm = record_bytes-4, gamma = record_bytes-8, signs start = record_bytes-8-dim/8.
template <typename InnerCodec>
static inline float turboq_codec_qk_dot_qjl(const uint8_t* in,
                                            const float* q_packed,
                                            const InnerCodec& inner,
                                            size_t record_bytes,
                                            int dim) {
    // Record layout: [indices | signs | gamma(4) | norm(4)]
    const size_t gamma_off = record_bytes - 8;
    const size_t sign_off = gamma_off - static_cast<size_t>(dim / 8);

    // Per-record scalars. base_scale already folds in 1/sqrt(dim), so sign_scale
    // reuses it directly — no separate norm read or inv_sqrt_dim needed.
    const float base_scale = turboq_norm_scale(in, record_bytes, dim);  // norm / sqrt(dim)
    const float gamma = read_as<float>(in + gamma_off);
    const float qjl_coeff = std::sqrt(TURBOQ_PI_F / 2.0F) / static_cast<float>(dim);
    const float sign_scale = base_scale * qjl_coeff * gamma;

    // Base codebook dot (against rotated_q) + sign correction (against projected_q).
    const float base_dot = codec_dot<float>(in, q_packed, dim, inner) * base_scale;
    const float sign_dot = codec_dot<float>(in + sign_off, q_packed + dim, dim, SignCodec{});
    return base_dot + sign_scale * sign_dot;
}

// Prefetch lookahead: number of records ahead to prefetch.
// At ~50 cycles/record and ~200 cycle DRAM latency, 4-8 records ahead hides latency.
static constexpr int PREFETCH_AHEAD = 8;

// Reduce thread-local accumulators into dst, optionally fuse QJL sign correction,
// then optionally apply Q^T inverse rotation.
// When sign_src != nullptr: reduce sign accum, apply S^T, add correction before Q^T.

// Sum per-thread accumulators: dst[i] = sum_t(src[i + t * stride]) for i in [0, dim).
// When T != float, converts f32 accumulator result to T on store.
template <typename T>
static void turboq_reduce_threads(T* dst, const float* src, size_t stride, int dim, int nthr) {
    constexpr int W = simd::f32::width;
    for (int i = 0; i < dim; i += W) {
        auto acc = simd::load<simd::f32>(src + i);
        const float* p = src + i + stride;
        for (int t = 1; t < nthr; t++) {
            acc = acc + simd::load<simd::f32>(p);
            p += stride;
        }
        store(acc, dst + i);
    }
}

// Reduce sign accumulators across threads, apply inverse QJL projection (S^T),
// and add scaled correction: dst[i] = base[i] + sqrt(pi/2)/dim * S^T(sum_threads(sign_accum))[i].
// When T != float, converts the final result to T on store.
template <typename T>
static void turboq_reduce_qjl_correction(T* dst, float* base, float* sign_src, size_t sign_stride, int dim, int nthr) {
    constexpr int W = simd::f32::width;
    // Reduce sign accumulators in-place into sign_src (reusable workspace).
    turboq_reduce_threads(sign_src, sign_src, sign_stride, dim, nthr);
    // WHT in-place on sign_src, scale+signs back to sign_src.
    turboq_qjl_project_inverse(sign_src, sign_src, dim);
    // base[i] + coeff * sign_src[i] → dst (typed output)
    simd::f32 coeff(std::sqrt(TURBOQ_PI_F / 2.0F) / static_cast<float>(dim));
    for (int i = 0; i < dim; i += W) {
        store(fmadd(coeff, simd::load<simd::f32>(sign_src + i), simd::load<simd::f32>(base + i)), dst + i);
    }
}

// Typed reduce: reduces thread accumulators + optional QJL + optional inverse rotation,
// writing the final result directly to typed output dst.
// src (thread-0 accumulator) is reused as f32 workspace when further steps follow.
template <typename T>
static void turboq_reduce_head_impl(int dim,
                                    T* dst,
                                    float* src,
                                    int nthr,
                                    size_t stride,
                                    bool apply_inv_rotation,
                                    float* sign_src,
                                    size_t sign_stride) {
    assert(dim % 64 == 0 && "dim must be divisible by 64");
    assert(dim <= 256 && "dim must be <= 256 for stack buffers");

    const bool has_qjl = sign_src != nullptr;

    if (!has_qjl && !apply_inv_rotation) {
        // reduce_threads is final step → convert directly to dst
        turboq_reduce_threads(dst, src, stride, dim, nthr);
    } else if (has_qjl && !apply_inv_rotation) {
        // reduce_threads → f32 src, qjl_correction is final step → convert to dst
        turboq_reduce_threads(src, src, stride, dim, nthr);
        turboq_reduce_qjl_correction(dst, src, sign_src, sign_stride, dim, nthr);
    } else if (!has_qjl && apply_inv_rotation) {
        // reduce_threads → f32 src, rotate_inverse is final step → write to dst
        turboq_reduce_threads(src, src, stride, dim, nthr);
        turboq_rotate_inverse(src, dst, dim);
    } else {
        // reduce_threads → f32 src, qjl_correction → f32 src, rotate_inverse → dst
        turboq_reduce_threads(src, src, stride, dim, nthr);
        turboq_reduce_qjl_correction(src, src, sign_src, sign_stride, dim, nthr);
        turboq_rotate_inverse(src, dst, dim);
    }
}

void turboq_reduce_head(int dim,
                        float* dst,
                        float* src,
                        int nthr,
                        size_t stride,
                        bool apply_inv_rotation,
                        float* sign_src,
                        size_t sign_stride) {
    turboq_reduce_head_impl(dim, dst, src, nthr, stride, apply_inv_rotation, sign_src, sign_stride);
}

size_t turboq_head_bytes(int head_dim, int bits) {
    // packed index bytes + 4 bytes for fp32 norm
    return static_cast<size_t>((head_dim * bits + 7) / 8) + 4;
}

size_t turboq_row_bytes(int num_kv_heads, int head_dim, int bits) {
    return static_cast<size_t>(num_kv_heads) * turboq_head_bytes(head_dim, bits);
}

size_t turboq_head_bytes_qjl(int head_dim, int lm_bits) {
    // packed (b-1)-bit indices + 16 bytes signs + 4 bytes fp32 norm + 4 bytes fp32 gamma
    return static_cast<size_t>((head_dim * lm_bits + 7) / 8) + static_cast<size_t>(head_dim / 8) + 4 + 4;
}

// ===========================================================================
// PolarQuant implementation — integrated into the TBQ pipeline.
// ===========================================================================

// ---------------------------------------------------------------------------
// Polar helpers.
// ---------------------------------------------------------------------------

// Delegates to the dim-aware accessor in polarq_tables.h.
static inline const int* polar_get_bits_per_level(int bits, int head_dim) {
    return polarq_get_bits_per_level(bits, head_dim);
}

static inline int polar_tree_depth(int dim) {
    int L = 0;
    for (int d = dim; d > 1; d >>= 1) {
        L++;
    }
    return L;
}

// Branchless linear scan over boundaries (same pattern as TBQ).
static inline uint8_t polar_scalar_quantize(float x, const float* boundaries, int n_boundaries) {
    int idx = 0;
    for (int i = 0; i < n_boundaries; i++) {
        idx += (x > boundaries[i]) ? 1 : 0;
    }
    return static_cast<uint8_t>(idx);
}

// ---------------------------------------------------------------------------
// Polar mixed-width bit packing.
// Each level's angles are packed contiguously at uniform width within the level.
// ---------------------------------------------------------------------------

// Pack n indices of bit_width bits into dst. Returns bytes written.
static size_t polar_pack_indices(const uint8_t* indices, uint8_t* dst, int n, int bit_width) {
    if (bit_width == 0) {
        return 0;
    }
    int total_bytes = (n * bit_width + 7) / 8;
    std::memset(dst, 0, total_bytes);

    int bit_pos = 0;
    for (int i = 0; i < n; i++) {
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        auto val = static_cast<uint16_t>(indices[i] & ((1 << bit_width) - 1));
        dst[byte_idx] |= static_cast<uint8_t>(val << bit_off);
        if (bit_off + bit_width > 8) {
            dst[byte_idx + 1] |= static_cast<uint8_t>(val >> (8 - bit_off));
        }
        bit_pos += bit_width;
    }
    return static_cast<size_t>(total_bytes);
}

// polar_unpack_indices and per-width scalar unpackers removed —
// superseded by SIMD unpack_2bit/3bit/4bit/5bit in codecs.hpp
// and PolarCodec structs in polar_codecs.hpp.

// ---------------------------------------------------------------------------
// PolarQuant quantize.
// ---------------------------------------------------------------------------

void polarq_quantize_head(const void* src, void* dst, int head_dim, int bits, ov::element::Type src_precision) {
    assert((bits == 3 || bits == 4) && "bits must be 3 or 4");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");
    assert((head_dim & (head_dim - 1)) == 0 && "head_dim must be power of 2");

    const int dim = head_dim;
    const int L = polar_tree_depth(dim);
    const float sqrt_dim = std::sqrt(static_cast<float>(dim));
    const int* bpl = polar_get_bits_per_level(bits, head_dim);

    // 1-2. Compute L2 norm and normalize to unit vector (handles bf16/f16→f32).
    std::vector<float> unit(dim);
    float norm;
    dispatch_normalize(src, unit.data(), dim, norm, src_precision);

    // 3. Rotate: rotated = R(unit) * sqrt(dim), using the active rotation mode.
    std::vector<float> rotated(dim);
    turboq_rotate_forward(unit.data(), rotated.data(), dim);
    for (int i = 0; i < dim; i++) {
        rotated[i] *= sqrt_dim;
    }

    // 4. Recursive polar decomposition + quantization.
    auto* out = static_cast<uint8_t*>(dst);
    size_t write_off = 0;

    // Working buffer: starts as dim/2 radii, halves each level.
    std::vector<float> radii(dim / 2);
    uint8_t idx_buf[256];  // max angles at any level

    // Level 1: pair Cartesian coordinates → (angle, radius)
    {
        int n = dim / 2;
        int bit_width = bpl[0];
        for (int i = 0; i < n; i++) {
            float x = rotated[2 * i];
            float y = rotated[2 * i + 1];
            float angle = std::atan2(y, x);
            if (angle < 0.0F) {
                angle += 2.0F * TURBOQ_PI_F;
            }
            radii[i] = std::sqrt(x * x + y * y);
            if (bit_width > 0) {
                auto lut = polarq_get_lut(1, bit_width);
                idx_buf[i] = polar_scalar_quantize(angle, lut.boundaries, lut.n_centroids - 1);
            }
        }
        write_off += polar_pack_indices(idx_buf, out + write_off, n, bit_width);
    }

    // Levels 2..L: pair radii from previous level
    for (int level = 2; level <= L; level++) {
        int n = dim >> level;
        int bit_width = bpl[level - 1];
        std::vector<float> new_radii(n);
        for (int i = 0; i < n; i++) {
            float re = radii[2 * i];
            float ro = radii[2 * i + 1];
            float angle = std::atan2(ro, re);
            new_radii[i] = std::sqrt(re * re + ro * ro);
            if (bit_width > 0) {
                auto lut = polarq_get_lut(level, bit_width);
                idx_buf[i] = polar_scalar_quantize(angle, lut.boundaries, lut.n_centroids - 1);
            }
        }
        write_off += polar_pack_indices(idx_buf, out + write_off, n, bit_width);
        radii.resize(n);
        std::memcpy(radii.data(), new_radii.data(), n * sizeof(float));
    }

    // 5. Store fp32 norm at end.
    std::memcpy(out + write_off, &norm, sizeof(norm));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// PolarQuant SIMD kernels.
// ---------------------------------------------------------------------------

// Interleave-and-store: given two simd::f32 vectors a[] and b[], write
// [a0,b0,a1,b1,...] to dst. Processes simd::f32::width pairs per call.
template <simd::isa i = simd::active_isa>
static inline void interleave_store(float* dst, simd::vec<float, i> a, simd::vec<float, i> b) {
    if constexpr (i == simd::isa::avx512) {
        static constexpr int32_t perm0_data[] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
        static constexpr int32_t perm1_data[] = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
        auto perm0 = simd::load<simd::vec<int32_t, i>>(perm0_data);
        auto perm1 = simd::load<simd::vec<int32_t, i>>(perm1_data);
        store(permute2(a, perm0, b), dst);
        store(permute2(a, perm1, b), dst + simd::vec<float, i>::width);
    } else if constexpr (i == simd::isa::avx2) {
        // AVX2 lacks two-source permute — use unpack + lane permute.
        auto lo = unpack_lo(a, b);                       // [a0,b0,a1,b1, a4,b4,a5,b5]
        auto hi = unpack_hi(a, b);                       // [a2,b2,a3,b3, a6,b6,a7,b7]
        constexpr int LO_LANES = simd::lane_ctrl(0, 2);  // [lo.lo, hi.lo]
        constexpr int HI_LANES = simd::lane_ctrl(1, 3);  // [lo.hi, hi.hi]
        store(simd::permute_lanes<LO_LANES>(lo, hi), dst);
        store(simd::permute_lanes<HI_LANES>(lo, hi), dst + simd::vec<float, i>::width);
    } else {
        store(a, dst);
        store(b, dst + 1);
    }
}

static inline uint8_t polar_extract_index(const uint8_t* packed, int elem_idx, int bit_width) {
    const int bit_pos = elem_idx * bit_width;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    uint16_t raw = static_cast<uint16_t>(packed[byte_idx]) | (static_cast<uint16_t>(packed[byte_idx + 1]) << 8);
    return static_cast<uint8_t>((raw >> bit_off) & ((1 << bit_width) - 1));
}

template <typename Codec>
static inline int polar_reconstruct_level_simd(const uint8_t* packed,
                                               const Codec& codec,
                                               const float* radii,
                                               float* new_radii,
                                               int num_angles) {
    constexpr int W = simd::f32::width;
    constexpr int GB = Codec::bits * W;
    constexpr int BYTE_STRIDE = GB / 8;
    constexpr int BIT_SUB = GB % 8;
    const uint8_t* p = packed;
    int bit = 0;
    int i = 0;
    for (; i + W <= num_angles; i += W, p += BYTE_STRIDE, bit += BIT_SUB) {
        simd::f32 rc, rs;
        codec.decode(p + bit / 8, bit % 8, rc, rs);
        auto rp = simd::load<simd::f32>(radii + i);
        interleave_store(new_radii + 2 * i, rp * rc, rp * rs);
    }
    return i;
}

template <typename Codec>
static inline int polar_apply_level1_simd(const uint8_t* packed,
                                          const Codec& codec,
                                          const float* radii,
                                          float* rc_buf,
                                          float* rs_buf,
                                          int num_angles) {
    constexpr int W = simd::f32::width;
    constexpr int GB = Codec::bits * W;
    constexpr int BYTE_STRIDE = GB / 8;
    constexpr int BIT_SUB = GB % 8;
    const uint8_t* p = packed;
    int bit = 0;
    int i = 0;
    for (; i + W <= num_angles; i += W, p += BYTE_STRIDE, bit += BIT_SUB) {
        simd::f32 rc, rs;
        codec.decode(p + bit / 8, bit % 8, rc, rs);
        auto rp = simd::load<simd::f32>(radii + i);
        store(rp * rc, rc_buf + i);
        store(rp * rs, rs_buf + i);
    }
    return i;
}

// Precomputed per-token polar data, shared across all heads.
struct PolarTokenData {
    float rc_buf[128];  // l1r[i] * cos[i]
    float rs_buf[128];  // l1r[i] * sin[i]
    float norm_scale;   // norm * inv_sqrt_dim
    int n_l1;
};

// Precompute all per-token polar data in a single pass:
// 1. Compute level byte layout + norm offset (once)
// 2. Unpack all level indices (once — no double unpack)
// 3. Top-down tree reconstruction (levels L..2) → level-1 radii
// 4. Apply level-1 cos/sin → rc_buf/rs_buf
static void polar_precompute_token_simd(const uint8_t* record, PolarTokenData& td, int head_dim, const int* bpl) {
    const int L = polar_tree_depth(head_dim);
    constexpr int W = simd::f32::width;

    // Compute level layout and norm offset in a single pass.
    int n_angles[POLARQ_MAX_LEVELS];
    int bit_widths[POLARQ_MAX_LEVELS];
    size_t byte_offsets[POLARQ_MAX_LEVELS];
    size_t norm_off = 0;
    for (int k = 0; k < L; k++) {
        n_angles[k] = head_dim >> (k + 1);
        bit_widths[k] = bpl[k];
        byte_offsets[k] = norm_off;
        norm_off += static_cast<size_t>((n_angles[k] * bit_widths[k] + 7) / 8);
    }

    const float inv_sqrt_dim = 1.0F / std::sqrt(static_cast<float>(head_dim));
    td.norm_scale = read_as<float>(record + norm_off) * inv_sqrt_dim;
    td.n_l1 = head_dim / 2;

    // Top-down tree reconstruction: levels L..2 → level-1 radii.
    float sqrt_dim = std::sqrt(static_cast<float>(head_dim));
    float radii[256];
    radii[0] = sqrt_dim;

    for (int level = L; level >= 2; level--) {
        int k = level - 1;
        int num_angles = n_angles[k];
        int bit_width = bit_widths[k];
        float new_radii[256];
        int i = 0;
        if (bit_width > 0) {
            auto lut = polarq_get_lut(level, bit_width);
            switch (bit_width) {
            case 5:
                i = polar_reconstruct_level_simd(record + byte_offsets[k],
                                                 PolarCodec5<>{lut},
                                                 radii,
                                                 new_radii,
                                                 num_angles);
                break;
            case 4:
                i = polar_reconstruct_level_simd(record + byte_offsets[k],
                                                 PolarCodec4<>{lut},
                                                 radii,
                                                 new_radii,
                                                 num_angles);
                break;
            case 3:
                i = polar_reconstruct_level_simd(record + byte_offsets[k],
                                                 PolarCodec3<>{lut},
                                                 radii,
                                                 new_radii,
                                                 num_angles);
                break;
            case 2:
                i = polar_reconstruct_level_simd(record + byte_offsets[k],
                                                 PolarCodec2<>{lut},
                                                 radii,
                                                 new_radii,
                                                 num_angles);
                break;
            default:
                break;
            }
            for (; i < num_angles; i++) {
                const uint8_t idx = polar_extract_index(record + byte_offsets[k], i, bit_width);
                assert(idx < lut.n_centroids && "polar index out of range");
                new_radii[2 * i] = radii[i] * lut.cos_lut[idx];
                new_radii[2 * i + 1] = radii[i] * lut.sin_lut[idx];
            }
        } else {
            auto fixed_cs = simd::f32(POLARQ_FIXED_COS);
            for (; i + W <= num_angles; i += W) {
                auto rc = simd::load<simd::f32>(radii + i) * fixed_cs;
                interleave_store(new_radii + 2 * i, rc, rc);
            }
        }
        for (; i < num_angles; i++) {
            float cv_s = POLARQ_FIXED_COS;
            float sv_s = POLARQ_FIXED_SIN;
            new_radii[2 * i] = radii[i] * cv_s;
            new_radii[2 * i + 1] = radii[i] * sv_s;
        }

        std::memcpy(radii, new_radii, num_angles * 2 * sizeof(float));
    }

    // Level 1: apply cos/sin to radii → rc_buf/rs_buf.
    int bw1 = bit_widths[0];
    if (bw1 > 0) {
        auto lut = polarq_get_lut(1, bw1);
        int i = 0;
        switch (bw1) {
        case 5:
            i = polar_apply_level1_simd(record + byte_offsets[0],
                                        PolarCodec5<>{lut},
                                        radii,
                                        td.rc_buf,
                                        td.rs_buf,
                                        td.n_l1);
            break;
        case 4:
            i = polar_apply_level1_simd(record + byte_offsets[0],
                                        PolarCodec4<>{lut},
                                        radii,
                                        td.rc_buf,
                                        td.rs_buf,
                                        td.n_l1);
            break;
        case 3:
            i = polar_apply_level1_simd(record + byte_offsets[0],
                                        PolarCodec3<>{lut},
                                        radii,
                                        td.rc_buf,
                                        td.rs_buf,
                                        td.n_l1);
            break;
        case 2:
            i = polar_apply_level1_simd(record + byte_offsets[0],
                                        PolarCodec2<>{lut},
                                        radii,
                                        td.rc_buf,
                                        td.rs_buf,
                                        td.n_l1);
            break;
        default:
            break;
        }
        for (; i < td.n_l1; i++) {
            const uint8_t idx = polar_extract_index(record + byte_offsets[0], i, bw1);
            assert(idx < lut.n_centroids && "polar index out of range");
            td.rc_buf[i] = radii[i] * lut.cos_lut[idx];
            td.rs_buf[i] = radii[i] * lut.sin_lut[idx];
        }
    } else {
        for (int i = 0; i < td.n_l1; i++) {
            td.rc_buf[i] = radii[i] * POLARQ_FIXED_COS;
            td.rs_buf[i] = radii[i] * POLARQ_FIXED_SIN;
        }
    }
}

// Deinterleave-load: read [a0,b0,a1,b1,...] from src, return even (a) and odd (b) vectors.
// Inverse of interleave_store.
template <simd::isa i = simd::active_isa>
static inline void deinterleave_load(const float* src, simd::vec<float, i>& even, simd::vec<float, i>& odd) {
    if constexpr (i == simd::isa::avx512) {
        static constexpr int32_t even_data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        static constexpr int32_t odd_data[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
        auto even_idx = simd::load<simd::vec<int32_t, i>>(even_data);
        auto odd_idx = simd::load<simd::vec<int32_t, i>>(odd_data);
        auto a = simd::load<simd::vec<float, i>>(src);
        auto b = simd::load<simd::vec<float, i>>(src + 16);
        even = permute2(a, even_idx, b);
        odd = permute2(a, odd_idx, b);
    } else if constexpr (i == simd::isa::avx2) {
        // AVX2 deinterleave: 3 steps to separate even/odd elements.
        // Input: [a0,b0,a1,b1,a2,b2,a3,b3, a4,b4,a5,b5,a6,b6,a7,b7]
        auto lo = simd::load<simd::vec<float, i>>(src);
        auto hi = simd::load<simd::vec<float, i>>(src + 8);
        // Step 1: group even/odd pairs within each 128-bit lane.
        constexpr int GROUP_PAIRS = simd::shuffle_ctrl(3, 1, 2, 0);  // 0xD8
        auto slo = simd::shuffle<GROUP_PAIRS>(lo, lo);               // [a0,a1,b0,b1, a2,a3,b2,b3]
        auto shi = simd::shuffle<GROUP_PAIRS>(hi, hi);               // [a4,a5,b4,b5, a6,a7,b6,b7]
        // Step 2: merge 64-bit pairs across sources.
        auto aa = simd::unpack_lo_64(slo, shi);  // [a0,a1,a4,a5, a2,a3,a6,a7]
        auto bb = simd::unpack_hi_64(slo, shi);  // [b0,b1,b4,b5, b2,b3,b6,b7]
        // Step 3: fix lane ordering with 64-bit cross-lane permute.
        constexpr int FIX_LANES = simd::shuffle_ctrl(3, 1, 2, 0);  // 0xD8
        even = simd::permute_64<FIX_LANES>(aa);
        odd = simd::permute_64<FIX_LANES>(bb);
    } else {
        even = {src[0]};
        odd = {src[1]};
    }
}

static float polar_qk_dot_precomputed_simd(const PolarTokenData& td, const float* q_rotated) {
    constexpr int W = simd::f32::width;
    int n_l1 = td.n_l1;
    simd::f32 dot_acc;
    int i = 0;
    for (; i + W <= n_l1; i += W) {
        auto rc = simd::load<simd::f32>(td.rc_buf + i);
        auto rs = simd::load<simd::f32>(td.rs_buf + i);
        simd::f32 q_even, q_odd;
        deinterleave_load(q_rotated + 2 * i, q_even, q_odd);
        dot_acc = fmadd(rc, q_even, fmadd(rs, q_odd, dot_acc));
    }
    float dot = reduce(dot_acc);
    for (; i < n_l1; i++) {
        dot += td.rc_buf[i] * q_rotated[2 * i] + td.rs_buf[i] * q_rotated[2 * i + 1];
    }
    return dot * td.norm_scale;
}

static void polar_v_accum_precomputed_simd(const PolarTokenData& td,
                                           StridedData<const float> weights,
                                           StridedData<float> accum,
                                           int num_heads,
                                           int head_dim) {
    constexpr int W = simd::f32::width;
    auto vs = simd::f32(td.norm_scale);
    int n_l1 = td.n_l1;

    // Phase 1: Reconstruct interleaved cartesian: [rc0*s, rs0*s, rc1*s, rs1*s, ...]
    float cartesian[256];
    for (int i = 0; i + W <= n_l1; i += W) {
        interleave_store(cartesian + 2 * i,
                         simd::load<simd::f32>(td.rc_buf + i) * vs,
                         simd::load<simd::f32>(td.rs_buf + i) * vs);
    }
    for (int i = (n_l1 / W) * W; i < n_l1; i++) {
        cartesian[2 * i] = td.rc_buf[i] * td.norm_scale;
        cartesian[2 * i + 1] = td.rs_buf[i] * td.norm_scale;
    }

    // Phase 2: Standard weighted accumulation through RawCodec.
    codec_weighted_accum(reinterpret_cast<const uint8_t*>(cartesian),
                         head_dim,
                         RawCodec<float>{},
                         1.0F,
                         weights,
                         accum,
                         num_heads);
}

// ---------------------------------------------------------------------------
// PolarQuant batch wrappers.
// ---------------------------------------------------------------------------

// Per-token polar QK dot: one K token, one Q head. Returns the score.
static inline float polar_token_qk_dot(const uint8_t* k_ptr, const float* q, int head_dim, const int* bpl) {
    PolarTokenData td{};
    polar_precompute_token_simd(k_ptr, td, head_dim, bpl);
    return polar_qk_dot_precomputed_simd(td, q);
}

// Per-token polar V accum. weights[h][0] is the softmax weight for head h.
static inline void polar_token_v_accum(const uint8_t* v_ptr,
                                       StridedData<const float> weights,
                                       StridedData<float> accum,
                                       int n_heads,
                                       int head_dim,
                                       const int* bpl) {
    PolarTokenData td{};
    polar_precompute_token_simd(v_ptr, td, head_dim, bpl);
    polar_v_accum_precomputed_simd(td, weights, accum, n_heads, head_dim);
}

// ---------------------------------------------------------------------------
// PolarQuant head/row byte sizes.
// ---------------------------------------------------------------------------

size_t polarq_head_bytes(int head_dim, int bits) {
    const int* bpl = polar_get_bits_per_level(bits, head_dim);
    return polarq_head_bytes_from_alloc(head_dim, bpl);
}

// ---------------------------------------------------------------------------
// mha_turboq — TBQ fused multi-head attention (4-phase pipeline).
// ---------------------------------------------------------------------------

using ov::intel_cpu::CpuParallelPtr;
using ov::intel_cpu::PlainTensor;

// Prepare Q for attention: convert to f32 and optionally rotate + project.
// When rotate=true: applies forward rotation (R*q) and optional QJL projection (SQ*q).
// When rotate=false: just converts Q to f32 (for non-codec K paths).
// q_precision: element type of q_input (f32, bf16, or f16).
// When qjl=true, output is packed as [rotated | projected] per head (2*S floats).
static void turboq_prepare_query(const PlainTensor& q_input,
                                 PlainTensor& out_q,
                                 bool rotate,
                                 bool qjl,
                                 const CpuParallelPtr& cpu_parallel,
                                 ov::element::Type q_precision) {
    auto B = q_input.size(0);
    auto H = q_input.size(1);
    auto L1 = q_input.size(2);
    auto S = q_input.size(3);

    const size_t q_dim = qjl ? 2 * S : S;
    out_q.resize<float>({B, H, L1, q_dim});
    const auto dim = static_cast<int>(S);
    const bool need_convert = q_precision != ov::element::f32;

    const bool need_qjl_proj = rotate && qjl;

    cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
        // Stack buffer for Q conversion when precision != f32 and rotation is needed.
        // Rotation reads q_src and writes dst, so we need a separate buffer.
        // When !rotate, we convert directly into dst.
        float q_buf[512];  // max dim supported
        for (size_t l = 0; l < L1; l++) {
            auto* dst = out_q.ptr<float>(b, h, l);
            const float* q_src = nullptr;
            if (need_convert) {
                // Convert to f32: into q_buf if rotating (rotation needs separate src/dst),
                // or directly into dst if not rotating (single pass).
                float* conv_dst = rotate ? q_buf : dst;
                if (q_precision == ov::element::bf16) {
                    const auto* src = q_input.ptr<ov::bfloat16>(b, h, l);
                    int d = 0;
                    for (; d + simd::f32::width - 1 < dim; d += simd::f32::width) {
                        store(simd::load<simd::f32>(src + d), conv_dst + d);
                    }
                    for (; d < dim; d++) {
                        conv_dst[d] = static_cast<float>(src[d]);
                    }
                } else {  // f16
                    const auto* src = q_input.ptr<ov::float16>(b, h, l);
                    int d = 0;
                    for (; d + simd::f32::width - 1 < dim; d += simd::f32::width) {
                        store(simd::load<simd::f32>(src + d), conv_dst + d);
                    }
                    for (; d < dim; d++) {
                        conv_dst[d] = static_cast<float>(src[d]);
                    }
                }
                q_src = conv_dst;
            } else {
                q_src = q_input.ptr<float>(b, h, l);
                if (!rotate) {
                    // f32 Q, no rotation: just copy.
                    std::memcpy(dst, q_src, static_cast<size_t>(dim) * sizeof(float));
                }
            }
            if (rotate) {
                turboq_rotate_forward(q_src, dst, dim);
                if (need_qjl_proj) {
                    // projected_q = QJL_project(q_rotated) — dst already holds rotated Q
                    turboq_qjl_project_forward(dst, dst + S, dim);
                }
            }
        }
    });
}

struct NoOpInit {
    void operator()(size_t /*unused*/) const {}
};

// Generic parallel loop over KV cache tokens.
// Covers both Q·K scoring and V accumulation — both traverse the same [B, Hk, kv_len]
// cache structure with identical parallelization and work splitting.
//
//   direct_fn(run_len, n_heads, head_dim, bits, b, h_group, start_pos, ithr)
//     Processes all tokens in a contiguous run. Computes pointers, handles prefetching,
//     and accesses data via captures.
//   per_thread_init(ithr)
//     Called once per thread before work (e.g. memset accumulators). Default: no-op.
template <typename DirectFn, typename InitFn = NoOpInit>
static void turboq_foreach_kv(size_t B,
                              size_t Hk,
                              size_t S,
                              size_t kv_len,
                              size_t h_each_group_len,
                              int nthr,
                              DirectFn&& direct_fn,
                              InitFn per_thread_init = {}) {
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        per_thread_init(ithr);

        size_t start{0}, end{0};
        splitter(B * Hk * kv_len, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }

        size_t pos = 0, b = 0, h_group = 0;
        parallel_it_init(start, h_group, Hk, b, B, pos, kv_len);

        size_t iwork = start;
        while (iwork < end) {
            size_t run_len = std::min(kv_len - pos, end - iwork);

            direct_fn(run_len, static_cast<int>(h_each_group_len), static_cast<int>(S), b, h_group, pos, ithr);

            for (size_t r = 0; r < run_len; r++) {
                parallel_it_step(h_group, Hk, b, B, pos, kv_len);
            }
            iwork += run_len;
        }
    });
}

static void turboq_softmax(const PlainTensor& attn_w,
                           const PlainTensor& alibi_mask,
                           const PlainTensor& attention_mask,
                           const PlainTensor& sink_input,
                           size_t B,
                           size_t H,
                           size_t q_len,
                           size_t kv_len,
                           bool auto_causal,
                           float d_scale,
                           const CpuParallelPtr& cpu_parallel) {
    auto precision = ov::element::f32;
    auto softmax_body = [&](size_t b, size_t h, size_t m) {
        auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
        float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, m, 0}, true) : nullptr;
        uint8_t* attn_mask_ptr = nullptr;
        auto attn_mask_prec = attention_mask.get_precision();
        if (attention_mask) {
            attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<float>({b, h, m, 0}, true));
        }
        float* sink = nullptr;
        if (sink_input) {
            sink = &sink_input.at<float>({b, h, m, 0}, true);
        }
        attn_softmax_kernel<float>(attn_w.ptr<float>(b, h, m),
                                   attn_w.ptr<float>(b, h, m),
                                   d_scale,
                                   alibi_ptr,
                                   attn_mask_ptr,
                                   nullptr,
                                   false,
                                   ncausal,
                                   kv_len,
                                   attn_mask_prec,
                                   precision,
                                   sink);
    };
    if (q_len == 1) {
        cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
            softmax_body(b, h, 0);
        });
    } else {
        cpu_parallel->parallel_for3d(B, H, q_len, softmax_body);
    }
}

template <typename T>
static void turboq_reduce_typed(const PlainTensor& attn_score,
                                PlainTensor& output_emb,
                                bool has_out_transpose,
                                bool v_rotation_fused,
                                bool qjl,
                                size_t B,
                                size_t H,
                                size_t q_len,
                                size_t S,
                                int nthr,
                                const CpuParallelPtr& cpu_parallel,
                                size_t q_offset) {
    const size_t thread_stride = attn_score.stride(0);

    auto reduce_body = [&](size_t b, size_t h, size_t m) {
        const size_t out_m = q_offset + m;
        auto* dst = has_out_transpose ? output_emb.ptr<T>(b, out_m, h * S) : output_emb.ptr<T>(b, h, out_m);
        auto* src0 = attn_score.ptr<float>(static_cast<size_t>(0), b, out_m, h);
        float* sign0 = qjl ? src0 + S : nullptr;
        turboq_reduce_head_impl(static_cast<int>(S),
                                dst,
                                src0,
                                nthr,
                                thread_stride,
                                !v_rotation_fused,
                                sign0,
                                thread_stride);
    };
    if (q_len == 1) {
        cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
            reduce_body(b, h, 0);
        });
    } else {
        cpu_parallel->parallel_for3d(B, H, q_len, reduce_body);
    }
}

static void turboq_reduce(const PlainTensor& attn_score,
                          PlainTensor& output_emb,
                          bool has_out_transpose,
                          bool v_rotation_fused,
                          bool qjl,
                          size_t B,
                          size_t H,
                          size_t q_len,
                          size_t S,
                          int nthr,
                          const CpuParallelPtr& cpu_parallel,
                          size_t q_offset = 0) {
    const auto out_prec = output_emb.get_precision();
    if (out_prec == ov::element::bf16) {
        turboq_reduce_typed<ov::bfloat16>(attn_score,
                                          output_emb,
                                          has_out_transpose,
                                          v_rotation_fused,
                                          qjl,
                                          B,
                                          H,
                                          q_len,
                                          S,
                                          nthr,
                                          cpu_parallel,
                                          q_offset);
    } else if (out_prec == ov::element::f16) {
        turboq_reduce_typed<ov::float16>(attn_score,
                                         output_emb,
                                         has_out_transpose,
                                         v_rotation_fused,
                                         qjl,
                                         B,
                                         H,
                                         q_len,
                                         S,
                                         nthr,
                                         cpu_parallel,
                                         q_offset);
    } else {
        turboq_reduce_typed<float>(attn_score,
                                   output_emb,
                                   has_out_transpose,
                                   v_rotation_fused,
                                   qjl,
                                   B,
                                   H,
                                   q_len,
                                   S,
                                   nthr,
                                   cpu_parallel,
                                   q_offset);
    }
}

// ---------------------------------------------------------------------------
// Record codecs and detection traits are in codec_record.hpp.
// ---------------------------------------------------------------------------

// Unified per-token QK dot product through record codec.
template <typename QT, typename Codec>
static inline float record_qk_dot(const uint8_t* k, const QT* q, int head_dim, Codec& codec) {
    if constexpr (is_qjl_v<Codec> && std::is_same_v<QT, float>) {
        return turboq_codec_qk_dot_qjl(k, q, codec.inner, codec_record_bytes(codec, head_dim), head_dim);
    } else if constexpr (is_polar_v<Codec> && std::is_same_v<QT, float>) {
        return polar_token_qk_dot(k, q, head_dim, codec.bpl);
    } else if constexpr (is_qjl_v<Codec> || is_polar_v<Codec>) {
        // Dead code: QJL is only used with encoded K where Q is always f32.
        (void)k;
        (void)q;
        (void)head_dim;
        (void)codec;
        return 0.0F;
    } else if constexpr (is_affine_v<Codec>) {
        // Deferred dequant: raw dot per group, then affine correction.
        // Saves per-element sub+mul by factoring out scale and zp.
        const int group_dim = codec.group_dim();
        const int num_groups = codec.n_groups(head_dim);
        float sum = 0.0F;
        for (int g = 0; g < num_groups; g++) {
            auto elem_codec = codec.group_codec(g);
            constexpr int bits_per_elem = decltype(elem_codec)::bits;
            const int group_elem_offset = g * group_dim;
            const size_t group_byte_offset = static_cast<size_t>(group_elem_offset) * bits_per_elem / 8;
            const uint8_t* group_k = k + group_byte_offset;
            const QT* group_q = q + group_elem_offset;
            float raw = codec_dot<QT>(group_k, group_q, group_dim, elem_codec);
            sum += codec.correct_dot(raw, g);
        }
        return sum;
    } else if constexpr (is_head_grouped_v<Codec>) {
        const int group_dim = codec.group_dim();
        const int num_groups = codec.n_groups(head_dim);
        float sum = 0.0F;
        for (int g = 0; g < num_groups; g++) {
            auto elem_codec = codec.group_codec(g);
            constexpr int bits_per_elem = decltype(elem_codec)::bits;
            const int group_elem_offset = g * group_dim;
            const size_t group_byte_offset = static_cast<size_t>(group_elem_offset) * bits_per_elem / 8;
            const uint8_t* group_k = k + group_byte_offset;
            const QT* group_q = q + group_elem_offset;
            sum += codec_dot<QT>(group_k, group_q, group_dim, elem_codec);
        }
        return sum;
    } else {
        return codec_dot<QT>(k, q, head_dim, codec.inner) * record_scale(codec, k, head_dim);
    }
}

// Unified per-token V weighted accumulation through record codec.
template <typename Codec>
static inline void record_v_accum(const uint8_t* v,
                                  StridedData<const float> weights,
                                  StridedData<float> accum,
                                  int n_heads,
                                  int head_dim,
                                  Codec& codec) {
    if constexpr (is_qjl_v<Codec>) {
        // Two passes: base codebook accum + sign accum (at accum + head_dim offset).
        // Record layout: [indices | signs | gamma(4) | norm(4)]
        const size_t record_bytes = codec_record_bytes(codec, head_dim);
        const size_t gamma_off = record_bytes - 8;
        const size_t sign_off = gamma_off - static_cast<size_t>(head_dim / 8);
        const float norm_scale = turboq_norm_scale(v, record_bytes, head_dim);
        const float gamma = read_as<float>(v + gamma_off);
        const float sign_scale = norm_scale * gamma;
        const uint8_t* signs_base = v + sign_off;
        // Pass 1: base codebook values → accum[0..head_dim)
        codec_weighted_accum(v, head_dim, codec.inner, norm_scale, weights, accum, n_heads);
        // Pass 2: sign bits → accum[head_dim..2*head_dim)  (sign correction buffer)
        StridedData<float> sign_accum{accum.data + head_dim, accum.stride};
        codec_weighted_accum(signs_base, head_dim, SignCodec{}, sign_scale, weights, sign_accum, n_heads);
    } else if constexpr (is_polar_v<Codec>) {
        polar_token_v_accum(v, weights, accum, n_heads, head_dim, codec.bpl);
    } else if constexpr (is_head_grouped_v<Codec>) {
        const int group_dim = codec.group_dim();
        const int num_groups = codec.n_groups(head_dim);
        for (int g = 0; g < num_groups; g++) {
            auto elem_codec = codec.group_codec(g);
            constexpr int bits_per_elem = decltype(elem_codec)::bits;
            const int group_elem_offset = g * group_dim;
            const size_t group_byte_offset = static_cast<size_t>(group_elem_offset) * bits_per_elem / 8;
            const uint8_t* group_v = v + group_byte_offset;
            StridedData<float> group_accum{accum.data + group_elem_offset, accum.stride};
            codec_weighted_accum(group_v, group_dim, elem_codec, 1.0F, weights, group_accum, n_heads);
        }
    } else {
        codec_weighted_accum(v, head_dim, codec.inner, record_scale(codec, v, head_dim), weights, accum, n_heads);
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Exported single-token wrappers — used by unit tests.
// Route through record_qk_dot / record_v_accum to exercise the same dispatch path.
// ---------------------------------------------------------------------------
float turboq_fused_qk_dot(const void* packed_k, const float* q_rotated, int head_dim, int bits) {
    const auto* k = static_cast<const uint8_t*>(packed_k);
    if (bits == 4) {
        RecordCodec<TurboQCodec4<>> rc{TurboQCodec4<>{head_dim}};
        return record_qk_dot(k, q_rotated, head_dim, rc);
    }
    if (bits == 3) {
        RecordCodec<TurboQCodec3<>> rc{TurboQCodec3<>{head_dim}};
        return record_qk_dot(k, q_rotated, head_dim, rc);
    }
    RecordCodec<TurboQCodec2<>> rc{TurboQCodec2<>{head_dim}};
    return record_qk_dot(k, q_rotated, head_dim, rc);
}

void turboq_fused_v_accum(const void* packed_v,
                          const float* weights,
                          float* const* accum_ptrs,
                          int num_heads,
                          int head_dim,
                          int bits) {
    const auto* v = static_cast<const uint8_t*>(packed_v);
    float* a_base = accum_ptrs[0];
    const size_t a_stride =
        (num_heads > 1) ? static_cast<size_t>(accum_ptrs[1] - accum_ptrs[0]) : static_cast<size_t>(head_dim);
    StridedData<const float> w{weights, 1};
    StridedData<float> a{a_base, a_stride};
    if (bits == 4) {
        RecordCodec<TurboQCodec4<>> rc{TurboQCodec4<>{head_dim}};
        record_v_accum(v, w, a, num_heads, head_dim, rc);
    } else if (bits == 3) {
        RecordCodec<TurboQCodec3<>> rc{TurboQCodec3<>{head_dim}};
        record_v_accum(v, w, a, num_heads, head_dim, rc);
    } else {
        RecordCodec<TurboQCodec2<>> rc{TurboQCodec2<>{head_dim}};
        record_v_accum(v, w, a, num_heads, head_dim, rc);
    }
}

float polarq_fused_qk_dot(const void* packed_k, const float* q, int head_dim, int bits) {
    const auto* k = static_cast<const uint8_t*>(packed_k);
    PolarRecordCodec rc{polar_get_bits_per_level(bits, head_dim)};
    return record_qk_dot(k, q, head_dim, rc);
}

void polarq_fused_v_accum(const void* packed_v,
                          const float* weights,
                          float* const* accum_ptrs,
                          int n_heads,
                          int head_dim,
                          int bits) {
    const auto* v = static_cast<const uint8_t*>(packed_v);
    float* a_base = accum_ptrs[0];
    size_t a_stride =
        (n_heads > 1) ? static_cast<size_t>(accum_ptrs[1] - accum_ptrs[0]) : static_cast<size_t>(head_dim);
    PolarRecordCodec rc{polar_get_bits_per_level(bits, head_dim)};
    record_v_accum(v,
                   StridedData<const float>{weights, 1},
                   StridedData<float>{a_base, a_stride},
                   n_heads,
                   head_dim,
                   rc);
}

// Unified per-token QK scoring loop.
// Handles beam table resolution, prefetching, and multi-head scoring.
// TokenScoreFn: (const uint8_t* k, int head, size_t t, size_t b_kv) -> float
// ---------------------------------------------------------------------------
template <typename TokenScoreFn>
static void score_tokens(const uint8_t* kv_base,
                         size_t stride_batch,
                         size_t stride_pos,
                         const int32_t* beam_tbl,
                         size_t b,
                         StridedData<float> scores,
                         size_t run_len,
                         int n_heads,
                         size_t pf_bytes,
                         TokenScoreFn score_fn) {
    for (size_t t = 0; t < run_len; t++) {
        const size_t b_kv = beam_tbl ? static_cast<size_t>(beam_tbl[t]) : b;
        const uint8_t* k_record = kv_base + b_kv * stride_batch + t * stride_pos;
        if (t + PREFETCH_AHEAD < run_len) {
            [[maybe_unused]] const size_t prefetch_t = t + PREFETCH_AHEAD;
            [[maybe_unused]] const size_t prefetch_b = beam_tbl ? static_cast<size_t>(beam_tbl[prefetch_t]) : b;
            [[maybe_unused]] const uint8_t* prefetch_record =
                kv_base + prefetch_b * stride_batch + prefetch_t * stride_pos;
            prefetch_bytes(pf_bytes, _MM_HINT_T0, 0, prefetch_record);
        }
        for (int g = 0; g < n_heads; g++) {
            scores[g][t] = score_fn(k_record, g, t, b_kv);
        }
    }
}

// ---------------------------------------------------------------------------
// Unified per-token V accumulation loop.
// TokenAccumFn: (const uint8_t* v, const float* w_base, size_t w_stride,
//                float* a_base, size_t a_stride, int n_heads, size_t t, size_t b_kv) -> void
// ---------------------------------------------------------------------------
template <typename TokenAccumFn>
static void accum_tokens(const uint8_t* kv_base,
                         size_t stride_batch,
                         size_t stride_pos,
                         const int32_t* beam_tbl,
                         size_t b,
                         StridedData<const float> weights,
                         StridedData<float> accum,
                         int n_heads,
                         size_t run_len,
                         size_t pf_bytes,
                         TokenAccumFn accum_fn) {
    for (size_t t = 0; t < run_len; t++) {
        const size_t b_kv = beam_tbl ? static_cast<size_t>(beam_tbl[t]) : b;
        const uint8_t* v_record = kv_base + b_kv * stride_batch + t * stride_pos;
        if (t + PREFETCH_AHEAD < run_len) {
            [[maybe_unused]] const size_t prefetch_t = t + PREFETCH_AHEAD;
            [[maybe_unused]] const size_t prefetch_b = beam_tbl ? static_cast<size_t>(beam_tbl[prefetch_t]) : b;
            [[maybe_unused]] const uint8_t* prefetch_record =
                kv_base + prefetch_b * stride_batch + prefetch_t * stride_pos;
            prefetch_bytes(pf_bytes, _MM_HINT_T0, 0, prefetch_record);
        }
        // Offset weights by t: weights[h][t] is the weight for head h at token t.
        StridedData<const float> weights_at_t{weights.data + t, weights.stride};
        accum_fn(v_record, weights_at_t, accum, n_heads, t, b_kv);
    }
}

// ---------------------------------------------------------------------------
// Phase 1 dispatch functions for QK scoring.
// ---------------------------------------------------------------------------

// Encoded K (TBQ/Polar): Q is pre-rotated f32 from prepared_q.
// When qjl=true, q layout is [rotated_q (S) | projected_q (S)] and K records have QJL layout.
// ---------------------------------------------------------------------------
// Q precision dispatch helper: resolves runtime Q element type to typed pointer,
// calls fn(typed_ptr, stride) with the concrete type.
// ---------------------------------------------------------------------------
template <typename Fn>
static void dispatch_q_precision(const PlainTensor& q_input,
                                 size_t b,
                                 size_t h_start,
                                 ov::element::Type q_precision,
                                 Fn&& fn,
                                 size_t q_idx = 0) {
    const size_t q_stride = q_input.stride(1);
    if (q_precision == ov::element::f16) {
        fn(StridedData<const ov::float16>{q_input.ptr<ov::float16>(b, h_start, q_idx), q_stride});
    } else if (q_precision == ov::element::bf16) {
        fn(StridedData<const ov::bfloat16>{q_input.ptr<ov::bfloat16>(b, h_start, q_idx), q_stride});
    } else {
        fn(StridedData<const float>{q_input.ptr<float>(b, h_start, q_idx), q_stride});
    }
}

// ---------------------------------------------------------------------------
// Total byte size of one cache record for `codec` at head_dim `head_dim`.
// Dispatches on codec shape via existing traits. Used for prefetch sizing.
// ---------------------------------------------------------------------------
template <typename Codec>
static inline size_t codec_record_bytes(const Codec& codec, int head_dim) {
    if constexpr (is_qjl_v<Codec>) {
        // indices + signs + gamma + norm
        return turboq_head_bytes_qjl(head_dim, Codec::inner_t::bits);
    } else if constexpr (is_polar_v<Codec>) {
        // Walk per-level bit widths from bpl + 4-byte norm.
        const int L = polar_tree_depth(head_dim);
        size_t total = 4;
        for (int k = 0; k < L; k++) {
            total += (static_cast<size_t>(head_dim >> (k + 1)) * codec.bpl[k] + 7) / 8;
        }
        return total;
    } else if constexpr (is_token_indexed_v<Codec>) {
        // Grouped / ByChannel / Affine — packed values only, no inline tail.
        return static_cast<size_t>(head_dim) * Codec::inner_t::bits / 8;
    } else if constexpr (is_raw_codec_v<typename Codec::inner_t>) {
        // RecordCodec<RawCodec<T>>: no trailing norm.
        return static_cast<size_t>(head_dim) * Codec::inner_t::bits / 8;
    } else {
        // RecordCodec<TurboQCodecN>: packed indices + trailing fp32 norm.
        return static_cast<size_t>(head_dim) * Codec::inner_t::bits / 8 + 4;
    }
}

// Per-record scale factor. TBQ reads norm from record tail; Raw is always 1.0;
// QJL computes its own scale inside turboq_codec_qk_dot_qjl.
template <typename Codec>
static inline float record_scale(const Codec& codec, const uint8_t* data, int head_dim) {
    (void)codec;
    if constexpr (is_raw_codec_v<typename Codec::inner_t>) {
        (void)data;
        (void)head_dim;
        return 1.0F;
    } else {
        return turboq_norm_scale(data, codec_record_bytes(codec, head_dim), head_dim);
    }
}

// ---------------------------------------------------------------------------
// Codec dispatch: maps runtime CacheCodec to concrete HeadCodec type.
// Covers TBQ, U8, U4, and Raw codecs. QJL and Polar are handled by the caller.
// ---------------------------------------------------------------------------
template <typename Fn>
static void dispatch_codec(CacheCodec codec,
                           int head_dim,
                           size_t group_size,
                           const PlainTensor& scale_zp,
                           Fn&& fn,
                           const float* q_group_sums = nullptr,
                           size_t q_group_sums_stride = 0) {
    switch (codec) {
    case CacheCodec::TBQ4:
        fn(RecordCodec<TurboQCodec4<>>{TurboQCodec4<>{head_dim}});
        break;
    case CacheCodec::TBQ3:
        fn(RecordCodec<TurboQCodec3<>>{TurboQCodec3<>{head_dim}});
        break;
    case CacheCodec::TBQ2:
        fn(RecordCodec<TurboQCodec2<>>{TurboQCodec2<>{head_dim}});
        break;
    case CacheCodec::U8:
        if (q_group_sums) {
            fn(AffineRecordCodec{scale_zp, group_size, q_group_sums, q_group_sums_stride});
        } else {
            fn(GroupedRecordCodec<U8Codec>{scale_zp,
                                           scale_zp ? scale_zp.stride_bytes(0) : 0,
                                           scale_zp ? scale_zp.stride_bytes(1) : 0,
                                           group_size});
        }
        break;
    case CacheCodec::U4:
        fn(GroupedRecordCodec<U4Codec>{scale_zp,
                                       scale_zp ? scale_zp.stride_bytes(0) : 0,
                                       scale_zp ? scale_zp.stride_bytes(1) : 0,
                                       group_size});
        break;
    case CacheCodec::U8_BY_CHANNEL:
        fn(ByChannelRecordCodec{scale_zp, scale_zp ? scale_zp.stride_bytes(1) : 0, group_size});
        break;
    case CacheCodec::RAW_F32:
        fn(RecordCodec<RawCodec<float>>{{}});
        break;
    case CacheCodec::RAW_F16:
        fn(RecordCodec<RawCodec<ov::float16>>{{}});
        break;
    case CacheCodec::RAW_BF16:
        fn(RecordCodec<RawCodec<ov::bfloat16>>{{}});
        break;
    case CacheCodec::POLAR4:
        fn(PolarRecordCodec{polar_get_bits_per_level(4, head_dim)});
        break;
    case CacheCodec::POLAR3:
        fn(PolarRecordCodec{polar_get_bits_per_level(3, head_dim)});
        break;
    case CacheCodec::TBQ4_QJL:
        fn(QJLRecordCodec<TurboQCodec3<>>{TurboQCodec3<>{head_dim}});
        break;
    case CacheCodec::TBQ3_QJL:
        fn(QJLRecordCodec<TurboQCodec2<>>{TurboQCodec2<>{head_dim}});
        break;
    }
}

// ---------------------------------------------------------------------------

void mha_turboq(PlainTensor& q_input,
                const PlainTensor& key_cache,
                const PlainTensor& packed_value,
                const PlainTensor& alibi_mask,
                const PlainTensor& attention_mask,
                const PlainTensor& beams,
                PlainTensor& output_emb,
                PlainTensor& buf_attn_w,
                PlainTensor& buf_attn_score,
                bool has_out_transpose,
                float d_scale,
                CacheCodec k_codec,
                CacheCodec v_codec,
                bool v_rotation_fused,
                bool auto_causal,
                const PlainTensor& sink_input,
                const CpuParallelPtr& cpu_parallel,
                const PlainTensor& k_scale_zp,
                size_t key_group_size,
                const PlainTensor& v_scale_zp,
                size_t value_group_size,
                ov::element::Type q_precision,
                size_t value_head_dim) {
    // ---------------------------------------------------------------------------
    // Setup: dimensions, scratch buffers, precomputed constants.
    // ---------------------------------------------------------------------------
    const bool k_qjl = k_codec == CacheCodec::TBQ4_QJL || k_codec == CacheCodec::TBQ3_QJL;
    const bool v_qjl = v_codec == CacheCodec::TBQ4_QJL || v_codec == CacheCodec::TBQ3_QJL;

    // OV_TURBOQ_SKIP_ROTATION: simulate weight fusion (8.1) — skip all rotations.
    // Results will be numerically wrong but timing shows the potential benefit.
    const bool skip_rotation = g_skip_rotation;
    if (skip_rotation) {
        v_rotation_fused = true;
    }
    const bool v_polar = v_codec == CacheCodec::POLAR4 || v_codec == CacheCodec::POLAR3;

    // QJL Q·K correction needs S*Q*q projection — required when K uses QJL.
    const bool need_qjl_projection = k_qjl;

    // Prepare f32 Q only when K has codec — rotation needed for QK dot in rotated domain.
    // When skip_rotation is set, still convert Q to f32 but skip the rotation itself.
    PlainTensor prepared_q;
    if (is_encoded(k_codec)) {
        turboq_prepare_query(q_input,
                             prepared_q,
                             !skip_rotation,
                             need_qjl_projection && !skip_rotation,
                             cpu_parallel,
                             q_precision);
    }

    const auto B = q_input.size(0);
    const auto H = q_input.size(1);
    const auto q_len = q_input.size(2);
    const auto S = q_input.size(3);  // K head dimension (= Q head dimension)
    const auto SV = value_head_dim;  // V head dimension (may differ from S)
    const auto Hk = key_cache.size(1);
    const auto kv_len = key_cache.size(2);
    const size_t h_each_group_len = H / Hk;
    const auto nthr = parallel_get_max_threads();

    if (d_scale == 0.0F) {
        d_scale = 1.0F / std::sqrt(static_cast<float>(S));
    }

    buf_attn_w.resize<float>({B, H, q_len, (kv_len + 15) / 16 * 16});
    // When V uses QJL, pack [base_accum | sign_accum] contiguously per head (2*SV floats).
    const size_t accum_dim = (v_qjl && !v_polar) ? 2 * SV : SV;
    buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, accum_dim});

    // Precompute per-group Q sums for affine u8 deferred dequant optimization.
    // q_group_sums[b, h, m, group] = sum(q[b, h, m, group*gs .. (group+1)*gs]).
    // Used to defer zp subtraction from per-element to per-group-after-dot.
    // AVX2-only: defer u8 zp subtraction from per-element to per-group-after-dot.
    // AVX-512 has enough throughput to absorb the per-element sub — not worth the overhead.
    constexpr bool is_avx2 = simd::f32::isa_value == simd::isa::avx2;
    const bool use_affine_k = is_avx2 && (k_codec == CacheCodec::U8);
    const size_t n_key_groups = use_affine_k ? S / key_group_size : 0;
    PlainTensor q_group_sums_buf;
    if (use_affine_k) {
        q_group_sums_buf.resize<float>({B, H, q_len, n_key_groups});
        cpu_parallel->parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t m_idx) {
            auto* sums = q_group_sums_buf.ptr<float>(b, h, m_idx);
            dispatch_q_precision(
                q_input,
                b,
                h,
                q_precision,
                [&](auto q) {
                    using QT = std::remove_const_t<std::remove_pointer_t<decltype(q.data)>>;
                    const QT* q_head = q[0];  // single head at (b, h)
                    for (size_t g = 0; g < n_key_groups; g++) {
                        constexpr int W = simd::f32::width;
                        simd::f32 acc;
                        const size_t offset = g * key_group_size;
                        size_t i = 0;
                        for (; i + W <= key_group_size; i += W) {
                            acc = acc + simd::load<simd::f32>(q_head + offset + i);
                        }
                        sums[g] = reduce(acc);
                        for (; i < key_group_size; i++) {
                            sums[g] += static_cast<float>(q_head[offset + i]);
                        }
                    }
                },
                m_idx);
        });
    }

    // For each query position m, phases 1-4 run independently. When q_len=1
    // (single-token decode) these loops execute once. When q_len>1 (fuse_concat
    // prompt), each position gets its own scores, softmax, and accumulation.

    // ---------------------------------------------------------------------------
    // Phase 1: Q·K scores for all query positions.
    // ---------------------------------------------------------------------------
    for (size_t m = 0; m < q_len; m++) {
        turboq_foreach_kv(
            B,
            Hk,
            S,
            kv_len,
            h_each_group_len,
            nthr,
            [&, k_codec, m](size_t run_len,
                            int n_heads,
                            int head_dim,
                            size_t b,
                            size_t h_group,
                            size_t start_pos,
                            size_t /*ithr*/) {
                const size_t h_start = h_group * h_each_group_len;
                const auto* kv_base = static_cast<const uint8_t*>(key_cache.ptr_v(size_t{0}, h_group, start_pos));
                const size_t stride_batch = key_cache.stride_bytes(0);
                const size_t stride_pos = key_cache.stride_bytes(2);
                const bool use_beams = beams && B > 1;
                const int32_t* beam_tbl_ptr = use_beams ? beams.ptr<int32_t>(b) + start_pos : nullptr;
                float* scores_row_base = buf_attn_w.ptr<float>(b, h_start, m) + start_pos;
                StridedData<float> scores{scores_row_base, buf_attn_w.stride(1)};

                auto make_score = [start_pos, h_group, head_dim](auto q, auto codec) {
                    return [=](const uint8_t* k, int g, size_t t, size_t b_kv) mutable {
                        using QT = std::remove_const_t<std::remove_pointer_t<decltype(q.data)>>;
                        using C = decltype(codec);
                        const QT* q_head = q[g];
                        if constexpr (has_for_head_v<C>) {
                            auto hc = codec.for_head(g);
                            auto tc = hc.for_token(start_pos, h_group, t, b_kv);
                            return record_qk_dot<QT>(k, q_head, head_dim, tc);
                        } else if constexpr (is_token_indexed_v<C>) {
                            auto tc = codec.for_token(start_pos, h_group, t, b_kv);
                            return record_qk_dot<QT>(k, q_head, head_dim, tc);
                        } else {
                            return record_qk_dot<QT>(k, q_head, head_dim, codec);
                        }
                    };
                };

                const bool encoded = is_encoded(k_codec);
                const auto& q_src = encoded ? prepared_q : q_input;
                const auto q_prec = encoded ? ov::element::f32 : q_precision;
                // q_group_sums base for first head in group; stride to step between heads.
                const float* q_group_sums = use_affine_k ? q_group_sums_buf.ptr<float>(b, h_start, m) : nullptr;
                const size_t q_group_sums_stride = use_affine_k ? q_group_sums_buf.stride(1) : 0;
                dispatch_q_precision(
                    q_src,
                    b,
                    h_start,
                    q_prec,
                    [&](auto q) {
                        dispatch_codec(
                            k_codec,
                            head_dim,
                            key_group_size,
                            k_scale_zp,
                            [&](auto head) {
                                score_tokens(kv_base,
                                             stride_batch,
                                             stride_pos,
                                             beam_tbl_ptr,
                                             b,
                                             scores,
                                             run_len,
                                             n_heads,
                                             codec_record_bytes(head, head_dim),
                                             make_score(q, head));
                            },
                            q_group_sums,
                            q_group_sums_stride);
                    },
                    m);
            });
    }

    // ---------------------------------------------------------------------------
    // Phase 2: Softmax — runs over all query positions at once (parallel over B, H, q_len).
    // ---------------------------------------------------------------------------
    turboq_softmax(buf_attn_w,
                   alibi_mask,
                   attention_mask,
                   sink_input,
                   B,
                   H,
                   q_len,
                   kv_len,
                   auto_causal,
                   d_scale,
                   cpu_parallel);

    // ---------------------------------------------------------------------------
    // Phases 3+4: V accumulation + reduce, per query position.
    // Each position m: zero thread accumulators → accumulate V weighted by
    // softmax scores → reduce across threads + inverse rotation → write output.
    // ---------------------------------------------------------------------------
    const bool do_inv_rotate = !v_rotation_fused && is_encoded(v_codec);
    const bool do_qjl_reduce = v_qjl && !v_polar;
    for (size_t m = 0; m < q_len; m++) {
        // Phase 3: V accumulation for query position m.
        turboq_foreach_kv(
            B,
            Hk,
            SV,
            kv_len,
            h_each_group_len,
            nthr,
            [&,
             v_codec,
             m](size_t run_len, int n_heads, int head_dim, size_t b, size_t h_group, size_t start_pos, size_t ithr) {
                const size_t h_start = h_group * h_each_group_len;
                const auto* kv_base = static_cast<const uint8_t*>(packed_value.ptr_v(size_t{0}, h_group, start_pos));
                const size_t stride_batch = packed_value.stride_bytes(0);
                const size_t stride_pos = packed_value.stride_bytes(2);
                const bool use_beams = beams && B > 1;
                const int32_t* beam_tbl_ptr = use_beams ? beams.ptr<int32_t>(b) + start_pos : nullptr;
                const float* weights_row_base = buf_attn_w.ptr<float>(b, h_start, m) + start_pos;
                StridedData<const float> weights{weights_row_base, buf_attn_w.stride(1)};
                float* accum_row_base = buf_attn_score.ptr<float>(ithr, b, m, h_start);
                StridedData<float> accum{accum_row_base, buf_attn_score.stride(3)};

                auto make_v_accum = [start_pos, h_group, head_dim](auto codec) {
                    return [=](const uint8_t* v,
                               StridedData<const float> w,
                               StridedData<float> a,
                               int n_heads,
                               size_t t,
                               size_t b_kv) mutable {
                        using C = decltype(codec);
                        if constexpr (is_token_indexed_v<C>) {
                            auto tc = codec.for_token(start_pos, h_group, t, b_kv);
                            record_v_accum(v, w, a, n_heads, head_dim, tc);
                        } else {
                            record_v_accum(v, w, a, n_heads, head_dim, codec);
                        }
                    };
                };

                dispatch_codec(v_codec, head_dim, value_group_size, v_scale_zp, [&](auto codec) {
                    accum_tokens(kv_base,
                                 stride_batch,
                                 stride_pos,
                                 beam_tbl_ptr,
                                 b,
                                 weights,
                                 accum,
                                 n_heads,
                                 run_len,
                                 codec_record_bytes(codec, head_dim),
                                 make_v_accum(codec));
                });
            },
            [&](size_t ithr) {
                std::memset(buf_attn_score.ptr<float>(ithr, 0, 0, 0, 0), 0, buf_attn_score.stride(0) * sizeof(float));
            });

        // Phase 4: Reduce for query position m.
        turboq_reduce(buf_attn_score,
                      output_emb,
                      has_out_transpose,
                      !do_inv_rotate,
                      do_qjl_reduce,
                      B,
                      H,
                      1,  // reduce one query position at a time
                      SV,
                      nthr,
                      cpu_parallel,
                      m);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
