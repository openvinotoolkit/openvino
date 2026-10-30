// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// OScaR codec — INT2 packed cache with per-channel (delta, zp) shared across
// G=32 tokens inside a residual block of R=128 tokens.
//
// Inputs to the encoder are post-Hadamard unit vectors (H(k)/||H(k)||) and the
// per-token norms ||H(k)||. The encoder is responsible only for the INT2
// uniform-asym quantization and packing — Hadamard + normalize happen upstream
// (turboq_wht_inplace + turboq_norm_signflip in scaled_attn).
//
// Block layout (single (batch, head)):
//   payload       : R * head_dim * 2 bits = R * head_dim / 4 bytes
//                   token-major within sub-group; head_dim/4 bytes per token
//   deltas, zps   : (R/G) * head_dim * fp16  (shared across G tokens)
//   norms_q       : R * fp16  (K side only — pass nullptr for V)
//
// Phase 2 scope: scalar reference. SIMD pass comes later.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "nodes/kernels/simd/simd.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH {

inline constexpr int OSCAR_R = 128;          // tokens per residual block
inline constexpr int OSCAR_G = 32;           // tokens sharing one (delta, zp) entry
inline constexpr int OSCAR_BITS = 2;
inline constexpr int OSCAR_LEVELS = 4;       // 1 << OSCAR_BITS
inline constexpr int OSCAR_SUBGROUPS = OSCAR_R / OSCAR_G;  // = 4

// Per-token in-cache bytes: payload row (head_dim/4) + optional norm (fp16).
// Shared (delta, zp) params live in a sidecar sized by oscar_params_per_block.
inline size_t oscar_per_token_bytes(int head_dim, bool with_norms) {
    assert(head_dim > 0 && (head_dim & (head_dim - 1)) == 0 && "head_dim must be power of 2");
    const size_t payload = static_cast<size_t>(head_dim) / 4;
    const size_t norm = with_norms ? sizeof(ov::float16) : 0;
    return payload + norm;
}

// fp16 count per block in the params sidecar: [SUBGROUPS][head_dim] deltas + [SUBGROUPS][head_dim] zps.
inline size_t oscar_params_per_block(int head_dim) {
    return static_cast<size_t>(OSCAR_SUBGROUPS) * static_cast<size_t>(head_dim) * 2;
}

// Encode one R-token block for one (batch, head).
// Payload rows and per-token norms live in cache slots that are strided by
// `slot_stride_bytes` (LBHS: B*H*S_cache). Params sit contiguous in the sidecar.
//   unit_vectors     : [R][head_dim] f32, post-Hadamard unit-normalized rows
//   norms            : [R]           f32, per-token L2 norm (nullptr for V side)
//   head_dim         : inner dim
//   payload_slot0    : cache slot pointer for token 0 of this block; row payload
//                      lives at slot[0 .. head_dim/4)
//   slot_stride_bytes: bytes between consecutive token slots (per-token S_cache)
//   deltas           : output sidecar, [SUBGROUPS][head_dim] fp16
//   zps              : output sidecar, [SUBGROUPS][head_dim] fp16
//   norms_q_slot0    : cache slot pointer for token 0's norm (nullptr if no norms).
//                      Written at slot[head_dim/4] as fp16.
inline void oscar_encode_block(const float* unit_vectors,
                               const float* norms,
                               int head_dim,
                               uint8_t* payload_slot0,
                               size_t slot_stride_bytes,
                               ov::float16* deltas,
                               ov::float16* zps,
                               ov::float16* norms_q_slot0) {
    assert(unit_vectors != nullptr);
    assert(payload_slot0 != nullptr && deltas != nullptr && zps != nullptr);
    assert(head_dim > 0 && (head_dim & (head_dim - 1)) == 0);

    const int row_bytes = head_dim / 4;

    // Scratch: per-channel min/max over G tokens, and per-channel (delta, zp) in f32.
    // OSCAR_G=32 tokens per subgroup, head_dim assumed <= 256.
    alignas(64) float vmin_buf[256];
    alignas(64) float vmax_buf[256];
    alignas(64) float delta_f32[256];
    alignas(64) float zp_f32[256];
    assert(head_dim <= 256);

    for (int g = 0; g < OSCAR_SUBGROUPS; ++g) {
        const float* base = unit_vectors + g * OSCAR_G * head_dim;

        // Init with token 0, then reduce t=1..G-1. simd_loop over head_dim.
        simd::simd_loop(head_dim, [&](int j, auto a) {
            constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
            using V = simd::f32_t<Ia>;
            auto x0 = simd::load<V>(base + j, a);
            simd::store(x0, vmin_buf + j, a);
            simd::store(x0, vmax_buf + j, a);
        });
        for (int t = 1; t < OSCAR_G; ++t) {
            const float* row_f = base + t * head_dim;
            simd::simd_loop(head_dim, [&](int j, auto a) {
                constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
                using V = simd::f32_t<Ia>;
                auto x = simd::load<V>(row_f + j, a);
                simd::store(simd::min(simd::load<V>(vmin_buf + j, a), x), vmin_buf + j, a);
                simd::store(simd::max(simd::load<V>(vmax_buf + j, a), x), vmax_buf + j, a);
            });
        }

        // delta = (vmax - vmin) / (L-1) with 1.0 fallback when range == 0.
        for (int j = 0; j < head_dim; ++j) {
            const float range = vmax_buf[j] - vmin_buf[j];
            const float delta = (range > 0.0F) ? (range / static_cast<float>(OSCAR_LEVELS - 1)) : 1.0F;
            delta_f32[j] = delta;
            zp_f32[j] = vmin_buf[j];
            deltas[g * head_dim + j] = ov::float16(delta);
            zps[g * head_dim + j] = ov::float16(vmin_buf[j]);
        }

        // Quantize + pack per token. inv_delta once per (g,j) to avoid per-token div.
        alignas(64) float inv_delta[256];
        for (int j = 0; j < head_dim; ++j) inv_delta[j] = 1.0F / delta_f32[j];

        for (int t = 0; t < OSCAR_G; ++t) {
            const int token_idx = g * OSCAR_G + t;
            uint8_t* row = payload_slot0 + static_cast<size_t>(token_idx) * slot_stride_bytes;
            const float* src = unit_vectors + token_idx * head_dim;
            std::memset(row, 0, row_bytes);
            // Quantize scalar for now — pack requires bit-level shuffling not in SIMD.
            for (int j = 0; j < head_dim; ++j) {
                const float q_f = (src[j] - zp_f32[j]) * inv_delta[j];
                int q = static_cast<int>(q_f + 0.5F);
                q = std::max(0, std::min(OSCAR_LEVELS - 1, q));
                row[j >> 2] |= static_cast<uint8_t>((q & 0x3) << ((j & 0x3) * 2));
            }
        }
    }

    if (norms != nullptr && norms_q_slot0 != nullptr) {
        for (int t = 0; t < OSCAR_R; ++t) {
            auto* dst = reinterpret_cast<ov::float16*>(
                reinterpret_cast<uint8_t*>(norms_q_slot0) + static_cast<size_t>(t) * slot_stride_bytes);
            *dst = ov::float16(norms[t]);
        }
    }
}

// Dequantize one G-token sub-group into a row-major f32 buffer [G][head_dim].
//   payload      : token-major payload bytes for the parent block
//   deltas       : [SUBGROUPS][head_dim] fp16 (full block params)
//   zps          : [SUBGROUPS][head_dim] fp16
//   sub_g        : sub-group index in [0, SUBGROUPS)
//   out          : [G][head_dim] f32 dequantized unit-vector recon
inline void oscar_decode_subgroup(const uint8_t* payload,
                                  const ov::float16* deltas,
                                  const ov::float16* zps,
                                  int head_dim,
                                  int sub_g,
                                  float* out) {
    assert(sub_g >= 0 && sub_g < OSCAR_SUBGROUPS);
    const int row_bytes = head_dim / 4;
    for (int t = 0; t < OSCAR_G; ++t) {
        const uint8_t* row = payload + (sub_g * OSCAR_G + t) * row_bytes;
        float* dst = out + t * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            const int code = (row[j >> 2] >> ((j & 0x3) * 2)) & 0x3;
            const float delta = static_cast<float>(deltas[sub_g * head_dim + j]);
            const float zp = static_cast<float>(zps[sub_g * head_dim + j]);
            dst[j] = static_cast<float>(code) * delta + zp;
        }
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
