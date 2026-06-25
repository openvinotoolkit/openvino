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

#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH {

inline constexpr int OSCAR_R = 128;          // tokens per residual block
inline constexpr int OSCAR_G = 32;           // tokens sharing one (delta, zp) entry
inline constexpr int OSCAR_BITS = 2;
inline constexpr int OSCAR_LEVELS = 4;       // 1 << OSCAR_BITS
inline constexpr int OSCAR_SUBGROUPS = OSCAR_R / OSCAR_G;  // = 4

// Total bytes per block, per (batch, head). `with_norms` adds R*sizeof(fp16) for K side.
inline size_t oscar_block_bytes(int head_dim, bool with_norms) {
    assert(head_dim > 0 && (head_dim & (head_dim - 1)) == 0 && "head_dim must be power of 2");
    const size_t payload = static_cast<size_t>(OSCAR_R) * head_dim / 4;       // 2 bits per value
    const size_t params = static_cast<size_t>(OSCAR_SUBGROUPS) * head_dim * 2 * sizeof(ov::float16);
    const size_t norms = with_norms ? static_cast<size_t>(OSCAR_R) * sizeof(ov::float16) : 0;
    return payload + params + norms;
}

// Encode one R-token block for one (batch, head).
//   unit_vectors : [R][head_dim] f32, post-Hadamard unit-normalized rows
//   norms        : [R]           f32, per-token L2 norm of the pre-norm vector
//                  (may be nullptr for V side)
//   payload      : output, oscar_block_bytes - param/norm bytes — token-major INT2
//   deltas       : output, [SUBGROUPS][head_dim] fp16
//   zps          : output, [SUBGROUPS][head_dim] fp16
//   norms_q      : output, [R] fp16 (or nullptr if norms == nullptr)
inline void oscar_encode_block(const float* unit_vectors,
                               const float* norms,
                               int head_dim,
                               uint8_t* payload,
                               ov::float16* deltas,
                               ov::float16* zps,
                               ov::float16* norms_q) {
    assert(unit_vectors != nullptr);
    assert(payload != nullptr && deltas != nullptr && zps != nullptr);
    assert(head_dim > 0 && (head_dim & (head_dim - 1)) == 0);

    const int row_bytes = head_dim / 4;  // payload bytes per token

    for (int g = 0; g < OSCAR_SUBGROUPS; ++g) {
        // Per-channel (delta, zp) over the G tokens of this sub-group.
        for (int j = 0; j < head_dim; ++j) {
            float vmin = std::numeric_limits<float>::infinity();
            float vmax = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < OSCAR_G; ++t) {
                const float v = unit_vectors[(g * OSCAR_G + t) * head_dim + j];
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            const float range = vmax - vmin;
            // INT2 levels: 0..3 → represent vmin..vmax. Δ = range / (levels-1).
            const float delta = (range > 0.0F) ? (range / static_cast<float>(OSCAR_LEVELS - 1)) : 1.0F;
            deltas[g * head_dim + j] = ov::float16(delta);
            zps[g * head_dim + j] = ov::float16(vmin);
        }

        // Pack codes token-major. j%4 → bit position (0,2,4,6) inside one byte.
        for (int t = 0; t < OSCAR_G; ++t) {
            uint8_t* row = payload + (g * OSCAR_G + t) * row_bytes;
            std::memset(row, 0, row_bytes);
            for (int j = 0; j < head_dim; ++j) {
                const float v = unit_vectors[(g * OSCAR_G + t) * head_dim + j];
                const float delta = static_cast<float>(deltas[g * head_dim + j]);
                const float zp = static_cast<float>(zps[g * head_dim + j]);
                const float q_f = (v - zp) / delta;
                int q = static_cast<int>(q_f + 0.5F);
                q = std::max(0, std::min(OSCAR_LEVELS - 1, q));
                row[j >> 2] |= static_cast<uint8_t>((q & 0x3) << ((j & 0x3) * 2));
            }
        }
    }

    if (norms != nullptr && norms_q != nullptr) {
        for (int t = 0; t < OSCAR_R; ++t) {
            norms_q[t] = ov::float16(norms[t]);
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
