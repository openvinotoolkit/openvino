// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// OScaR write-path staging: append L1 freshly-rotated unit-vector tokens to a
// fp16 residual buffer. Once the residual reaches R tokens, encode one packed
// block via oscar_encode_block and advance the packed-cache cursor.
//
// Caller responsibilities:
//  - Hadamard rotation and L2 normalization upstream — inputs to this function
//    are already H(k)/||H(k)|| in f32 plus the per-token norms ||H(k)||.
//  - Pre-allocate the residual buffers, packed cache, and a per-(B,H) f32
//    scratch large enough for one R-token block ([R][head_dim] f32).
//  - Track residual_count per batch entry across calls — this function
//    operates on a single (b, h) slot.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "oscar_quantize.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH {

struct OscarStageResult {
    size_t flushed_blocks;       // full R-token blocks emitted this call
    size_t new_residual_count;   // post-call residue, in [0, R)
};

// Stage L1 tokens for a single (batch, head) slot.
//   unit_vectors  : [L1][head_dim] f32 — post-Hadamard unit vectors
//   norms         : [L1]           f32 — pre-norm token magnitudes
//                   (may be nullptr for V side; pass nullptr also for residual_norms)
//   residual_unit : [R][head_dim] fp16  — persistent residual buffer
//   residual_norms: [R]           fp16 (or nullptr matching norms)
//   residual_count: in,   prior residue (0..R-1)
//   packed_cursor : output cursor, advanced by flushed_blocks * block_stride
//   block_stride  : bytes per block (= oscar_block_bytes(head_dim, with_norms))
//   block_scratch_f32 : [R][head_dim] f32 working buffer used at flush time
inline OscarStageResult oscar_stage_and_flush(const float* unit_vectors,
                                              const float* norms,
                                              size_t L1,
                                              int head_dim,
                                              ov::float16* residual_unit,
                                              ov::float16* residual_norms,
                                              size_t residual_count,
                                              uint8_t* packed_cursor,
                                              size_t block_stride,
                                              float* block_scratch_f32) {
    OscarStageResult res{};
    const bool with_norms = (norms != nullptr) && (residual_norms != nullptr);
    size_t consumed = 0;

    while (consumed < L1) {
        const size_t space = static_cast<size_t>(OSCAR_R) - residual_count;
        const size_t take = (L1 - consumed < space) ? (L1 - consumed) : space;

        // Append `take` tokens to residual.
        for (size_t t = 0; t < take; ++t) {
            const float* src = unit_vectors + (consumed + t) * head_dim;
            ov::float16* dst = residual_unit + (residual_count + t) * head_dim;
            for (int j = 0; j < head_dim; ++j) {
                dst[j] = ov::float16(src[j]);
            }
            if (with_norms) {
                residual_norms[residual_count + t] = ov::float16(norms[consumed + t]);
            }
        }
        residual_count += take;
        consumed += take;

        // If full, flush one packed block.
        if (residual_count == static_cast<size_t>(OSCAR_R)) {
            // Materialize residual fp16 → f32 scratch.
            for (int t = 0; t < OSCAR_R; ++t) {
                const ov::float16* src = residual_unit + t * head_dim;
                float* dst = block_scratch_f32 + t * head_dim;
                for (int j = 0; j < head_dim; ++j) {
                    dst[j] = static_cast<float>(src[j]);
                }
            }
            float norms_scratch[OSCAR_R];
            if (with_norms) {
                for (int t = 0; t < OSCAR_R; ++t) {
                    norms_scratch[t] = static_cast<float>(residual_norms[t]);
                }
            }

            // Block layout: payload | deltas | zps | (norms_q if K side).
            const size_t payload_bytes = static_cast<size_t>(OSCAR_R) * head_dim / 4;
            const size_t param_bytes = static_cast<size_t>(OSCAR_SUBGROUPS) * head_dim * sizeof(ov::float16);
            uint8_t* payload = packed_cursor;
            auto* deltas = reinterpret_cast<ov::float16*>(packed_cursor + payload_bytes);
            auto* zps = reinterpret_cast<ov::float16*>(packed_cursor + payload_bytes + param_bytes);
            ov::float16* norms_q = with_norms
                ? reinterpret_cast<ov::float16*>(packed_cursor + payload_bytes + 2 * param_bytes)
                : nullptr;

            oscar_encode_block(block_scratch_f32, with_norms ? norms_scratch : nullptr,
                               head_dim, payload, deltas, zps, norms_q);

            packed_cursor += block_stride;
            ++res.flushed_blocks;
            residual_count = 0;
        }
    }

    res.new_residual_count = residual_count;
    return res;
}

}  // namespace ov::Extensions::Cpu::XARCH
