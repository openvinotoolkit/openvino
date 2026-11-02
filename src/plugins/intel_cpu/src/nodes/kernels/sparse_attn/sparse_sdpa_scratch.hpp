// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>

#include "utils/general_utils.h"

namespace ov::Extensions::Cpu {

// Per-thread f32 scratch of the decode path of sparse_sdpa_compute() (everything but the brgemm prefill), laid out for
// the largest pack of query heads (max_q_head_pack_size()) that one work item processes together:
//   q[pack, Dh] | acc[pack, Dv] | run_max[pack] | run_sum[pack] | scores[pack, score_stride] | dequantized rows
// score_stride is the widest run of key rows one call handles: max_run_rows rows of adjacent selected blocks, or the
// 4 merge units of an AMX chunk (amx_unit_blocks > 0). A quantized cache is dequantized into the last region
// (rows of K, then of V, f32 -- or 16-bit for the AMX chunks, which needs less); the AMX chunks of blocks that are not
// adjacent are gathered into it as well, with any cache.
struct SdpaDecodeScratch {
    static constexpr size_t kMaxRunRows = 128;  // rows of adjacent selected blocks attended by one run (>= Bk)

    size_t max_run_rows = 0;
    size_t score_stride = 0;
    size_t q = 0;
    size_t acc = 0;
    size_t run_max = 0;
    size_t run_sum = 0;
    size_t scores = 0;
    size_t dequant = 0;
    size_t total = 0;  // floats per thread, a multiple of 16

    SdpaDecodeScratch() = default;
    SdpaDecodeScratch(size_t Dh, size_t Dv, size_t max_pack, size_t Bk, size_t amx_unit_blocks, bool quantized) {
        max_run_rows = Bk * std::max<size_t>(kMaxRunRows / Bk, 1);
        score_stride = std::max(max_run_rows, 4 * amx_unit_blocks * Bk);
        size_t offset = 0;
        auto take = [&offset](size_t floats) {
            const size_t start = offset;
            offset += ov::intel_cpu::rnd_up(floats, size_t{16});
            return start;
        };
        q = take(max_pack * Dh);
        acc = take(max_pack * Dv);
        run_max = take(max_pack);
        run_sum = take(max_pack);
        scores = take(max_pack * score_stride);
        // also the gathered rows of an AMX chunk of blocks that are not adjacent (16-bit, which needs less than this)
        dequant = take((quantized || amx_unit_blocks > 0) ? score_stride * (Dh + Dv) : 0);
        total = offset;
    }
};

}  // namespace ov::Extensions::Cpu
