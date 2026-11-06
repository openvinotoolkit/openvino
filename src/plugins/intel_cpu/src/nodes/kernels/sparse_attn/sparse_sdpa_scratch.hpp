// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>

#include "utils/general_utils.h"

namespace ov::Extensions::Cpu {

// Per-thread scratch of the decode path, for the largest head pack of one item:
//   q[pack, Dh] | acc[pack, Dv] | run_max[pack] | run_sum[pack] | scores[pack, score_stride] | dequantized rows
// score_stride is max_run_rows, the widest run of adjacent blocks; the last region holds a quantized cache's K, then V
// rows.
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
    SdpaDecodeScratch(size_t Dh, size_t Dv, size_t max_pack, size_t Bk, bool quantized)
        : max_run_rows(Bk * std::max<size_t>(kMaxRunRows / Bk, 1)),
          score_stride(max_run_rows) {
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
        dequant = take(quantized ? score_stride * (Dh + Dv) : 0);
        total = offset;
    }
};

}  // namespace ov::Extensions::Cpu
