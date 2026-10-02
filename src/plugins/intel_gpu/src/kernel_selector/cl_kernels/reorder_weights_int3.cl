// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/int3_utils.cl"

// Repack u3 weights from a plain oiyx bit stream into os_is_yx_osv16_isv32.
//
// The destination groups the filter into granules of one output channel by 32
// input channels. 32 u3 values are 96 bits, so a granule is exactly three uints
// and a (16 output x 32 input) block is exactly 192 bytes with no slack.
//
// Within a block the three words of each of the 16 output channels are stored as
// three planes of 16 uints rather than contiguously:
//
//   uint index = (n_block * CHUNKS_K + k_chunk) * 48 + w * 16 + lane
//
// so the consuming GEMM issues three subgroup block reads and lane `lane` ends up
// holding all 32 weights of its own output channel, ready for the matrix engine.
//
// One work item owns one granule, so every destination word has exactly one writer.

KERNEL(reorder_weights_int3)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const uint o = (uint)get_global_id(0);
    const uint k_chunk = (uint)get_global_id(1);

    const uint lane = o % 16;
    const uint n_block = o / 16;

    uint w0 = 0, w1 = 0, w2 = 0;

    unroll_for (uint kk = 0; kk < 32; ++kk) {
        const uint k = k_chunk * 32 + kk;

        // Blocks on the OFM/IFM edges are only partly covered by real weights; the
        // remainder stays zero so the GEMM can read full granules unconditionally.
        uint v = 0;
        if (o < OUTPUT_OFM_NUM && k < OUTPUT_IFM_NUM)
            v = UNPACK_UINT3_AT(input, GET_FILTER_INDEX(INPUT0, 0, o, k, 0, 0));

        const uint bit = kk * UINT3_BITS;
        const uint word = bit >> 5;
        const uint off = bit & 31;

        if (word == 0) w0 |= v << off;
        else if (word == 1) w1 |= v << off;
        else w2 |= v << off;

        // kk == 10 and kk == 21 straddle a word boundary; the rest fold away.
        if (off > 32 - UINT3_BITS) {
            if (word == 0) w1 |= v >> (32 - off);
            else w2 |= v >> (32 - off);
        }
    }

    __global uint* dst = (__global uint*)output;
    const uint base = (n_block * CHUNKS_K + k_chunk) * 48 + lane;
    dst[base] = w0;
    dst[base + 16] = w1;
    dst[base + 32] = w2;
}
