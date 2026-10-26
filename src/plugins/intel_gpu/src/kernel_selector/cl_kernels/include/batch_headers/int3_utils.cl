//  Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ================================================================================================
// U3 (3-bit unsigned) support
//
// Unlike u2/u4, three bits do not divide a byte evenly, so u3 has no fixed
// packed struct. Values form a linear LSB-first bit stream where value i
// occupies bits [i*3, i*3+3) and may straddle two bytes.
// ================================================================================================

#include "common.cl"

#define UINT3_BITS      3
#define UINT3_MASK      0x7

// Read the u3 value at element index `idx` from a packed bit stream.
inline uchar unpack_uint3_at(const __global uchar* base, uint idx) {
    const uint bit_off = idx * UINT3_BITS;
    const uint byte_idx = bit_off >> 3;
    const uint bit_shift = bit_off & 7;

    uint window = base[byte_idx];
    // Only touch the next byte when the value actually crosses into it, so the
    // final element of a buffer never reads past the end.
    if (bit_shift > 8 - UINT3_BITS)
        window |= (uint)base[byte_idx + 1] << 8;

    return (uchar)((window >> bit_shift) & UINT3_MASK);
}

#define UNPACK_UINT3_AT(base, idx) unpack_uint3_at((const __global uchar*)(base), (idx))
