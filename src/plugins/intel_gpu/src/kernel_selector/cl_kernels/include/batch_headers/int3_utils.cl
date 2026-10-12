//  Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ================================================================================================
// U3 (3-bit unsigned) support
//
// Unlike u2/u4, three bits do not divide a byte evenly, so u3 has no fixed
// packed struct. Values form a linear LSB-first bit stream where value i
// occupies bits [i*3, i*3+3) and may straddle two bytes. Eight values fit
// exactly in three bytes, which is the natural granule for bulk reads.
// ================================================================================================

#include "common.cl"

#define UINT3_BITS      3
#define UINT3_MASK      0x7
// Eight u3 values occupy exactly three bytes.
#define UINT3_GRANULE   8
#define UINT3_GRANULE_BYTES 3

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

// Unpack the eight u3 values held in the three bytes at `base`.
inline uchar8 unpack_uint3x8(const __global uchar* base) {
    const uint w = (uint)base[0] | ((uint)base[1] << 8) | ((uint)base[2] << 16);
    return (uchar8)((uchar)((w >>  0) & UINT3_MASK),
                    (uchar)((w >>  3) & UINT3_MASK),
                    (uchar)((w >>  6) & UINT3_MASK),
                    (uchar)((w >>  9) & UINT3_MASK),
                    (uchar)((w >> 12) & UINT3_MASK),
                    (uchar)((w >> 15) & UINT3_MASK),
                    (uchar)((w >> 18) & UINT3_MASK),
                    (uchar)((w >> 21) & UINT3_MASK));
}

inline char8 unpack_uint3x8_to_char(const __global uchar* base) {
    return convert_char8(unpack_uint3x8(base));
}

inline float8 unpack_uint3x8_to_float(const __global uchar* base) {
    return convert_float8(unpack_uint3x8(base));
}

#if defined(cl_khr_fp16)
inline half8 unpack_uint3x8_to_half(const __global uchar* base) {
    return convert_half8(unpack_uint3x8(base));
}
#endif

#define UNPACK_UINT3_AT(base, idx) unpack_uint3_at((const __global uchar*)(base), (idx))
#define UNPACK_UINT3x8(base)       unpack_uint3x8((const __global uchar*)(base))
