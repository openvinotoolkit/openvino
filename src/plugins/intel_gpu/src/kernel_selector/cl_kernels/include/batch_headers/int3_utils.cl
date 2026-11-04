//  Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// u3 weights are a tightly packed LSB-first bit stream: element i occupies bits [3 * i, 3 * i + 3),
// so it may span two bytes. The second byte is read only when needed to stay inside the buffer.
inline uchar get_uint3_weight(const __global uchar* weights, uint idx) {
    const uint bit_offset = idx * 3;
    const uint byte_idx = bit_offset / 8;
    const uint shift = bit_offset % 8;
    uint packed = weights[byte_idx];
    if (shift > 5)
        packed |= (uint)weights[byte_idx + 1] << 8;
    return (uchar)((packed >> shift) & 0x7);
}
