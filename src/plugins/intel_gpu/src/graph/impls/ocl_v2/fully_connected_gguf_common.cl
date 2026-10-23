// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight layouts shared by the fully_connected_gguf kernels.

// Q5_K block, as stored: f16 d, f16 dmin, 12 bytes of 6-bit scales and mins (8 sub-blocks of 32), qh[32], qs[128].
// w = d * sc * q - dmin * m, q = 4 bits from qs + 1 bit from qh.
#define GGUF_Q5K_BLOCK_BYTES 176
#define GGUF_Q5K_SCALES 4
#define GGUF_Q5K_QH 16
#define GGUF_Q5K_QS 48

// Q6_K, regrouped by the plugin into planes over all nblk blocks: QL [nblk][128] | QH [nblk][64] | SC [nblk][16] | D [nblk].
// w = d * sc * (q - 32), q = 4 bits from QL + 2 bits from QH, one int8 scale per 16 values.
#define GGUF_Q6K_QH(W, nblk) ((W) + (nblk) * 128)
#define GGUF_Q6K_SC(W, nblk) ((W) + (nblk) * 192)
#define GGUF_Q6K_D(W, nblk) ((W) + (nblk) * 208)

// Scale and min of Q5_K sub-block j (ggml get_scale_min_k4).
inline int2 FUNC(q5k_scale_min)(const __global uchar* s, int j) {
    if (j < 4) {
        return (int2)(s[j] & 63, s[j + 4] & 63);
    }
    return (int2)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4));
}
