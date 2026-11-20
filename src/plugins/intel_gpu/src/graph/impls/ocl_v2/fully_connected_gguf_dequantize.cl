// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_gguf_common.cl"

// Decodes the GGUF weights into f16 [N, K] for the oneDNN matmul. One work-item per 16 values (one scale).
// Dispatch: global = N * K / 16 rounded up to the local size.
KERNEL(fully_connected_gguf_dequantize)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* W, __global half* weights_f16) {
    const uint nb = GGUF_K / 256;
    const size_t id = get_global_id(0);
    if (id >= (size_t)GGUF_N * nb * 16) {
        return;
    }
    const uint t = id & 15;      // values 16t .. 16t + 15 of the block
    const size_t blk = id >> 4;  // row * nb + block
    const uint l0 = (t & 1) * 16;

#if GGUF_Q5_K
    // Sub-block j = values 32j ..: low (j even) or high (j odd) nibbles of qs[32 * (j / 2) ..], bit j of qh.
    const __global uchar* b = W + blk * GGUF_Q5K_BLOCK_BYTES;
    const uint j = t >> 1;
    const float d = vload_half(0, (const __global half*)b);
    const float dmin = vload_half(1, (const __global half*)b);
    const int2 sm = FUNC_CALL(q5k_scale_min)(b + GGUF_Q5K_SCALES, j);
    const int16 ql = convert_int16(vload16(0, b + GGUF_Q5K_QS + 32 * (j >> 1) + l0));
    const int16 qh = convert_int16(vload16(0, b + GGUF_Q5K_QH + l0));
    const int16 q = ((j & 1) ? (ql >> 4) : (ql & 0xF)) | (((qh >> j) & 1) << 4);
    vstore_half16(d * (float)sm.x * convert_float16(q) - dmin * (float)sm.y, 0, weights_f16 + blk * 256 + 16 * t);
#else  // GGUF_Q6_K
    // Half n of 128 values, quarter q4: low (q4 < 2) or high nibbles of QL[64n + 32(q4 & 1) ..], bits 2q4.. of QH[32n ..].
    const size_t nblk = (size_t)GGUF_N * nb;
    const uint n = t >> 3;
    const uint q4 = (t >> 1) & 3;
    const float d = vload_half(blk, (const __global half*)GGUF_Q6K_D(W, nblk));
    const float sc = (float)((const __global char*)GGUF_Q6K_SC(W, nblk))[blk * 16 + t];
    const int16 ql = convert_int16(vload16(0, W + blk * 128 + 64 * n + 32 * (q4 & 1) + l0));
    const int16 qh = convert_int16(vload16(0, GGUF_Q6K_QH(W, nblk) + blk * 64 + 32 * n + l0));
    const int16 q = ((q4 >= 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (2 * q4)) & 3) << 4);
    vstore_half16(d * sc * convert_float16(q - 32), 0, weights_f16 + blk * 256 + 16 * t);
#endif
}
