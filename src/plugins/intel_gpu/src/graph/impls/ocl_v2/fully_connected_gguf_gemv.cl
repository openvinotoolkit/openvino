// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_gguf_common.cl"

// y[t][n] = sum_k W[n][k] * x[t][k] with int8 dot products; x quantized by fully_connected_gguf_quantize.
// A work-group of 64 owns 4 weight rows as 2 row pairs. Its 16 units of 4 lanes walk the (row pair, block) items; lane c
// of a unit takes a quarter of the block for both rows of the pair, so each x load serves two rows. K is split over the
// units and reduced through local memory.
// Dispatch: global = (ceil(N / 4) * 64, M), local = (64, 1).

#define WG_SIZE 64
#define ROWS_PER_WG 4
#define SG 16
#define NSG (WG_SIZE / SG)
#define UNITS (NSG * 4)
#define NP (ROWS_PER_WG / 2)

inline int FUNC(dp16_ss)(uint4 a, uint4 b) {  // 16 signed bytes each
    return dot_4x8packed_ss_int(a.x, b.x) + dot_4x8packed_ss_int(a.y, b.y) + dot_4x8packed_ss_int(a.z, b.z) +
           dot_4x8packed_ss_int(a.w, b.w);
}

#if GGUF_Q5_K
inline int FUNC(dp32_us)(uint8 a, uint8 b) {  // a: 32 unsigned bytes, b: 32 signed bytes
    return dot_4x8packed_us_int(a.s0, b.s0) + dot_4x8packed_us_int(a.s1, b.s1) + dot_4x8packed_us_int(a.s2, b.s2) +
           dot_4x8packed_us_int(a.s3, b.s3) + dot_4x8packed_us_int(a.s4, b.s4) + dot_4x8packed_us_int(a.s5, b.s5) +
           dot_4x8packed_us_int(a.s6, b.s6) + dot_4x8packed_us_int(a.s7, b.s7);
}

// Lane c: sub-blocks 2c (low nibbles) and 2c + 1 (high nibbles). The 5th bits are merged in registers, so q stays
// unsigned; the scales and mins are unpacked as 16-bit pairs without branches.
// Returns d * sc * (x . q) - dmin * m * sum(x) for both sub-blocks.
inline float FUNC(chunk_dot)(const __global uchar* blk, int c, uint8 xl, uint8 xh, float2 dx, float4 sx) {
    const uint4 hd = vload4(0, (const __global uint*)blk);
    const uint8 qh = vload8(0, (const __global uint*)(blk + GGUF_Q5K_QH));
    const uint8 qw = vload8(0, (const __global uint*)(blk + GGUF_Q5K_QS + 32 * c));
    const uint8 ql = (qw & (uint8)(0x0F0F0F0F)) | (((qh >> (2 * c)) & (uint8)(0x01010101)) << 4);
    const uint8 qu = ((qw >> 4) & (uint8)(0x0F0F0F0F)) | (((qh >> (2 * c + 1)) & (uint8)(0x01010101)) << 4);
    const int il = FUNC_CALL(dp32_us)(ql, xl), ih = FUNC_CALL(dp32_us)(qu, xh);
    const float2 dm = convert_float2(as_half2(hd.x));
    const int sh = (c & 1) * 16;
    const uint A = (hd.y >> sh) & 0xFFFF, B = (hd.z >> sh) & 0xFFFF, C = (hd.w >> sh) & 0xFFFF;
    const bool hi = c >= 2;
    const uint scp = hi ? ((C & 0x0F0F) | ((A >> 2) & 0x3030)) : (A & 0x3F3F);
    const uint mnp = hi ? (((C >> 4) & 0x0F0F) | ((B >> 2) & 0x3030)) : (B & 0x3F3F);
    return dm.x * ((float)(scp & 0xFF) * dx.x * (float)il + (float)(scp >> 8) * dx.y * (float)ih) -
           dm.y * ((float)(mnp & 0xFF) * (sx.x + sx.y) + (float)(mnp >> 8) * (sx.z + sx.w));
}

#    define ITEM(J)                                                                                                      \
        {                                                                                                                \
            const int p = (J) / nb;                                                                                      \
            const int b = (J)-p * nb;                                                                                    \
            const int ra = row0 + 2 * p;                                                                                 \
            const int kl = b * 256 + 64 * c;                                                                             \
            const uint8 xl = vload8(0, (const __global uint*)(xq + kl));                                                 \
            const uint8 xh = vload8(0, (const __global uint*)(xq + kl + 32));                                            \
            const float2 dx = vload2(0, xd + (kl >> 5));                                                                 \
            const float4 sx = vload4(0, xs + (kl >> 4));                                                                 \
            const float ca = FUNC_CALL(chunk_dot)(W + min(ra, N - 1) * rs + b * GGUF_Q5K_BLOCK_BYTES, c, xl, xh, dx, sx); \
            const float cb = FUNC_CALL(chunk_dot)(W + min(ra + 1, N - 1) * rs + b * GGUF_Q5K_BLOCK_BYTES, c, xl, xh, dx, sx); \
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < NP; q++) {                                           \
                acc[2 * q] += p == q ? ca : 0.0f;                                                                        \
                acc[2 * q + 1] += p == q ? cb : 0.0f;                                                                    \
            }                                                                                                            \
        }

#else  // GGUF_Q6_K
// q in 0..63 -> q - 32 as int8, without carries between bytes: (q + 96) ^ 0x80
inline uint8 FUNC(to_s8)(uint8 q) {
    return (q + (uint8)(0x60606060)) ^ (uint8)(0x80808080);
}

// Lane c: QL bytes 32c .. 32c + 31 of the block; their low nibbles are values 128h + 32hi + (0..31), the high nibbles
// those + 64 (h = c / 2, hi = c % 2).
inline float FUNC(chunk_dot)(const __global uchar* QL,
                             const __global uchar* QH,
                             const __global uint* SC,
                             const __global half* D,
                             size_t blk,
                             int c,
                             uint8 xl,
                             uint8 xh,
                             float2 dx) {
    const int h = c >> 1, sh = (c & 1) * 2, s16 = (c & 1) * 16;
    const uint8 ql = vload8(0, (const __global uint*)(QL + blk * 128 + 32 * c));
    const uint8 qh = vload8(0, (const __global uint*)(QH + blk * 64 + 32 * h));
    const uint2 sw = vload2(0, SC + blk * 4 + 2 * h);
    const float d = vload_half(blk, D);
    const uint8 lo = FUNC_CALL(to_s8)((ql & (uint8)(0x0F0F0F0F)) | (((qh >> sh) & (uint8)(0x03030303)) << 4));
    const uint8 hi = FUNC_CALL(to_s8)(((ql >> 4) & (uint8)(0x0F0F0F0F)) | (((qh >> (sh + 4)) & (uint8)(0x03030303)) << 4));
    const float s0 = (float)(char)(sw.x >> s16), s1 = (float)(char)(sw.x >> (s16 + 8));
    const float s2 = (float)(char)(sw.y >> s16), s3 = (float)(char)(sw.y >> (s16 + 8));
    const int l0 = FUNC_CALL(dp16_ss)(lo.lo, xl.lo), l1 = FUNC_CALL(dp16_ss)(lo.hi, xl.hi);
    const int h0 = FUNC_CALL(dp16_ss)(hi.lo, xh.lo), h1 = FUNC_CALL(dp16_ss)(hi.hi, xh.hi);
    return d * (dx.x * (s0 * (float)l0 + s1 * (float)l1) + dx.y * (s2 * (float)h0 + s3 * (float)h1));
}

#define ITEM(J)                                                                                       \
    {                                                                                                 \
        const int p = (J) / nb;                                                                       \
        const int b = (J)-p * nb;                                                                     \
        const int ra = row0 + 2 * p;                                                                  \
        const int kl = b * 256 + 32 * (4 * (c >> 1) + (c & 1));                                       \
        const uint8 xl = vload8(0, (const __global uint*)(xq + kl));                                  \
        const uint8 xh = vload8(0, (const __global uint*)(xq + kl + 64));                             \
        const float2 dx = (float2)(xd[kl >> 5], xd[(kl >> 5) + 2]);                                   \
        const float ca = FUNC_CALL(chunk_dot)(QL, QH, SC, D, (size_t)min(ra, N - 1) * nb + b, c, xl, xh, dx);     \
        const float cb = FUNC_CALL(chunk_dot)(QL, QH, SC, D, (size_t)min(ra + 1, N - 1) * nb + b, c, xl, xh, dx); \
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < NP; q++) {                            \
            acc[2 * q] += p == q ? ca : 0.0f;                                                         \
            acc[2 * q + 1] += p == q ? cb : 0.0f;                                                     \
        }                                                                                             \
    }
#endif

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
KERNEL(fully_connected_gguf_gemv)(OPTIONAL_SHAPE_INFO_ARG
                                  const __global uchar* W,
                                  const __global char* xpack,
                                  __global OUTPUT_TYPE* output
#if GGUF_FUSED_ADD
                                  , const __global OUTPUT_TYPE* residual
#endif
                                  ) {
    const int N = GGUF_N, K = GGUF_K;
    const int t = get_global_id(1);
    const __global char* xq = xpack + (size_t)t * GGUF_XPACK;
    const __global float* xd = (const __global float*)(xq + K);
#if GGUF_Q5_K
    const __global float* xs = (const __global float*)(xq + K + K / 8);
#endif
    __global OUTPUT_TYPE* y = output + (size_t)t * N;
    __local float part[NSG][ROWS_PER_WG];
    const int sg = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int c = lane & 3;
    const int unit = sg * 4 + (lane >> 2);
    const int row0 = get_group_id(0) * ROWS_PER_WG;
    const int nb = K >> 8;
#if GGUF_Q5_K
    const size_t rs = (size_t)nb * GGUF_Q5K_BLOCK_BYTES;
#else
    const size_t nblk = (size_t)N * nb;
    const __global uchar* QL = W;
    const __global uchar* QH = GGUF_Q6K_QH(W, nblk);
    const __global uint* SC = (const __global uint*)GGUF_Q6K_SC(W, nblk);
    const __global half* D = (const __global half*)GGUF_Q6K_D(W, nblk);
#endif
    const int ni = NP * nb;
    float acc[ROWS_PER_WG];
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < ROWS_PER_WG; r++) {
        acc[r] = 0.0f;
    }
    for (int j = unit; j < ni; j += 2 * UNITS) {
        ITEM(j)
        if (j + UNITS < ni) {
            ITEM(j + UNITS)
        }
    }

    __attribute__((opencl_unroll_hint)) for (int r = 0; r < ROWS_PER_WG; r++) {
        const float s = sub_group_reduce_add(acc[r]);
        if (lane == 0) {
            part[sg][r] = s;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int lid = get_local_id(0);
    if (lid < ROWS_PER_WG && row0 + lid < N) {
        float s = 0.0f;
        for (int i = 0; i < NSG; i++) {
            s += part[i][lid];
        }
#if GGUF_FUSED_ADD
        s += (float)residual[(size_t)t * N + row0 + lid];
#endif
        y[row0 + lid] = (OUTPUT_TYPE)s;
    }
}

#undef WG_SIZE
#undef ROWS_PER_WG
#undef SG
#undef NSG
#undef UNITS
#undef NP
#undef ITEM
