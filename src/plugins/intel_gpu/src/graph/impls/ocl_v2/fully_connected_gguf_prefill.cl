// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_gguf_common.cl"

// y[t][n] = sum_k W[n][k] * x[t][k] with f16 DPAS, for many activation rows.
// A work-group of SGM x SGN sub-groups (set by the host) owns NT = SGN * TN * 16 weight rows and SGM * TM * 8 activation
// rows. For each 256-value block of K, the work-group decodes the block of its NT weight rows into local memory in the
// DPAS B layout; sub-group (sm, sn) then multiplies TM x TN tiles of 8 x 16 with A from 2D block reads.
// The loads of block b + 1 are issued when block b starts and their decode is spread over the 16 k steps of block b;
// local memory is double buffered.
// Decode task = (weight row r, chunk c of 64 values) = 16 pieces of 4 values.
// Local layout of a block: [kk][tile of 16 weight rows][int j][lane].
// Dispatch: global = (ceil(N / NT) * 16, ceil(M / (SGM * TM * 8)) * SGM * SGN), local = (16, SGM * SGN).

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

#define NSG (SGM * SGN)
#define NTN (SGN * TN)
#define NT (16 * NTN)
#define NTASK ((NT * 4 + 16 * NSG - 1) / (16 * NSG))
// The 16 lanes of a sub-group take 16 consecutive rows of one chunk, so a piece store is 16 consecutive dwords.
#define TASK_ROW(task) ((task) % NT)
#define TASK_CHUNK(task) ((task) / NT)
// Split barrier: a sub-group waits only before its first local memory read of the next block.
#ifdef cl_intel_split_work_group_barrier
#    define GGUF_SPLIT_BARRIER 1
#else
#    define GGUF_SPLIT_BARRIER 0
#endif
#if TM == 1
#    define READ_A intel_sub_group_2d_block_read_16b_8r16x1c
#elif TM == 2
#    define READ_A intel_sub_group_2d_block_read_16b_16r16x1c
#elif TM == 4
#    define READ_A intel_sub_group_2d_block_read_16b_32r16x1c
#else
#    error "TM must be 1, 2 or 4"
#endif

inline uint FUNC(pick8)(uint8 v, int j) {
    return j == 0 ? v.s0 : j == 1 ? v.s1 : j == 2 ? v.s2 : j == 3 ? v.s3 : j == 4 ? v.s4 : j == 5 ? v.s5 : j == 6 ? v.s6 : v.s7;
}

// Two quants (< 64) -> two weights q * ws - wm: q in the low mantissa bits of 1024.0h is exactly 1024 + q.
inline half2 FUNC(dequant2)(uint q2, half2 ws, half2 wm) {
    return fma(as_half2(q2 | 0x64006400u) - (half2)(1024.0h), ws, -wm);
}

// Local memory index of a piece = slm_base(chunk start kc, weight row r) + slm_piece(dk, j) for its values kc + dk + 4j
// (dk = 32h for Q5_K, 64h for Q6_K).
inline int FUNC(slm_base)(int kc, int r) {
    return (((kc >> 4) * NTN + (r >> 4)) * 8) * 16 + (r & 15);
}
inline int FUNC(slm_piece)(int dk, int j) {
    return (((dk >> 4) + (j >> 2)) * NTN * 8 + 2 * (j & 3)) * 16;
}

#if GGUF_Q5_K
// Chunk c = qs bytes 32c ..: low nibbles are sub-block 2c, high nibbles sub-block 2c + 1; 5th bits 2c, 2c + 1 of qh.
typedef struct {
    uint8 ql;     // the chunk's 32 qs bytes
    uint8 qh;     // the block's 32 qh bytes
    half2 ws[2];  // d * sc of sub-blocks 2c, 2c + 1
    half2 wm[2];  // dmin * m of sub-blocks 2c, 2c + 1
    int base;     // local memory index of piece 0
} FUNC(task_t);

// row: the weight row to read (clamped to N - 1), r: its slot in the work-group's NT rows
inline FUNC(task_t) FUNC(load_task)(const __global uchar* W, int b, int row, int r, int c) {
    const __global uchar* blk = W + ((size_t)row * (GGUF_K / 256) + b) * GGUF_Q5K_BLOCK_BYTES;
    FUNC(task_t) t;
    t.ql = vload8(0, (const __global uint*)(blk + GGUF_Q5K_QS + 32 * c));
    t.qh = vload8(0, (const __global uint*)(blk + GGUF_Q5K_QH));
    const float d = vload_half(0, (const __global half*)blk);
    const float dmin = vload_half(1, (const __global half*)blk);
    __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++) {
        const int2 sm = FUNC_CALL(q5k_scale_min)(blk + GGUF_Q5K_SCALES, 2 * c + h);
        t.ws[h] = (half2)((half)(d * (float)sm.x));
        t.wm[h] = (half2)((half)(dmin * (float)sm.y));
    }
    t.base = FUNC_CALL(slm_base)(64 * c, r);
    return t;
}

// Piece p (0..15) of task (row r, chunk c): h = p >> 3 picks the low or high nibbles, j = p & 7 the 4-byte group.
inline void FUNC(decode_piece)(const FUNC(task_t)* t, int r, int c, int p, __local int* dst) {
    const int h = p >> 3, j = p & 7;
    const uint w = FUNC_CALL(pick8)(t->ql, j);
    const uint q = (h ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu)) | (((FUNC_CALL(pick8)(t->qh, j) >> (2 * c + h)) & 0x01010101u) << 4);
    const uint q01 = (q & 0x1Fu) | ((q & 0x1F00u) << 8);
    const uint q23 = ((q >> 16) & 0x1Fu) | ((q >> 8) & 0x1F0000u);
    const int base = t->base + FUNC_CALL(slm_piece)(32 * h, j);
    dst[base] = as_int(FUNC_CALL(dequant2)(q01, t->ws[h], t->wm[h]));
    dst[base + 16] = as_int(FUNC_CALL(dequant2)(q23, t->ws[h], t->wm[h]));
}
#else  // GGUF_Q6_K
typedef struct {
    uint8 ql;     // 32 low-bit bytes: values v0 + l (low nibbles), v0 + 64 + l (high nibbles)
    uint8 qh;     // the half's 32 high-bit bytes: bits sh, sh + 1 (low nibbles), sh + 4, sh + 5 (high nibbles)
    half2 ws[4];  // d * scale of values v0 .. v0 + 15, v0 + 16 .., v0 + 64 .., v0 + 80 ..
    half2 wm[4];  // 32 * d * scale: w = (q - 32) * d * scale
    int base;     // local memory index of piece 0
} FUNC(task_t);

// row: the weight row to read (clamped to N - 1), r: its slot in the work-group's NT rows
inline FUNC(task_t) FUNC(load_task)(const __global uchar* W, int b, int row, int r, int c) {
    const size_t nblk = (size_t)GGUF_N * (GGUF_K / 256);
    const size_t blk = (size_t)row * (GGUF_K / 256) + b;
    FUNC(task_t) t;
    t.ql = vload8(0, (const __global uint*)(W + blk * 128 + 32 * c));
    t.qh = vload8(0, (const __global uint*)(GGUF_Q6K_QH(W, nblk) + blk * 64 + 32 * (c >> 1)));
    const int v0 = 128 * (c >> 1) + 32 * (c & 1);
    const __global char* sc = (const __global char*)(GGUF_Q6K_SC(W, nblk) + blk * 16) + (v0 >> 4);
    const float d = vload_half(blk, (const __global half*)GGUF_Q6K_D(W, nblk));
    const int si[4] = {0, 1, 4, 5};
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < 4; i++) {
        const half ws = (half)(d * (float)sc[si[i]]);
        t.ws[i] = (half2)(ws);
        t.wm[i] = (half2)(ws * 32.0h);
    }
    t.base = FUNC_CALL(slm_base)(v0, r);
    return t;
}

// Piece p (0..15) of task (row r, chunk c): h = p >> 3 picks the low or high nibbles, j = p & 7 the 4-byte group.
inline void FUNC(decode_piece)(const FUNC(task_t)* t, int r, int c, int p, __local int* dst) {
    const int h = p >> 3, j = p & 7;
    const uint w = FUNC_CALL(pick8)(t->ql, j);
    const int sh = 2 * (c & 1) + 4 * h;
    const uint q = (h ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu)) | (((FUNC_CALL(pick8)(t->qh, j) >> sh) & 0x03030303u) << 4);
    const int si = 2 * h + (j >> 2);
    const uint q01 = (q & 0x3Fu) | ((q & 0x3F00u) << 8);
    const uint q23 = ((q >> 16) & 0x3Fu) | ((q >> 8) & 0x3F0000u);
    const int base = t->base + FUNC_CALL(slm_piece)(64 * h, j);
    dst[base] = as_int(FUNC_CALL(dequant2)(q01, t->ws[si], t->wm[si]));
    dst[base + 16] = as_int(FUNC_CALL(dequant2)(q23, t->ws[si], t->wm[si]));
}
#endif

__attribute__((reqd_work_group_size(16, NSG, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(fully_connected_gguf_prefill)(OPTIONAL_SHAPE_INFO_ARG
                                     const __global uchar* W,
                                     const __global INPUT0_TYPE* x,
                                     __global OUTPUT_TYPE* y,
#if GGUF_FUSED_ADD
                                     const __global OUTPUT_TYPE* residual,
#endif
                                     int M) {
    const int lane = get_local_id(0);
    const int lid = get_local_id(1) * 16 + lane;
    const int sm = get_local_id(1) % SGM, sn = get_local_id(1) / SGM;
    const int n0 = get_group_id(0) * NT;
    const int t0 = get_group_id(1) * (SGM * TM * 8) + sm * (TM * 8);
    const int nb = GGUF_K / 256;

    __local int slm[2][16 * NT * 8];

    float8 acc[TM][TN];
    __attribute__((opencl_unroll_hint)) for (int tm = 0; tm < TM; tm++) {
        __attribute__((opencl_unroll_hint)) for (int tn = 0; tn < TN; tn++) {
            acc[tm][tn] = (float8)(0.0f);
        }
    }

    // block 0: load and decode up front
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < NTASK; i++) {
        const int task = lid + i * 16 * NSG;
        if (task < NT * 4) {
            const int r = TASK_ROW(task), c = TASK_CHUNK(task);
            const FUNC(task_t) t = FUNC_CALL(load_task)(W, 0, min(n0 + r, GGUF_N - 1), r, c);
            __attribute__((opencl_unroll_hint)) for (int p = 0; p < 16; p++) {
                FUNC_CALL(decode_piece)(&t, r, c, p, slm[0]);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int b = 0; b < nb; b++) {
        const bool more = b + 1 < nb;
        __local int* nxt = slm[(b + 1) & 1];
        const __local int* cur = slm[b & 1];
        FUNC(task_t) tk[NTASK];
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < NTASK; i++) {
            const int task = min(lid + i * 16 * NSG, NT * 4 - 1);
            tk[i] = FUNC_CALL(load_task)(W, min(b + 1, nb - 1), min(n0 + TASK_ROW(task), GGUF_N - 1), TASK_ROW(task), TASK_CHUNK(task));
        }
        __attribute__((opencl_unroll_hint)) for (int kk = 0; kk < 16; kk++) {
            const int k = b * 256 + kk * 16;
            ushort ar[TM * 8];
            READ_A((const __global void*)x, GGUF_K * 2, M, GGUF_K * 2, (int2)(k, t0), ar);
#if GGUF_SPLIT_BARRIER
            if (kk == 0 && b > 0) {
                intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
            }
#endif
            int8 bt[TN];
            __attribute__((opencl_unroll_hint)) for (int tn = 0; tn < TN; tn++) {
                bt[tn] = as_int8(intel_sub_group_block_read8((const __local uint*)(cur + (kk * NTN + sn * TN + tn) * 128)));
            }
            __attribute__((opencl_unroll_hint)) for (int tm = 0; tm < TM; tm++) {
                const short8 a = (short8)(as_short(ar[tm * 8 + 0]), as_short(ar[tm * 8 + 1]), as_short(ar[tm * 8 + 2]), as_short(ar[tm * 8 + 3]),
                                          as_short(ar[tm * 8 + 4]), as_short(ar[tm * 8 + 5]), as_short(ar[tm * 8 + 6]), as_short(ar[tm * 8 + 7]));
                __attribute__((opencl_unroll_hint)) for (int tn = 0; tn < TN; tn++) {
                    acc[tm][tn] = intel_sub_group_f16_f16_matrix_mad_k16(a, bt[tn], acc[tm][tn]);
                }
            }
            // a sixteenth of the next block's decode between this step's matrix instructions and the next step's
            if (more) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < NTASK; i++) {
                    const int task = lid + i * 16 * NSG;
                    if (task < NT * 4) {
                        FUNC_CALL(decode_piece)(&tk[i], TASK_ROW(task), TASK_CHUNK(task), kk, nxt);
                    }
                }
            }
        }
#if GGUF_SPLIT_BARRIER
        if (more) {
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
        }
#else
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

    __attribute__((opencl_unroll_hint)) for (int tn = 0; tn < TN; tn++) {
        const int n = n0 + (sn * TN + tn) * 16 + lane;
        if (n < GGUF_N) {
            __attribute__((opencl_unroll_hint)) for (int tm = 0; tm < TM; tm++) {
                const float r8[8] = {acc[tm][tn].s0, acc[tm][tn].s1, acc[tm][tn].s2, acc[tm][tn].s3,
                                     acc[tm][tn].s4, acc[tm][tn].s5, acc[tm][tn].s6, acc[tm][tn].s7};
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < 8; r++) {
                    const int t = t0 + tm * 8 + r;
                    if (t < M) {
#if GGUF_FUSED_ADD
                        y[(size_t)t * GGUF_N + n] = (OUTPUT_TYPE)(r8[r] + (float)residual[(size_t)t * GGUF_N + n]);
#else
                        y[(size_t)t * GGUF_N + n] = (OUTPUT_TYPE)r8[r];
#endif
                    }
                }
            }
        }
    }
}

#undef SGM
#undef SGN
#undef NSG
#undef NTN
#undef TM
#undef TN
#undef NT
#undef NTASK
#undef GGUF_SPLIT_BARRIER
#undef TASK_ROW
#undef TASK_CHUNK
#undef READ_A
