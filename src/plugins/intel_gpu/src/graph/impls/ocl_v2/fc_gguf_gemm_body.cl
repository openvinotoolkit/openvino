// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF prefill GEMM pipeline template (ported from the CM gemm_q*_full.cm kernels). Included by
// fc_gguf_gemm_opt.cl after it has defined the per-format tile codec:
//
//   GG_RAW_T gg_fetch(const __global uchar* W, uint num_blocks, uint blk, uint j)   // raw encoded tile
//   void gg_decode(GG_RAW_T raw, uint j, uint8* blo, uint8* bhi)                  // -> B operands
//
// for sub-block `j` of row-group/K-block `blk` and this lane's output column; blo/bhi are the two
// B operands (K 0..15, K 16..31) of intel_sub_group_f16_f16_matrix_mad_k16.
//
// Work decomposition (identical to the CM kernel, with the 16 lanes of a CM thread becoming one
// sub-group): work-group = TOKEN_LOCAL sub-groups, each owning TOKENS_PER_THREAD tokens x
// (ROW_GROUPS * 16) outputs. The sub-groups of a work-group share the weight tile: every phase the
// SLM_NPROD (row group, sub-block) tiles of that phase are decoded cooperatively (one per
// sub-group), published through an SLM ring, then consumed by all sub-groups.
//
// JIT constants: ROW_GROUPS, TOKEN_GROUPS, TOKEN_LOCAL, ROW_BLOCKS, SLM_JBLK, PF_A, GG_A_F32, GG_OUT_F32.

#define GG_TOKENS_PER_TILE 8
#define GG_SLM_SLOTS 2
#define GG_SLM_NPROD (SLM_JBLK * ROW_GROUPS * ROW_BLOCKS)
#define GG_TOKEN_SLOTS (TOKEN_LOCAL / ROW_BLOCKS)
#define GG_TOKENS_PER_THREAD (GG_TOKENS_PER_TILE * TOKEN_GROUPS)
#define GG_SLM_JPS (8 / SLM_JBLK)
#define GG_TILE_U 128
#define GG_PHASE_U (GG_SLM_NPROD * 2 * GG_TILE_U)
#if GG_TOKENS_PER_THREAD > 32
#define GG_PFA_ROWS 32
#else
#define GG_PFA_ROWS GG_TOKENS_PER_THREAD
#endif
#define GG_PFA_SETS (GG_TOKENS_PER_THREAD / GG_PFA_ROWS)

// Prefetch GG_PFA_ROWS rows x 32 K columns of the next K-block's activations (cache-line sized window).
#if GG_PFA_ROWS == 32
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_32r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_32r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));   \
    intel_sub_group_2d_block_prefetch_32b_32r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#elif GG_PFA_ROWS == 16
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_16r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_16r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));   \
    intel_sub_group_2d_block_prefetch_32b_16r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#else
#define GG_PREFETCH_A_F16(A, K, M, pc) intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (pc))
#define GG_PREFETCH_A_F32(A, K, M, pc)                                                                  \
    intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc));    \
    intel_sub_group_2d_block_prefetch_32b_8r16x1c((__global void*)(A), (K) * 4, (M), (K) * 4, (pc) + (int2)(16, 0))
#endif

#if GG_SLM_NPROD >= TOKEN_LOCAL
#define GG_PER_SG (GG_SLM_NPROD / TOKEN_LOCAL)
#else
#define GG_PER_SG 1
#endif

// (blk, j) of producer slot w (w = lid + u * TOKEN_LOCAL) in phase s.
#define GG_TILE_COORD(s, w, blk, j)                                              \
    const uint kb_##blk = (s) / GG_SLM_JPS;                                      \
    const uint jb_##blk = (s) - kb_##blk * GG_SLM_JPS;                           \
    const uint rbp_##blk = (w) / (ROW_GROUPS * SLM_JBLK);                        \
    const uint rem_##blk = (w) - rbp_##blk * (ROW_GROUPS * SLM_JBLK);            \
    const uint rg_##blk = rem_##blk / SLM_JBLK;                                  \
    const uint j = jb_##blk * SLM_JBLK + (rem_##blk - rg_##blk * SLM_JBLK);      \
    const uint blk = ((gh * ROW_BLOCKS + rbp_##blk) * ROW_GROUPS + rg_##blk) * nbpr + kb_##blk

// Load the raw (still encoded) weight tiles this sub-group produces in phase s.
inline void FUNC(gg_fetch_stage)(const __global uchar* W, uint num_blocks, uint nbpr, uint gh, uint lid, uint s, uint nstages,
                                 GG_RAW_T* raw) {
    if (s >= nstages) {
        return;
    }
    __attribute__((opencl_unroll_hint))
    for (uint u = 0; u < GG_PER_SG; ++u) {
        const uint w = lid + u * TOKEN_LOCAL;
        if (GG_SLM_NPROD >= TOKEN_LOCAL || w < GG_SLM_NPROD) {
            GG_TILE_COORD(s, w, blk, j);
            raw[u] = FUNC_CALL(gg_fetch)(W, num_blocks, blk, j);
        }
    }
}

// Decode the raw tiles of phase s and publish them into SLM slot (s & 1).
inline void FUNC(gg_store_stage)(__local uint* slm, uint lid, uint lane, uint s, uint nstages, const GG_RAW_T* raw) {
    if (s >= nstages) {
        return;
    }
    const uint sbase = (s & (GG_SLM_SLOTS - 1)) * GG_PHASE_U;
    const uint jb = s % GG_SLM_JPS;
    __attribute__((opencl_unroll_hint))
    for (uint u = 0; u < GG_PER_SG; ++u) {
        const uint w = lid + u * TOKEN_LOCAL;
        if (GG_SLM_NPROD >= TOKEN_LOCAL || w < GG_SLM_NPROD) {
            const uint j = jb * SLM_JBLK + w % SLM_JBLK;
            uint8 blo;
            uint8 bhi;
            FUNC_CALL(gg_decode)(raw[u], j, &blo, &bhi);
            __local uint* dst = slm + sbase + (w * 2) * GG_TILE_U;
            FUNC_CALL(gg_slm_wr8)(dst, lane, blo);
            FUNC_CALL(gg_slm_wr8)(dst + GG_TILE_U, lane, bhi);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, TOKEN_LOCAL, 1)))
KERNEL(fc_gguf_gemm)(OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,
    const __global uchar* W,
    __global OUTPUT_TYPE* C,
    uint M,
    uint K,
    uint N
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint lane = get_sub_group_local_id();
    const uint nbpr = K / 256;
    const uint num_blocks = (K / 256) * N;

#if GG_N_BAND
    // Walk the token blocks of GG_N_BAND row-group blocks before moving on, which shortens the reuse
    // distance of a weight slice when the whole weight matrix does not fit the GPU caches.
    const uint ngn = get_num_groups(0);
    const uint lin = get_group_id(1) * ngn + get_group_id(0);
    const uint band_groups = GG_N_BAND * get_num_groups(1);
    const uint bi = lin / band_groups;
    const uint brem = lin - bi * band_groups;
    const uint bw = min((uint)GG_N_BAND, ngn - bi * GG_N_BAND);
    const uint gt = brem / bw;
    const uint gh = bi * GG_N_BAND + (brem - gt * bw);
#else
    const uint gh = get_group_id(0);
    const uint gt = get_group_id(1);
#endif
    const uint lid = get_local_id(1);
    const uint rb = lid % ROW_BLOCKS;
    const uint ts = lid / ROW_BLOCKS;
    const uint h0 = (gh * ROW_BLOCKS + rb) * ROW_GROUPS;
    const uint t = gt * GG_TOKEN_SLOTS + ts;
    const uint tile_base = t * GG_TOKENS_PER_THREAD;

    float8 acc[ROW_GROUPS * TOKEN_GROUPS];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < ROW_GROUPS * TOKEN_GROUPS; ++i) {
        acc[i] = (float8)(0.0f);
    }

    __local uint slm[GG_SLM_SLOTS * GG_PHASE_U];

    const uint nstages = nbpr * GG_SLM_JPS;

    // Weight tiles of phase s+1 are decoded into the other SLM slot while phase s is consumed.
    GG_RAW_T raw[GG_PER_SG];
    FUNC_CALL(gg_fetch_stage)(W, num_blocks, nbpr, gh, lid, 0, nstages, raw);
    FUNC_CALL(gg_store_stage)(slm, lid, lane, 0, nstages, raw);

    for (uint s = 0; s < nstages; ++s) {
        barrier(CLK_LOCAL_MEM_FENCE);

        FUNC_CALL(gg_fetch_stage)(W, num_blocks, nbpr, gh, lid, s + 1, nstages, raw);
        FUNC_CALL(gg_store_stage)(slm, lid, lane, s + 1, nstages, raw);

        const uint kb = s / GG_SLM_JPS;
        const uint jb = s - kb * GG_SLM_JPS;
#if PF_A
        if (jb == 0 && kb + 1 < nbpr) {
            __attribute__((opencl_unroll_hint))
            for (uint r = 0; r < GG_PFA_SETS; ++r) {
                const int2 pc = (int2)((kb + 1) * 256, tile_base + r * GG_PFA_ROWS);
#if GG_A_F32
                GG_PREFETCH_A_F32(A, K, M, pc);
#else
                GG_PREFETCH_A_F16(A, K, M, pc);
#endif
            }
        }
#endif
        const uint sbase = (s & (GG_SLM_SLOTS - 1)) * GG_PHASE_U;

        __attribute__((opencl_unroll_hint))
        for (uint jl = 0; jl < SLM_JBLK; ++jl) {
            const uint j = jb * SLM_JBLK + jl;
            short8 alo[TOKEN_GROUPS];
            short8 ahi[TOKEN_GROUPS];
#if !GG_A_F32 && (TOKEN_GROUPS == 2 || TOKEN_GROUPS == 4)
            GG_LOAD_A_MULTI(A, K, M, (int)(kb * 256 + j * 32), (int)tile_base, TOKEN_GROUPS, alo, ahi);
#else
            __attribute__((opencl_unroll_hint))
            for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
                FUNC_CALL(gg_load_a)(A, K, M, (int)(kb * 256 + j * 32), (int)(tile_base + tg * GG_TOKENS_PER_TILE), &alo[tg], &ahi[tg]);
            }
#endif
            __attribute__((opencl_unroll_hint))
            for (uint rg = 0; rg < ROW_GROUPS; ++rg) {
                const uint ti = (rb * ROW_GROUPS + rg) * SLM_JBLK + jl;
                const __local uint* tp = slm + sbase + (ti * 2) * GG_TILE_U;
                const int8 blo = FUNC_CALL(gg_slm_rd8)(tp, lane);
                const int8 bhi = FUNC_CALL(gg_slm_rd8)(tp + GG_TILE_U, lane);
                __attribute__((opencl_unroll_hint))
                for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
                    acc[rg * TOKEN_GROUPS + tg] = intel_sub_group_f16_f16_matrix_mad_k16(alo[tg], blo, acc[rg * TOKEN_GROUPS + tg]);
                    acc[rg * TOKEN_GROUPS + tg] = intel_sub_group_f16_f16_matrix_mad_k16(ahi[tg], bhi, acc[rg * TOKEN_GROUPS + tg]);
                }
            }
        }
    }

    // Epilogue: lane = output column, element i of acc = token row i.
    __attribute__((opencl_unroll_hint))
    for (uint tg = 0; tg < TOKEN_GROUPS; ++tg) {
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < GG_TOKENS_PER_TILE; ++i) {
            const uint bm = tile_base + tg * GG_TOKENS_PER_TILE + i;
            if (bm < M) {
                __attribute__((opencl_unroll_hint))
                for (uint rg = 0; rg < ROW_GROUPS; ++rg) {
                    const uint n = (h0 + rg) * 16 + lane;
                    const float v = FUNC_CALL(gg_f8_el)(acc[rg * TOKEN_GROUPS + tg], i);
#if HAS_FUSED_OPS
                    {
                        const uint out_b = bm / OUTPUT_FEATURE_NUM;
                        const uint out_f = bm - out_b * OUTPUT_FEATURE_NUM;
                        OUTPUT_TYPE dequantized = TO_OUTPUT_TYPE(v);
                        FUSED_OPS;
                        C[(ulong)bm * N + n] = FUSED_OPS_RESULT;
                    }
#elif GG_OUT_F32
                    C[(ulong)bm * N + n] = v;
#else
                    C[(ulong)bm * N + n] = convert_half_rte(v);
#endif
                }
            }
        }
    }
}
