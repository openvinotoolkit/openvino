// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Attention-mask helpers for sdpa_ocl.cl. Included once, by sdpa_ocl.cl only.

#if IS_CAUSAL && BIDIR_MASK
// Image-group scans over token_type_ids, all in LOCAL coordinates. The two boundary scans are
// subgroup-cooperative and uniform, so every lane reaches the break and the sub_group_reduce.

// End (exclusive) of the image group holding the LOCAL index wg_q_end. Never above q: the loop only
// assigns an index below q or the clamped chunk end.
SDPA_OCL_INLINE int FUNC(bidir_scan_end)(const __global int *token_type_ids, const int wg_q_end, const int q,
                                         const int lane_i) {
    int group_end = wg_q_end + 1;
    while (group_end < q) {
        const int chunk_end = min(q, group_end + SUBGROUP_SIZE);  // exclusive
        const int idx = group_end + lane_i;
        const bool ends_group = (idx < chunk_end) && (token_type_ids[idx] != 1);
        const int first = sub_group_reduce_min(ends_group ? idx : INT_MAX);
        if (first != INT_MAX) {
            group_end = first;
            break;
        }
        group_end = chunk_end;
    }
    return group_end;
}

// Start of the image group holding the LOCAL index window_begin_local (> 0).
SDPA_OCL_INLINE int FUNC(bidir_scan_begin)(const __global int *token_type_ids, const int window_begin_local,
                                           const int lane_i) {
    int group_begin = window_begin_local;
    while (group_begin > 0) {
        const int chunk_begin = max(0, group_begin - SUBGROUP_SIZE);
        const int idx = chunk_begin + lane_i;
        const bool ends_group = (idx < group_begin) && (token_type_ids[idx] != 1);
        const int last = sub_group_reduce_max(ends_group ? idx : -1);
        if (last >= 0) {
            group_begin = last + 1;
            break;
        }
        group_begin = chunk_begin;
    }
    return group_begin;
}
#endif

#if WITH_ATTN_MASK
// Full 2D mask tile [query x key]: each lane loads its own query row, pre-scaled by iscale, so the
// max loop only adds (sdpa_micro's tile_load_t + unscale). The caller runs it only for
// MASK_IS_FULL_2D, but for a dynamic mask the host infers kind 2 from the stage, so a [B, H, 1, K]
// per-key mask can arrive here: clamp its query row to 0 (every query row IS row 0), or the read
// walks past the single row (OOB -> CL_OUT_OF_RESOURCES or a NaN mask). The same holds for the key
// side of a [B, H, q, 1] mask. The selects fold when MSK_D2/MSK_D3 are literals.
SDPA_OCL_INLINE void FUNC(mask_tile_2d)(OPTIONAL_SHAPE_INFO_ARG
                                        __private float16 (*mask_full)[kq_sg_tile_keys / MASK_VEC_KEYS],
                                        const __global half *msk, const float iscale, const size_t wg_j0,
                                        const size_t sg_j0_kq, const size_t lane, const int key_base) {
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        const int mask_query = (MSK_D2 == 1) ? 0
                                             : (wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane);
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / MASK_VEC_KEYS; ++ii) {
            const int mask_key = key_base + ii * MASK_VEC_KEYS;
            half16 mv = (half16)0.0f;
            if (mask_query < MSK_D2) {
                if (MSK_D3 == 1) {
                    mv = (half16)msk[MSK_OFF(0, 0, mask_query, 0)];
                } else if (mask_key + MASK_VEC_KEYS <= MSK_D3) {
                    mv = vload16(0, msk + MSK_OFF(0, 0, mask_query, mask_key));
                } else {
                    #pragma unroll
                    for (int kk = 0; kk < MASK_VEC_KEYS; ++kk) {
                        if (mask_key + kk < MSK_D3)
                            mv[kk] = msk[MSK_OFF(0, 0, mask_query, mask_key + kk)];
                    }
                }
            }
            mask_full[qb][ii] = MASK_TO_FLOAT16(mv) * iscale;
        }
    }
}
#endif

#define SDPA_OCL_MASK_INL 1
