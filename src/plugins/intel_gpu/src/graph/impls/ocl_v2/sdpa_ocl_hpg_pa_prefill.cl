// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// xe_hpg (DG2) PA PREFILL kernel, f16 head 128 without the PA features (HPG_PA_PREFILL, set by the host:
// SDPAOclGenerator's hpg_pa_prefill()). Included by sdpa_ocl.cl in place of its own kernel, after the config and loader
// headers. The loop is sdpa_ocl.cl's (online softmax over 256-key workgroup tiles, causal bound, S_max_slm / S_sum_slm
// exchange); what differs is built around the SG8 operand forms that keep DG2's DPAS fed:
//  - KQ is DPAS(Q, K): A = Q from Q_slm (staged so one ulong4 SLM block read is the operand), B = K straight from the
//    input (lane = key, 16 depth halves per lane: one 32 B read per key row, no repacking). The scores come out
//    lane = key; a per-subgroup 8 x 8 SLM transpose turns them to lane = query for the mask and softmax.
//  - SV is DPAS(V, P): A = V (lane = key pair, component = value), B = P (lane = query), so the output tile is
//    lane = query and the rescale by alpha needs no broadcast.
//  - V of the first HPG_PA_V_LEAD key blocks is loaded before the wait of the P barrier and kept HPG_PA_V_LEAD blocks
//    ahead in the S*V loop: IGC 2.41 sinks loads that are not pinned that way toward their use, which left the V
//    latency exposed (5% slower S*V on DG2).
//  - A subgroup tile entirely below the diagonal (every key <= its first query, all keys < k) skips the mask.
// Out-of-range rows are clamped to the last valid key or query, so no read leaves the subsequence, and their scores are
// masked to -INFINITY (P = 0 times a finite V). The design and its measurements: "xe_hpg PA PREFILL kernel" in
// docs/sdpa_ocl.md.

#if !(SG8 && IS_PAGED_ATTENTION && IS_PREFILL && K_HEAD_SIZE == 128 && V_HEAD_SIZE == 128 && kq_sg_tile_keys == 32 && \
      kq_sg_tile_queries == 16 && kq_sg_per_wg_keys == 8 && kq_sg_per_wg_queries == 2 && sv_sg_tile_scores == 16 && \
      sv_sg_tile_values == 16 && sv_sg_per_wg_scores == 2 && sv_sg_per_wg_values == 8)
#  error "sdpa_ocl_hpg_pa_prefill.cl: the host gate (hpg_pa_prefill()) and this kernel disagree on the shape or the tiling"
#endif
#if WITH_ATTN_MASK || defined(HAS_SCALAR_ATTN_MASK) || defined(STATIC_SCALAR_ATTN_MASK_VALUE) || defined(HAS_SINK_INPUT) || \
    HAS_TOKEN_TYPE_IDS || SLIDING_WINDOW_SIZE || WITH_SCALE || defined(KV_COMPRESSED) || MICRO_MATH || !IS_CAUSAL || \
    INPUT0_IS_BF16
#  error "sdpa_ocl_hpg_pa_prefill.cl: an op the host gate (hpg_pa_prefill()) should have sent to the generic kernel"
#endif
#if !defined(HPG_PA_V_LEAD) || HPG_PA_V_LEAD < 1 || HPG_PA_V_LEAD > sv_key_blocks
#  error "sdpa_ocl_hpg_pa_prefill.cl: HPG_PA_V_LEAD must be 1..sv_key_blocks"
#endif

// P in S_slm: the 16-key block key_block of the workgroup tile for the 8 queries of query_block, one SG8 block-read
// operand (dword j of lane l = keys 2j, 2j + 1 of query l) at P_SLM_OFF, 64 dwords.
#define P_SLM_OFF(key_block, query_block) \
    (((key_block) * kq_wg_tile_queries + (query_block) * SUBGROUP_SIZE) * DPAS_K / 2)

// Scores of one subgroup tile, lane = query, component = key. A subgroup whose first key is at/past causal_k leaves
// them 0 (masked by the caller).
SDPA_OCL_INLINE void FUNC(hpg_kq)(__private float8 (*scores)[kq_query_blocks], const __global half *K, const uint ldk,
                                  const int key_base, const int causal_k, const __local uint *q_slm,
                                  __local uint *transpose, const uint lane) {
    if (key_base >= causal_k)
        return;
    #pragma unroll
    for (int db = 0; db < DKS_ACTIVE; ++db) {
        int8 k_b[kq_key_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb) {
            const int row = min(key_base + mb * DPAS_ROWS + (int)lane, causal_k - 1);
            k_b[mb] = as_int8(vload8(0, (const __global uint *)(K + (size_t)row * ldk + db * DPAS_K)));
        }
        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            const int8 q_a = as_int8(intel_sub_group_block_read_ul4((const __local ulong *)(q_slm + Q_SLM_OFF(db, qb))));
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb)
                scores[mb][qb] = DPAS_MAD_K16(q_a, k_b[mb], scores[mb][qb]);
        }
    }
    // lane = key -> lane = query, one 8 x 8 block at a time through this subgroup's 64 dwords.
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb) {
            intel_sub_group_block_write8(transpose, as_uint8(scores[mb][qb]));
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            scores[mb][qb] = as_float8(vload8(0, transpose + lane * 8));
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

// The V rows of one 16-key block for this lane: keys key0 + 2 lane and key0 + 2 lane + 1 (clamped to causal_k - 1),
// value columns col..col + 15.
SDPA_OCL_INLINE void FUNC(hpg_v_rows)(const __global half *V, const uint ldv, const int key0, const int causal_k,
                                      const uint col, const uint lane, __private uint8 *even, __private uint8 *odd) {
    const int even_row = min(key0 + (int)lane * 2, causal_k - 1);
    const int odd_row = min(even_row + 1, causal_k - 1);
    *even = vload8(0, (const __global uint *)(V + (size_t)even_row * ldv + col));
    *odd = vload8(0, (const __global uint *)(V + (size_t)odd_row * ldv + col));
}

// acc[qb][vb] += V^T P over the sv_key_blocks key blocks of the tile, with V HPG_PA_V_LEAD blocks ahead in a ring
// (the caller fills slots 0..HPG_PA_V_LEAD - 1 with blocks 0..HPG_PA_V_LEAD - 1).
SDPA_OCL_INLINE void FUNC(hpg_sv)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                  const __global half *V, const uint ldv, const int k0, const int causal_k,
                                  const uint col, const uint lane, __private uint8 *ring_even, __private uint8 *ring_odd) {
    int8 p_cur[sv_score_blocks];
    #pragma unroll
    for (int qb = 0; qb < sv_score_blocks; ++qb)
        p_cur[qb] = as_int8(intel_sub_group_block_read8(p_slm + P_SLM_OFF(0, qb)));
    #pragma unroll
    for (int kb = 0; kb < sv_key_blocks; ++kb) {
        int8 p_next[sv_score_blocks];
        if (kb + 1 < sv_key_blocks) {
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                p_next[qb] = as_int8(intel_sub_group_block_read8(p_slm + P_SLM_OFF(kb + 1, qb)));
        }
        const int slot = kb % HPG_PA_V_LEAD;
        const uint8 even = ring_even[slot];
        const uint8 odd = ring_odd[slot];
        if (kb + HPG_PA_V_LEAD < sv_key_blocks)
            FUNC_CALL(hpg_v_rows)(V, ldv, k0 + (kb + HPG_PA_V_LEAD) * DPAS_K, causal_k, col, lane, &ring_even[slot],
                                  &ring_odd[slot]);
        #pragma unroll
        for (int vb = 0; vb < sv_value_blocks; ++vb) {
            // A operand: component = value, lane = (key 2 lane, key 2 lane + 1). Slice the 8 values first, then
            // interleave the two keys halfword by halfword.
            const int w = vb * 4;
            const ushort8 e = as_ushort8((uint4)(even[w], even[w + 1], even[w + 2], even[w + 3]));
            const ushort8 o = as_ushort8((uint4)(odd[w], odd[w + 1], odd[w + 2], odd[w + 3]));
            const int8 v_a = as_int8(shuffle2(e, o, (ushort16)(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)));
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                acc[qb][vb] = DPAS_MAD_K16(v_a, p_cur[qb], acc[qb][vb]);
        }
        if (kb + 1 < sv_key_blocks) {
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                p_cur[qb] = p_next[qb];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)))
KERNEL(sdpa_ocl)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
        global OUTPUT_TYPE *A,
        const __global INPUT3_TYPE* subsequence_begins,
        const __global int* blocked_indexes_start_and_gws_mapping)
{
    const uint query_block_idx = get_group_id(0) << 1;
    const uint block_start_pos = blocked_indexes_start_and_gws_mapping[query_block_idx];
    const uint gws_mapping = blocked_indexes_start_and_gws_mapping[query_block_idx + 1];
    const uint subsequence_begin = subsequence_begins[gws_mapping];
    const uint subsequence_end = subsequence_begins[gws_mapping + 1];
    const int q = subsequence_end - subsequence_begin;
    const int k = q;
    const int wg_j0 = block_start_pos - subsequence_begin;

    const uint lane = get_sub_group_local_id();
    const uint sg_ij = get_local_id(1);
    const uint b0 = get_group_id(1);
    const uint b0_kv = b0 / KV_GROUP_SIZE;

    const uint sg_i_kq = sg_ij % kq_sg_per_wg_keys;
    const uint sg_j_kq = sg_ij / kq_sg_per_wg_keys;
    const int sg_i0_kq = sg_i_kq * kq_sg_tile_keys;
    const int sg_j0_kq = sg_j_kq * kq_sg_tile_queries;
    // The S*V subgroup of sg_ij owns the same 16 queries as its KQ subgroup (sg_i_sv == sg_j_kq), so alpha[] applies to
    // the accumulator rows as they are.
    const uint sg_i_sv = sg_ij / sv_sg_per_wg_values;
    const uint sg_j_sv = sg_ij % sv_sg_per_wg_values;
    const int sg_i0_sv = sg_i_sv * sv_sg_tile_scores;
    const uint sg_j0_sv = sg_j_sv * sv_sg_tile_values;

#ifdef STATIC_SCALE_VALUE
    #if INVERT_SCALE
    float scale = convert_float(STATIC_SCALE_VALUE_INV);
    #else
    float scale = convert_float(STATIC_SCALE_VALUE);
    #endif
#else
    float scale = native_recip(sqrt(convert_float(K_HEAD_SIZE)));
#endif
    scale *= 1.4426950408889634f;  // log2(e): the softmax runs on exp2

    const uint ldq = K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint ldk = K_HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = V_HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    const uint lda = V_HEAD_SIZE * HEADS_NUM;
    Q += (size_t)subsequence_begin * ldq + b0 * K_HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    K += (size_t)subsequence_begin * ldk + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    V += (size_t)subsequence_begin * ldv + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    A += (size_t)subsequence_begin * lda + b0 * V_HEAD_SIZE;

    local uint Q_slm[DKS_ACTIVE * q_blocks * Q_DWORDS * SUBGROUP_SIZE] __attribute__((aligned(8)));
    local uint S_slm[kq_wg_tile_keys * kq_wg_tile_queries / 2];
    // Per-subgroup scratch of the KQ transpose, apart from S_slm: P may still be read by another subgroup when the
    // next tile's KQ starts.
    local uint transpose_slm[sg_per_wg * SUBGROUP_SIZE * SUBGROUP_SIZE];
    local float S_sum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
        S_max_slm[qi] = -INFINITY;

    // Q staging, one (query block, depth-tile pair) per subgroup step: lane l reads 32 halves of query row
    // wg_j0 + 8 q_block + l (clamped to q - 1), and adjacent lanes trade halves so that the ulong4 block read in hpg_kq()
    // gives component = query, lane = depth pair.
    for (int tile = sg_ij; tile < q_blocks * (DKS_ACTIVE / 2); tile += sg_per_wg) {
        const int q_block = tile % q_blocks;
        const int db = (tile / q_blocks) * 2;
        const int query = min(wg_j0 + q_block * SUBGROUP_SIZE + (int)lane, q - 1);
        const uint16 qq = vload16(0, (const global uint *)(Q + (size_t)query * ldq + db * DPAS_K));
        #pragma unroll
        for (int half_db = 0; half_db < 2; ++half_db) {
            const uint8 src = half_db ? qq.hi : qq.lo;
            uint8 out;
            #pragma unroll
            for (uint pair = 0; pair < 4; ++pair) {
                const uint a = src[pair];
                const uint b = src[pair + 4];
                const uint xa = sub_group_shuffle_xor(a, 1);
                const uint xb = sub_group_shuffle_xor(b, 1);
                out[pair * 2] = (lane & 1) ? xb : a;
                out[pair * 2 + 1] = (lane & 1) ? b : xa;
            }
            vstore8(out, 0, Q_slm + Q_SLM_OFF(db + half_db, q_block) + lane * 8);
        }
    }

    float S_max_tile[kq_query_blocks];
    float S_sum_tile[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        S_max_tile[qb] = -INFINITY;
        S_sum_tile[qb] = 0.0f;
    }

    float8 A_tile[sv_score_blocks][sv_value_blocks];
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r)
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            A_tile[r][cd] = (float8)0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Causal bound: no query of the workgroup needs a key past its last query.
    const int causal_k = min(k, wg_j0 + kq_wg_tile_queries);

    for (int k0 = 0; k0 < causal_k; k0 += kq_wg_tile_keys) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == 0);
        const bool last = (k0 + kq_wg_tile_keys >= causal_k);

        float8 S_tile[kq_key_blocks][kq_query_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

        // A subgroup whose 16 queries are all past q contributes nothing that is stored.
        if (wg_j0 + sg_j0_kq < q)
            FUNC_CALL(hpg_kq)(S_tile, (const global half *)K, ldk, key_base, causal_k,
                              Q_slm + Q_SLM_OFF(0, sg_j0_kq / SUBGROUP_SIZE),
                              transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE, lane);

        float k_mask[kq_sg_tile_keys / SUBGROUP_SIZE];
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            k_mask[ii] = (key_base + ii * SUBGROUP_SIZE + (int)lane < causal_k) ? 0.0f : -INFINITY;

        float alpha[kq_query_blocks];
        // Uniform per subgroup: a tile wholly below the diagonal (every key <= the subgroup's first query, and all keys
        // < k) has nothing to mask.
        if (key_base + kq_sg_tile_keys - 1 <= wg_j0 + sg_j0_kq && key_base + kq_sg_tile_keys <= k) {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                float lmax = -INFINITY;
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb)
                    #pragma unroll
                    for (int mm = 0; mm < DPAS_ROWS; ++mm)
                        lmax = fmax(lmax, S_tile[mb][qb][mm]);
                __builtin_IB_atomic_max_local_f32(&S_max_slm[sg_j0_kq + qb * SUBGROUP_SIZE + lane], lmax);
            }
        } else {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + (int)lane;
                // Every key of the tile <= the block's first query: only the k_mask term is left.
                const bool causal_block_clear = key_base + kq_sg_tile_keys - 1 <= wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE;
                float lmax = -INFINITY;
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb) {
                    #pragma unroll
                    for (int mm = 0; mm < DPAS_ROWS; ++mm) {
                        const int key_rel = mb * DPAS_ROWS + mm;
                        float s = S_tile[mb][qb][mm] +
                                  sub_group_broadcast(k_mask[key_rel / SUBGROUP_SIZE], key_rel % SUBGROUP_SIZE);
                        if (!causal_block_clear && key_base + key_rel > query)
                            s = -INFINITY;
                        S_tile[mb][qb][mm] = s;
                        lmax = fmax(lmax, s);
                    }
                }
                __builtin_IB_atomic_max_local_f32(&S_max_slm[sg_j0_kq + qb * SUBGROUP_SIZE + lane], lmax);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            const float m_new = S_max_slm[sg_j0_kq + qb * SUBGROUP_SIZE + lane];
            // -INFINITY when every key of the query is masked so far: keep the state, P = 0.
            const bool ok = isfinite(m_new);
            const float m_log2 = ok ? m_new * scale : 0.0f;
            const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;
            S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
            alpha[qb] = a;

            float lsum = 0.0f;
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                const float8 p = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                lsum += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
                // Keys mb * 8.. of the subgroup tile: dwords 0..3 (mb even) or 4..7 (mb odd) of the operand rows of
                // 16-key block 2 sg_i_kq + mb / 2.
                intel_sub_group_block_write4(
                    S_slm + P_SLM_OFF(sg_i_kq * (kq_sg_tile_keys / DPAS_K) + mb / 2, sg_j_kq * kq_query_blocks + qb) +
                        (mb % 2) * (SUBGROUP_SIZE * 4),
                    as_uint4(convert_half8(p)));
            }
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
        }

        if (last) {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_sum_slm[(sg_j0_kq + qb * SUBGROUP_SIZE + lane) * kq_sg_per_wg_keys + sg_i_kq] = S_sum_tile[qb];
        }

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        if (!first) {
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r)
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= alpha[r];
        }

        // V does not depend on P, so the first blocks are loaded before the wait (see the header).
        uint8 v_even[HPG_PA_V_LEAD];
        uint8 v_odd[HPG_PA_V_LEAD];
        #pragma unroll
        for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
            FUNC_CALL(hpg_v_rows)((const global half *)V, ldv, k0 + pb * DPAS_K, causal_k, sg_j0_sv, lane, &v_even[pb],
                                  &v_odd[pb]);
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        FUNC_CALL(hpg_sv)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), (const global half *)V, ldv, k0,
                          causal_k, sg_j0_sv, lane, v_even, v_odd);
    }

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        const int query = sg_i0_sv + r * SUBGROUP_SIZE + (int)lane;
        float l = 0.0f;
        #pragma unroll
        for (int p = 0; p < kq_sg_per_wg_keys; ++p)
            l += S_sum_slm[query * kq_sg_per_wg_keys + p];
        const float inv_l = l > 0.0f ? native_recip(l) : 0.0f;
        const int row = wg_j0 + query;
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd) {
            const half8 o = convert_half8(A_tile[r][cd] * inv_l);
            if (row < q)
                vstore4(as_uint4(o), 0, (global uint *)(A + (size_t)row * lda + sg_j0_sv + cd * SUBGROUP_SIZE));
        }
    }
}

#undef P_SLM_OFF

#define SDPA_OCL_HPG_PA_PREFILL_INL 1
