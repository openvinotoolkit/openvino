// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// xe_hpg (DG2) PA PREFILL kernel, f16 head 64 / 96 / 128 / 256 / 512 (k == v), with or without a sliding window, a sink and
// token_type_ids, which also serves PA MIXED on an f16 cache (without token_type_ids) and plain SDPA without causality,
// with or without an attention mask (HPG_PA_PREFILL, set by the host: SDPAOclGenerator's hpg_pa_prefill_ok() /
// hpg_plain_prefill_ok()). Included by sdpa_ocl.cl in place of its own kernel, after the config and loader headers. The loop is sdpa_ocl.cl's (online softmax over workgroup key tiles,
// causal and window bounds, S_max_slm / S_sum_slm exchange); what differs is built around the SG8 operand forms that keep
// DG2's DPAS fed:
//  - KQ is DPAS(Q, K): A = Q from Q_slm (staged so one ulong4 SLM block read is the operand), B = K straight from the
//    input (lane = key, 16 depth halves per lane: one 32 B read per key row, no repacking). The scores come out
//    lane = key; a per-subgroup 8 x 8 SLM transpose turns them to lane = query for the mask and softmax.
//  - SV is DPAS(V, P): A = V (lane = key pair, component = value), B = P (lane = query), so the output tile is
//    lane = query and the rescale by alpha needs no broadcast.
//  - V of the first HPG_PA_V_LEAD key blocks is loaded before the wait of the P barrier and kept HPG_PA_V_LEAD blocks
//    ahead in the S*V loop: IGC 2.41 sinks loads that are not pinned that way toward their use, which left the V
//    latency exposed (5% slower S*V on DG2).
//  - A subgroup tile entirely below the diagonal and inside the window (every key <= its first query and in the window
//    of its last query, all keys < k) skips the mask.
// The sink and the image groups of token_type_ids follow sdpa_ocl.cl: the sink seeds the online-softmax state (max = the
// sink logit, sum = 1 in one subgroup), and an image group is bidirectional (the causal bound reaches the end of the group
// of the workgroup's last query, the window start the start of the group it cuts, and a key in a query's own group is
// never masked).
// Plain SDPA takes its rows from the 4D layout (batch = get_group_id(2), a workgroup per 32 queries, q and k from the
// arguments) and has no causal bound: every workgroup visits all k keys, and only those past k are masked. A mask is added
// in the raw score domain, as in sdpa_ocl.cl, 8 keys of the lane's query row at a time (no tile skips it).
// Out-of-range rows are clamped to the last valid key or query, so no read leaves the subsequence, and their scores are
// masked to -INFINITY (P = 0 times a finite V). The design and its measurements: "xe_hpg PA PREFILL kernel" in
// docs/sdpa_ocl.md.
// PA MIXED (IS_PA_MIXED) reads every key, past and new, from the f16 cache pages (the cache update ran first; an f16 page
// holds the new tokens bit for bit, so Kc / Vc are unused), with the queries at key positions past_len + query (wg_p0) for
// the causal and window bounds and masks. Its K page is token-major ([token, head], the only page sdpa_ocl's MIXED reads),
// so a key's row is a page row and the K read is PREFILL's with the row address taken from the key's page; V likewise.
// The rows are clamped to the last valid key as in PREFILL, so no read reaches a page slot at/past k.

// The S*V subgroup of sg_ij owns the 16 queries of its KQ subgroup (sv_sg_per_wg_values == kq_sg_per_wg_keys, the score
// tile is the KQ query tile), and each holds 16 or 32 value columns of the head (hpg_pa_prefill_tiling()). The head need not
// be a power of two (96) nor a multiple of 16: plain SDPA at head 72 (HPG_TAIL = 8) has a last depth tile of 8 depths (K and
// Q read 16 B there and the rest is zero) and a last value subgroup of 8 columns, which reads the 16 columns that end the
// head and stores its upper 8 (sv_col below), so no load leaves the head.
#define HPG_TAIL (K_HEAD_SIZE % DPAS_K)
#if !(SG8 && (IS_PREFILL || IS_PA_MIXED) && K_HEAD_SIZE == V_HEAD_SIZE && (HPG_TAIL == 0 || HPG_TAIL == DPAS_K / 2) && \
      kq_sg_tile_queries == 16 && sv_sg_tile_scores == 16 && sv_sg_per_wg_scores == kq_sg_per_wg_queries && \
      sv_sg_per_wg_values == kq_sg_per_wg_keys && (sv_sg_tile_values == 16 || sv_sg_tile_values == 32) && \
      sv_sg_tile_values * sv_sg_per_wg_values == V_HEAD_SIZE + HPG_TAIL)
#  error "sdpa_ocl_hpg_pa_prefill.cl: the host gate (hpg_pa_prefill_ok()) and this kernel disagree on the shape or the tiling"
#endif
// A static scalar mask (plain SDPA only, finite: hpg_plain_prefill_ok()) adds the same value to every score of a row, which
// the softmax ignores, so the kernel leaves it out.
#if defined(HAS_SCALAR_ATTN_MASK) || (defined(STATIC_SCALAR_ATTN_MASK_VALUE) && IS_PAGED_ATTENTION) || WITH_SCALE || \
    defined(KV_COMPRESSED) || MICRO_MATH || INPUT0_IS_BF16 || (IS_PAGED_ATTENTION && !IS_CAUSAL) || \
    (!IS_PAGED_ATTENTION && (IS_CAUSAL || defined(HAS_SINK_INPUT)))
#  error "sdpa_ocl_hpg_pa_prefill.cl: an op the host gate (hpg_pa_prefill()) should have sent to the generic kernel"
#endif
#if HPG_TAIL && IS_PAGED_ATTENTION
#  error "sdpa_ocl_hpg_pa_prefill.cl: a head that is not a multiple of 16 is plain SDPA's (the K page reads have no tail)"
#endif
#if IS_PA_MIXED && (IS_PA_KV_COMPRESSED || PA_CUR_KV_F16 || HAS_QQ_BIAS || HAS_TOKEN_TYPE_IDS || PAGED_ATTENTION_BLOCK_SIZE != DPAS_K)
#  error "sdpa_ocl_hpg_pa_prefill.cl: MIXED reads f16 token-major pages of 16 tokens, without qq_bias or token_type_ids"
#endif
#if !defined(HPG_PA_V_LEAD) || HPG_PA_V_LEAD < 1 || HPG_PA_V_LEAD > sv_key_blocks
#  error "sdpa_ocl_hpg_pa_prefill.cl: HPG_PA_V_LEAD must be 1..sv_key_blocks"
#endif

// The mask keeps the keys (query - HPG_PA_WINDOW, query]. SDPA_OCL_NEG_SG8=9 makes it one 16-key block short (empty for a
// window of 16 or less), a negative control of the window tests (sdpa_ocl.cl has the same one).
#if SLIDING_WINDOW_SIZE && defined(NEG_SG8) && NEG_SG8 == 9
#  define HPG_PA_WINDOW max(SLIDING_WINDOW_SIZE - DPAS_K, 0)
#else
#  define HPG_PA_WINDOW SLIDING_WINDOW_SIZE
#endif

// One lane's slice of a V row: the sv_sg_tile_values value columns of its S*V subgroup, as dwords.
#if sv_sg_tile_values == 16
#  define HPG_V_T       uint8
#  define HPG_V_LOAD(p) vload8(0, p)
#else
#  define HPG_V_T       uint16
#  define HPG_V_LOAD(p) vload16(0, p)
#endif

// MIXED finds K and V through the subsequence's block table (bumped to its first entry); K / V then point at the KV
// head's slice of page 0.
#if IS_PA_MIXED
#  define HPG_PAGES_ARG const __global INPUT3_TYPE *block_indices,
#  define HPG_PAGES     block_indices,
#else
#  define HPG_PAGES_ARG
#  define HPG_PAGES
#endif

// P in S_slm: the 16-key block key_block of the workgroup tile for the 8 queries of query_block, one SG8 block-read
// operand (dword j of lane l = keys 2j, 2j + 1 of query l) at P_SLM_OFF, 64 dwords.
#define P_SLM_OFF(key_block, query_block) \
    (((key_block) * kq_wg_tile_queries + (query_block) * SUBGROUP_SIZE) * DPAS_K / 2)

// Scores of one subgroup tile, lane = query, component = key. A subgroup whose first key is at/past causal_k leaves
// them 0 (masked by the caller).
SDPA_OCL_INLINE void FUNC(hpg_kq)(__private float8 (*scores)[kq_query_blocks], HPG_PAGES_ARG const __global half *K,
                                  const uint ldk, const int key_base, const int causal_k, const __local uint *q_slm,
                                  __local uint *transpose, const uint lane) {
    if (key_base >= causal_k)
        return;
#if IS_PA_MIXED
    // key_base is a multiple of 16, so each 16 keys of the tile are one page: page pg holds keys page_key0[pg].. A page
    // past causal_k takes the page of the last key, which is where the clamped rows of its two row blocks land.
    const __global half *k_page[kq_key_blocks / 2];
    int page_key0[kq_key_blocks / 2];
    #pragma unroll
    for (int pg = 0; pg < kq_key_blocks / 2; ++pg) {
        page_key0[pg] = min(key_base + pg * DPAS_K, (causal_k - 1) & ~(PAGED_ATTENTION_BLOCK_SIZE - 1));
        k_page[pg] = K + PA_K_PAGE_OFF(block_indices[page_key0[pg] / PAGED_ATTENTION_BLOCK_SIZE], 0);
    }
#endif
#ifdef HPG_PA_KQ_UNROLL
    // Unrolled whole at head 256, IGC 2.41 would load K of every depth tile ahead (the host's HPG_PA_KQ_UNROLL).
    __attribute__((opencl_unroll_hint(HPG_PA_KQ_UNROLL)))
#else
    #pragma unroll
#endif
    for (int db = 0; db < DKS_ACTIVE; ++db) {
        int8 k_b[kq_key_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb) {
            const int row = min(key_base + mb * DPAS_ROWS + (int)lane, causal_k - 1);
    #if IS_PA_MIXED
            // The key's row of its token-major page.
        #if defined(NEG_SG8) && NEG_SG8 == 14
            // Negative control: the two 8-key halves of each page swapped (smoke_paged_attention_mixed_tm must fail the
            // cases it can see).
            const int page_row = (row - page_key0[mb / 2]) ^ DPAS_ROWS;
        #else
            const int page_row = row - page_key0[mb / 2];
        #endif
            k_b[mb] = as_int8(vload8(0, (const __global uint *)(k_page[mb / 2] + (size_t)page_row * ldk + db * DPAS_K)));
    #else
        #if HPG_TAIL
            // The last tile's upper 8 depths are past the head (the next head's, or past the buffer): zeros.
            if (db == DKS_ACTIVE - 1)
                k_b[mb] = as_int8((uint8)(vload4(0, (const __global uint *)(K + (size_t)row * ldk + db * DPAS_K)), (uint4)0));
            else
        #endif
            k_b[mb] = as_int8(vload8(0, (const __global uint *)(K + (size_t)row * ldk + db * DPAS_K)));
    #endif
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
// value columns col..col + sv_sg_tile_values - 1. MIXED: rows of the block's page (key0 is a multiple of 16), or of the
// last key's page when the block is past causal_k, so the clamped rows stay in the page.
SDPA_OCL_INLINE void FUNC(hpg_v_rows)(HPG_PAGES_ARG const __global half *V, const uint ldv, const int key0,
                                      const int causal_k, const uint col, const uint lane, __private HPG_V_T *even,
                                      __private HPG_V_T *odd) {
    const int even_row = min(key0 + (int)lane * 2, causal_k - 1);
    const int odd_row = min(even_row + 1, causal_k - 1);
#if IS_PA_MIXED
    const int page_key0 = min(key0, (causal_k - 1) & ~(PAGED_ATTENTION_BLOCK_SIZE - 1));
    int page = page_key0 / PAGED_ATTENTION_BLOCK_SIZE;
    #if defined(NEG_SG8) && NEG_SG8 == 16
    // Negative control: V of adjacent pages swapped (a MIXED case whose keys span two pages must fail).
    page = min(page ^ 1, (causal_k - 1) / PAGED_ATTENTION_BLOCK_SIZE);
    #endif
    V += PA_V_PAGE_OFF(block_indices[page], 0);
    *even = HPG_V_LOAD((const __global uint *)(V + (size_t)(even_row - page_key0) * ldv + col));
    *odd = HPG_V_LOAD((const __global uint *)(V + (size_t)(odd_row - page_key0) * ldv + col));
#else
    *even = HPG_V_LOAD((const __global uint *)(V + (size_t)even_row * ldv + col));
    *odd = HPG_V_LOAD((const __global uint *)(V + (size_t)odd_row * ldv + col));
#endif
}

#if WITH_ATTN_MASK
// The mask of keys key0..key0 + 7 for this lane's query: row 0 of a per-key mask and key 0 of a per-query one (the selects
// fold when MSK_D2 / MSK_D3 are literals), 0 past the mask's rows or keys (such a row is not stored, such a key is masked
// by k_mask). dword_ok: a row of whole dwords, read with one 16 B load (key0 is a multiple of 8).
SDPA_OCL_INLINE float8 FUNC(hpg_mask8)(OPTIONAL_SHAPE_INFO_ARG const __global half *msk, const int query, int key0,
                                       const bool dword_ok) {
    #if defined(NEG_SG8) && NEG_SG8 == 11
    const int row = 0;  // negative control: the first query's row (smoke_sdpa_hpg_rows must fail its full-mask cases)
    #else
    const int row = (MSK_D2 == 1) ? 0 : query;
    #endif
    #if defined(NEG_SG8) && NEG_SG8 == 13
    key0 ^= DPAS_ROWS;  // negative control: the two 8-key halves of each 16-key block swapped (every mask case must fail)
    #endif
    half8 mv = (half8)0.0f;
    if (row < MSK_D2) {
        const __global half *p = msk + MSK_OFF(0, 0, row, 0);
        if (MSK_D3 == 1) {
            mv = (half8)p[0];
        } else if (key0 + DPAS_ROWS <= MSK_D3) {
            mv = dword_ok ? as_half8(vload4(0, (const __global uint *)(p + key0))) : vload8(0, p + key0);
        } else {
            #pragma unroll
            for (int kk = 0; kk < DPAS_ROWS; ++kk) {
                if (key0 + kk < MSK_D3)
                    mv[kk] = p[key0 + kk];
            }
        }
    }
    return convert_float8(mv);
}
#endif

// acc[qb][vb] += V^T P over the sv_key_blocks key blocks of the tile, with V HPG_PA_V_LEAD blocks ahead in a ring
// (the caller fills slots 0..HPG_PA_V_LEAD - 1 with blocks 0..HPG_PA_V_LEAD - 1).
SDPA_OCL_INLINE void FUNC(hpg_sv)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                  HPG_PAGES_ARG const __global half *V, const uint ldv, const int k0, const int causal_k,
                                  const uint col, const uint lane, __private HPG_V_T *ring_even, __private HPG_V_T *ring_odd) {
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
        const HPG_V_T even = ring_even[slot];
        const HPG_V_T odd = ring_odd[slot];
        if (kb + HPG_PA_V_LEAD < sv_key_blocks)
            FUNC_CALL(hpg_v_rows)(HPG_PAGES V, ldv, k0 + (kb + HPG_PA_V_LEAD) * DPAS_K, causal_k, col, lane,
                                  &ring_even[slot], &ring_odd[slot]);
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
#if IS_PA_MIXED
        const global QRY_DATA_T *Kc,  // the new tokens' K / V, unused: an f16 page holds the same values
        const global QRY_DATA_T *Vc,
#endif
        global OUTPUT_TYPE *A,
#if IS_PAGED_ATTENTION
        const __global INPUT3_TYPE* subsequence_begins,
#endif
#if IS_PA_MIXED
        const __global INPUT3_TYPE* past_lens,
        const __global INPUT3_TYPE* block_indices,
        const __global INPUT3_TYPE* block_indices_begins,
#endif
#if WITH_ATTN_MASK
        const global half *msk,
#endif
#ifdef HAS_SINK_INPUT
        const global SINK_DATA_T *sink_ptr,
#endif
#if HAS_TOKEN_TYPE_IDS
        const __global int* token_type_ids,
        const int token_type_ids_count,
#endif
#if IS_PAGED_ATTENTION
        const __global int* blocked_indexes_start_and_gws_mapping)
#else
        const int d, const int k, const int q)
#endif
{
#if IS_PAGED_ATTENTION
    const uint query_block_idx = get_group_id(0) << 1;
    const uint block_start_pos = blocked_indexes_start_and_gws_mapping[query_block_idx];
    const uint gws_mapping = blocked_indexes_start_and_gws_mapping[query_block_idx + 1];
    const uint subsequence_begin = subsequence_begins[gws_mapping];
    const uint subsequence_end = subsequence_begins[gws_mapping + 1];
    const int q = subsequence_end - subsequence_begin;
    const int wg_j0 = block_start_pos - subsequence_begin;
    #if IS_PA_MIXED
    const int past_len = past_lens[gws_mapping];
    const int k = past_len + q;
        #if defined(NEG_SG8) && NEG_SG8 == 15
    const int wg_p0 = wg_j0;  // negative control: the past ignored by the causal bound and mask (MIXED cases must fail)
        #else
    const int wg_p0 = past_len + wg_j0;
        #endif
    #else
    const int k = q;
    const int wg_p0 = wg_j0;
    #endif
#else
    const int wg_j0 = get_group_id(0) * kq_wg_tile_queries;
    const int wg_p0 = wg_j0;
    const uint b1 = get_group_id(2);  // batch
#endif
    // wg_j0 is the workgroup's first query in the subsequence (Q and output rows), wg_p0 its key position (the causal and
    // window bounds and masks): they differ by MIXED's past keys.

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
#if HPG_TAIL
    // The value columns this subgroup reads: the last one's start moves back to the last 16 of the head.
    const uint sv_col = min(sg_j0_sv, (uint)(V_HEAD_SIZE - sv_sg_tile_values));
#else
    const uint sv_col = sg_j0_sv;
#endif

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
#if defined(HAS_SINK_INPUT) || WITH_ATTN_MASK
    // The sink logit and the mask enter the raw score domain (divided by the attention scale), with sdpa_ocl.cl's inverse
    // scale.
    #if defined(STATIC_SCALE_VALUE) && INVERT_SCALE
    const float iscale = convert_float(STATIC_SCALE_VALUE);
    #elif defined(STATIC_SCALE_VALUE)
    const float iscale = convert_float(STATIC_SCALE_VALUE_INV);
    #else
    const float iscale = sqrt(convert_float(K_HEAD_SIZE));
    #endif
#endif
#ifdef HAS_SINK_INPUT
    const float sink_raw = SINK_TO_FLOAT(sink_ptr[b0]) * iscale;
#endif

#if IS_PAGED_ATTENTION
    const uint ldq = K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint lda = V_HEAD_SIZE * HEADS_NUM;
    Q += (size_t)subsequence_begin * ldq + b0 * K_HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    A += (size_t)subsequence_begin * lda + b0 * V_HEAD_SIZE;
    #if IS_PA_MIXED
    // The caches: the in-page row pitches (token-major K and V pages), this KV head's slice of page 0, and the
    // subsequence's block table.
    const uint ldk = PA_K_TOKEN_STRIDE;
    const uint ldv = PA_V_ROW_ELEMS;
    K += PA_K_PAGE_OFF(0, b0_kv);
    V += PA_V_PAGE_OFF(0, b0_kv);
    block_indices += block_indices_begins[gws_mapping];
    #else
    const uint ldk = K_HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = V_HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    K += (size_t)subsequence_begin * ldk + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    V += (size_t)subsequence_begin * ldv + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    #endif
#else
    // The host admits only layouts whose head dimension is innermost and unpadded, so every row starts at a multiple
    // of the (even) head size.
    const uint ldq = QRY_S2;
    const uint ldk = KEY_S2;
    const uint ldv = VAL_S2;
    const uint lda = DST_S2;
    // Q, K and V each read their own broadcast batch (1 or the output's), as in sdpa_ocl.cl: BROADCAST_{Q,K,V}_BATCH is
    // set where that batch is not statically the output's.
    #if BROADCAST_Q_BATCH
    const uint b1_q = (QRY_D0 == 1) ? 0 : b1;
    #else
    const uint b1_q = b1;
    #endif
    #if BROADCAST_K_BATCH
    const uint b1_k = (KEY_D0 == 1) ? 0 : b1;
    #else
    const uint b1_k = b1;
    #endif
    #if BROADCAST_V_BATCH
    const uint b1_v = (VAL_D0 == 1) ? 0 : b1;
    #else
    const uint b1_v = b1;
    #endif
    Q += QRY_OFF(b1_q, b0, 0, 0) + INPUT0_OFFSET;
    #if defined(NEG_SG8) && NEG_SG8 == 10
    // Negative control: K / V of batch 0, KV head 0 (smoke_sdpa_hpg_rows must fail its cases with two batches or KV heads).
    K += KEY_OFF(0, 0, 0, 0) + INPUT1_OFFSET;
    V += VAL_OFF(0, 0, 0, 0) + INPUT2_OFFSET;
    #else
    K += KEY_OFF(b1_k, b0_kv, 0, 0) + INPUT1_OFFSET;
    V += VAL_OFF(b1_v, b0_kv, 0, 0) + INPUT2_OFFSET;
    #endif
    A += DST_OFF(b1, b0, 0, 0, 0);
    #if WITH_ATTN_MASK
        #if defined(NEG_SG8) && NEG_SG8 == 12
    // Negative control: the mask of batch 0, head 0 (smoke_sdpa_hpg_rows must fail its cases with a mask per batch or head).
        #else
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
        #endif
    #endif
#endif
    // A dynamic padding is taken on trust by the host (dword_token_matrix_ok()); Q's is checked here, where a row 2 B off
    // costs two ushort loads per dword once per workgroup (an unaligned dword load returns wrong data without an error), and
    // so is the mask's (an odd key count per row takes two-byte loads).
#if defined(NEG_SG8) && NEG_SG8 == 4
    // Not a negative control: forces the fallbacks so the suites cover them (they must still PASS).
    const bool q_dword_ok = false;
    #if WITH_ATTN_MASK
    const bool mask_dword_ok = false;
    #endif
#else
    const bool q_dword_ok = ((ldq & 1u) == 0) && ((as_long(Q) & 3) == 0);
    #if WITH_ATTN_MASK
    const bool mask_dword_ok = (MSK_D2 == 1 || (MSK_S2 & 1) == 0) && ((as_long(msk) & 3) == 0);
    #endif
#endif
#if BIDIR_MASK
    // As in sdpa_ocl.cl: workgroup-uniform, false for an empty token_type_ids (never read then), and the buffer holds one
    // entry per new token, so it takes the subsequence bump. PREFILL keys and queries share the local coordinates.
    #if USE_BIDIR_GATE
    const bool bidir_active = token_type_ids_count > 0;
    #else
    const bool bidir_active = true;  // negative control (SDPA_OCL_BIDIR_GATE=0)
    #endif
    if (bidir_active)
        token_type_ids += subsequence_begin;
#endif

    local uint Q_slm[DKS_ACTIVE * q_blocks * Q_DWORDS * SUBGROUP_SIZE] __attribute__((aligned(8)));
    local uint S_slm[kq_wg_tile_keys * kq_wg_tile_queries / 2];
    // Per-subgroup scratch of the KQ transpose, apart from S_slm: P may still be read by another subgroup when the
    // next tile's KQ starts.
    local uint transpose_slm[sg_per_wg * SUBGROUP_SIZE * SUBGROUP_SIZE];
    local float S_sum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
#ifdef HAS_SINK_INPUT
        S_max_slm[qi] = sink_raw;  // the running max over every tile and subgroup starts at the sink
#else
        S_max_slm[qi] = -INFINITY;
#endif

    // Q staging, one (query block, depth-tile pair) per subgroup step: lane l reads 32 halves of query row
    // wg_j0 + 8 q_block + l (clamped to q - 1), and adjacent lanes trade halves so that the ulong4 block read in hpg_kq()
    // gives component = query, lane = depth pair. With an odd tile count the last pair is the last tile alone, of 16 or (head
    // 72) 8 depths, zero past them.
    for (int tile = sg_ij; tile < q_blocks * ((DKS_ACTIVE + 1) / 2); tile += sg_per_wg) {
        const int q_block = tile % q_blocks;
        const int db = (tile / q_blocks) * 2;
        const int query = min(wg_j0 + q_block * SUBGROUP_SIZE + (int)lane, q - 1);
        const global QRY_DATA_T *q_row = Q + (size_t)query * ldq + db * DPAS_K;
        uint16 qq;
#if DKS_ACTIVE % 2
        if (db == DKS_ACTIVE - 1) {
            uint8 last = (uint8)0;
    #if HPG_TAIL
            last.lo = q_dword_ok ? vload4(0, (const global uint *)q_row) : as_uint4(vload8(0, (const global ushort *)q_row));
    #else
            last = q_dword_ok ? vload8(0, (const global uint *)q_row) : as_uint8(vload16(0, (const global ushort *)q_row));
    #endif
            qq = (uint16)(last, (uint8)0);
        } else
#endif
        if (q_dword_ok) {
            qq = vload16(0, (const global uint *)q_row);
        } else {
            qq.lo = as_uint8(vload16(0, (const global ushort *)q_row));
            qq.hi = as_uint8(vload16(1, (const global ushort *)q_row));
        }
        #pragma unroll
        for (int half_db = 0; half_db < 2; ++half_db) {
#if DKS_ACTIVE % 2
            if (db + half_db == DKS_ACTIVE)
                break;
#endif
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
#ifdef HAS_SINK_INPUT
        // exp2(sink - sink) = 1, counted by one subgroup only: the epilogue sums every subgroup's S_sum_slm slot.
        S_max_tile[qb] = sink_raw * scale;
    #if defined(NEG_SG8) && NEG_SG8 == 7
        S_sum_tile[qb] = 1.0f;  // negative control: every subgroup counts the sink
    #else
        S_sum_tile[qb] = sg_i_kq == 0 ? 1.0f : 0.0f;
    #endif
#else
        S_max_tile[qb] = -INFINITY;
        S_sum_tile[qb] = 0.0f;
#endif
    }

    float8 A_tile[sv_score_blocks][sv_value_blocks];
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r)
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            A_tile[r][cd] = (float8)0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

#if IS_CAUSAL
    // Causal bound: no query of the workgroup needs a key past its last query (or past the end of its image group).
    int causal_k = min(k, wg_p0 + kq_wg_tile_queries);
#else
    const int causal_k = k;
#endif
#if BIDIR_MASK
    {
        const int wg_q_end = causal_k - 1;
        if (bidir_active && token_type_ids[wg_q_end] == 1)
            causal_k = max(causal_k, FUNC_CALL(bidir_scan_end)(token_type_ids, wg_q_end, q, (int)lane));
    }
#endif
#if SLIDING_WINDOW_SIZE
    // Window bound: none needs a key before its first query's window (or before the start of the image group that window
    // cuts), rounded down to a workgroup tile as in sdpa_ocl.cl.
    int window_k_begin = max(0, wg_p0 - SLIDING_WINDOW_SIZE + 1);
    #if BIDIR_MASK
    if (bidir_active && window_k_begin > 0 && token_type_ids[window_k_begin] == 1)
        window_k_begin = FUNC_CALL(bidir_scan_begin)(token_type_ids, window_k_begin, (int)lane);
    #endif
    const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;
#else
    const int window_k0_begin = 0;
#endif
#if BIDIR_MASK
    // Each query keeps its causal (and window) keys and its image group: one interval [keep_begin, keep_begin + keep_len),
    // as both hold the query, so the mask stays one unsigned compare (a separate group test spilled 0.6-1.9 KB with a
    // window on DG2, IGC 2.41). The group is scanned within the keys the loop visits, as sdpa_ocl.cl does.
    int keep_begin[kq_query_blocks];
    uint keep_len[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        const int query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + (int)lane;
    #if SLIDING_WINDOW_SIZE
        int begin = query + 1 - HPG_PA_WINDOW;
    #else
        int begin = 0;
    #endif
        int end = query + 1;
        if (bidir_active && query < q && token_type_ids[query] == 1) {
            int group_begin = query;
            while (group_begin > window_k0_begin && token_type_ids[group_begin - 1] == 1)
                --group_begin;
            int group_end = query + 1;
            while (group_end < causal_k && token_type_ids[group_end] == 1)
                ++group_end;
            begin = min(begin, group_begin);
            end = max(end, group_end);
        }
        keep_begin[qb] = begin;
        keep_len[qb] = (uint)(end - begin);
    }
#endif

    for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == window_k0_begin);
        const bool last = (k0 + kq_wg_tile_keys >= causal_k);

        float8 S_tile[kq_key_blocks][kq_query_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

        // A subgroup whose 16 queries are all past q contributes nothing that is stored.
        if (wg_j0 + sg_j0_kq < q)
            FUNC_CALL(hpg_kq)(S_tile, HPG_PAGES (const global half *)K, ldk, key_base, causal_k,
                              Q_slm + Q_SLM_OFF(0, sg_j0_kq / SUBGROUP_SIZE),
                              transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE, lane);

        float k_mask[kq_sg_tile_keys / SUBGROUP_SIZE];
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            k_mask[ii] = (key_base + ii * SUBGROUP_SIZE + (int)lane < causal_k) ? 0.0f : -INFINITY;

        float alpha[kq_query_blocks];
#if !WITH_ATTN_MASK
        // Uniform per subgroup: a tile wholly below the diagonal (every key <= the subgroup's first query, and all keys
        // < k) and inside the window of the subgroup's last query has nothing to mask.
        if (
#if IS_CAUSAL
            key_base + kq_sg_tile_keys - 1 <= wg_p0 + sg_j0_kq &&
#endif
            key_base + kq_sg_tile_keys <= k
#if SLIDING_WINDOW_SIZE
            && key_base > wg_p0 + sg_j0_kq + kq_sg_tile_queries - 1 - HPG_PA_WINDOW
#endif
        ) {
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
        } else
#endif
        {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
#if IS_CAUSAL || WITH_ATTN_MASK
                const int query = wg_p0 + sg_j0_kq + qb * SUBGROUP_SIZE + (int)lane;  // key position (plain SDPA: mask row)
#endif
#if IS_CAUSAL
                // Every key of the tile <= the block's first query (and in the window of its last): only the k_mask term
                // is left.
                const int block_query0 = wg_p0 + sg_j0_kq + qb * SUBGROUP_SIZE;
                const bool causal_block_clear = key_base + kq_sg_tile_keys - 1 <= block_query0
    #if SLIDING_WINDOW_SIZE
                                                && key_base > block_query0 + SUBGROUP_SIZE - 1 - HPG_PA_WINDOW
    #endif
                    ;
#endif
                float lmax = -INFINITY;
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb) {
#if WITH_ATTN_MASK
                    const float8 mask8 = FUNC_CALL(hpg_mask8)(OPTIONAL_SHAPE_INFO_TENSOR msk, query, key_base + mb * DPAS_ROWS,
                                                              mask_dword_ok) * iscale;
#endif
                    #pragma unroll
                    for (int mm = 0; mm < DPAS_ROWS; ++mm) {
                        const int key_rel = mb * DPAS_ROWS + mm;
                        float s = S_tile[mb][qb][mm] +
                                  sub_group_broadcast(k_mask[key_rel / SUBGROUP_SIZE], key_rel % SUBGROUP_SIZE);
#if WITH_ATTN_MASK
                        s += mask8[mm];
#endif
#if IS_CAUSAL  // plain SDPA masks only the keys past k (k_mask)
    #if BIDIR_MASK
                        if (!causal_block_clear && (uint)(key_base + key_rel - keep_begin[qb]) >= keep_len[qb])
    #elif SLIDING_WINDOW_SIZE
                        // 0 <= query - key < window as one unsigned compare: the two signed ones (key > query, key <=
                        // query - window) spilled 2.5 KB on DG2 (IGC 2.41).
                        if (!causal_block_clear && (uint)(query - key_base - key_rel) >= (uint)HPG_PA_WINDOW)
    #else
                        if (!causal_block_clear && key_base + key_rel > query)
    #endif
                            s = -INFINITY;
#endif
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
        HPG_V_T v_even[HPG_PA_V_LEAD];
        HPG_V_T v_odd[HPG_PA_V_LEAD];
        #pragma unroll
        for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
            FUNC_CALL(hpg_v_rows)(HPG_PAGES (const global half *)V, ldv, k0 + pb * DPAS_K, causal_k, sv_col, lane,
                                  &v_even[pb], &v_odd[pb]);
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        FUNC_CALL(hpg_sv)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), HPG_PAGES (const global half *)V, ldv,
                          k0, causal_k, sv_col, lane, v_even, v_odd);
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
#if HPG_TAIL && defined(NEG_SG8) && NEG_SG8 == 18
            // Negative control: the last value subgroup stores its block unshifted (smoke_sdpa_hpg_rows must fail every head-72
            // case).
            if (row < q && sg_j0_sv + cd * SUBGROUP_SIZE < V_HEAD_SIZE)
                vstore4(as_uint4(o), 0, (global uint *)(A + (size_t)row * lda + sg_j0_sv + cd * SUBGROUP_SIZE));
#elif HPG_TAIL
            if (row < q && sv_col + cd * SUBGROUP_SIZE >= sg_j0_sv)  // the last subgroup's lower block belongs to the one before
                vstore4(as_uint4(o), 0, (global uint *)(A + (size_t)row * lda + sv_col + cd * SUBGROUP_SIZE));
#else
            if (row < q)
                vstore4(as_uint4(o), 0, (global uint *)(A + (size_t)row * lda + sg_j0_sv + cd * SUBGROUP_SIZE));
#endif
        }
    }
}

#undef HPG_TAIL
#undef P_SLM_OFF
#undef HPG_PA_WINDOW
#undef HPG_V_T
#undef HPG_V_LOAD
#undef HPG_PAGES_ARG
#undef HPG_PAGES

#define SDPA_OCL_HPG_PA_PREFILL_INL 1
