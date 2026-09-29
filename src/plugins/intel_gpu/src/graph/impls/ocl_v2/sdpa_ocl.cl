#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io               : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                         : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                   : enable

#include "include/batch_headers/sdpa_utils.cl"
#if INPUT0_IS_BF16
#include "include/batch_headers/bf16_utils.cl"
#endif

float __builtin_IB_atomic_max_local_f32(__local float *, float);

#include "sdpa_ocl_config.cl"
#include "sdpa_ocl_mask.cl"
#include "sdpa_ocl_qk_load.cl"
#include "sdpa_ocl_v_load.cl"
// kernels_db_gen replaces an include it cannot resolve with nothing, silently; each header ends
// with a sentinel so a missing one fails the build instead.
#if !defined(SDPA_OCL_CONFIG_INL) || !defined(SDPA_OCL_MASK_INL) || \
    !defined(SDPA_OCL_QK_LOAD_INL) || !defined(SDPA_OCL_V_LOAD_INL)
#  error "sdpa_ocl.cl: an sdpa_ocl_*.cl header was not inlined"
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)))
KERNEL(sdpa_ocl)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
#if IS_PA_MIXED
        const global QRY_DATA_T *Kc,
        const global QRY_DATA_T *Vc,
#endif
    global OUTPUT_TYPE *A,
#if IS_PAGED_ATTENTION
        const __global INPUT3_TYPE* subsequence_begins,
    #if !IS_PREFILL
        const __global INPUT3_TYPE* past_lens,
        const __global INPUT3_TYPE* block_indices,
        const __global INPUT3_TYPE* block_indices_begins,
    #endif
#endif
#if WITH_ATTN_MASK || defined(HAS_SCALAR_ATTN_MASK)
        const global half *msk,
#endif
#if WITH_SCALE
        global SCALE_DATA_T *scale_ptr,
#endif
#ifdef HAS_SINK_INPUT
        const global SINK_DATA_T *sink_ptr,
#endif
#if HAS_QQ_BIAS && IS_PA_MIXED
        const global QQ_BIAS_DATA_T *qq_bias,
        const global QQ_BIAS_BEGINS_DATA_T *qq_bias_begins,
#endif
#if HAS_TOKEN_TYPE_IDS
        const __global int* token_type_ids,
        const int token_type_ids_count,
#endif
#if IS_PAGED_ATTENTION
        const __global int* blocked_indexes_start_and_gws_mapping
#else
        const int d,
        const int k,
        const int q
#endif
    #ifdef KV_COMPRESSED
        , const global KEY_ATTR_SCALES_DATA_T *K_scales
        , const global KEY_ATTR_ZP_DATA_T *K_zp
        , const global VAL_ATTR_SCALES_DATA_T *V_scales
        , const global VAL_ATTR_ZP_DATA_T *V_zp
    #endif
        )
{
#if IS_PAGED_ATTENTION
    const uint query_block_idx = get_group_id(0) << 1;
    const uint block_start_pos = blocked_indexes_start_and_gws_mapping[query_block_idx];
    const uint gws_mapping = blocked_indexes_start_and_gws_mapping[query_block_idx + 1];
    const uint subsequence_begin = subsequence_begins[gws_mapping];
    const uint subsequence_end = subsequence_begins[gws_mapping + 1];
    const uint subsequence_query_block_idx = block_start_pos - subsequence_begin;
    int q = subsequence_end - subsequence_begin;
    #if HAS_QQ_BIAS && IS_PA_MIXED
        // Speculative tree mask over this subsequence's NEW keys: qq_bias holds spec_num x spec_num
        // entries per subsequence, starting at qq_bias_begins[gws_mapping].
        const uint qq_bias_num = qq_bias_begins[gws_mapping + 1] - qq_bias_begins[gws_mapping];
        const uint cumulated_spec_num = qq_bias_begins[gws_mapping];
        const uint spec_num = (uint)native_sqrt((float)qq_bias_num);
    #endif
    #if IS_PREFILL
        const int past_len = 0;
        const int k = q;
    #else
        const int past_len = past_lens[gws_mapping];
        const int k = q + past_len;
    #endif
    const int d = K_HEAD_SIZE;
#endif

    // Head dim of the V / output side: bounds everything from the S*V B operand to the output
    // store, while `d` bounds Q staging and the KQ contraction. They differ only when k_head_size
    // != v_head_size.
    const int dv = V_HEAD_SIZE;

    const size_t lane  = get_sub_group_local_id();
    const size_t sg_ij = get_local_id(1);
#if IS_PAGED_ATTENTION
    // The query block comes from blocked_indexes_start_and_gws_mapping, relative to this
    // workgroup's subsequence, not from get_group_id(0).
    const size_t wg_j0 = subsequence_query_block_idx;
#else
    const size_t wg_j0 = get_group_id(0) * kq_wg_tile_queries;
#endif
    const size_t b0 = get_group_id(1);     // heads_num
    const size_t b1 = get_group_id(2);     // batch
    const size_t b0_kv = b0 / KV_GROUP_SIZE;

    const size_t sg_i_kq  = sg_ij % kq_sg_per_wg_keys;
    const size_t sg_j_kq  = sg_ij / kq_sg_per_wg_keys;
    const size_t sg_i0_kq = sg_i_kq * kq_sg_tile_keys;
    const size_t sg_j0_kq = sg_j_kq * kq_sg_tile_queries;

    const size_t sg_i_sv = sg_ij / sv_sg_per_wg_values;
    const size_t sg_j_sv = sg_ij % sv_sg_per_wg_values;
    const size_t sg_i0_sv = sg_i_sv * sv_sg_tile_scores;
    const size_t sg_j0_sv = sg_j_sv * sv_sg_tile_values;

    const float LOG2E = 1.4426950408889634f;

    #if WITH_SCALE
        /* Load scale */
        #if INVERT_SCALE
            float iscale = SCALE_TO_FLOAT(*scale_ptr);
            float scale = native_recip(iscale);
        #else
            float scale = SCALE_TO_FLOAT(*scale_ptr);
            float iscale = native_recip(scale);
        #endif
    #else
        #ifdef STATIC_SCALE_VALUE
            #if INVERT_SCALE
                float iscale = convert_float(STATIC_SCALE_VALUE);
                float scale = convert_float(STATIC_SCALE_VALUE_INV);
            #else
                float scale = convert_float(STATIC_SCALE_VALUE);
                float iscale = convert_float(STATIC_SCALE_VALUE_INV);
            #endif
        #else
            float iscale = sqrt(convert_float(K_HEAD_SIZE));
            float scale = native_recip(iscale);
        #endif
    #endif

    scale *= LOG2E;

#ifdef HAS_SINK_INPUT
    // Attention sink: an extra per-head logit with a ZERO value vector, i.e. the online-softmax
    // state seeded with one synthetic key (running max = sink, running sum = 1, A_tile = 0); the
    // key loop is untouched. It enters in the raw score domain (divided by the attention scale),
    // like the mask.
    const float sink_raw = convert_float(sink_ptr[b0]) * iscale;
#endif

    /* Row stride (in elements) of the Q/K/V/A matrices. */
#if IS_PAGED_ATTENTION
    // Paged attention tensors are 2D [total_tokens, heads * head_size] with no Y dimension, so the
    // token stride comes from the head layout (the generic *_S2 pitches do not apply).
    const uint ldq = K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint ldk = K_HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = V_HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    const uint lda = V_HEAD_SIZE * HEADS_NUM;
#else
    const uint ldq = QRY_S2;
    const uint ldk = KEY_S2;
    const uint ldv = VAL_S2;
    const uint lda = DST_S2;
#endif

#if IS_PAGED_ATTENTION
    // Tokens of all subsequences are packed into one matrix, so a batch index does not exist:
    // seek to the first token of this workgroup's subsequence and to this head's column slice.
    Q += (size_t)subsequence_begin * ldq + b0 * K_HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    A += (size_t)subsequence_begin * lda + b0 * V_HEAD_SIZE;
    #if IS_PREFILL
        K += (size_t)subsequence_begin * ldk + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
        V += (size_t)subsequence_begin * ldv + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    #else
        const uint base_block_index = block_indices_begins[gws_mapping];
        #if PA_CUR_KV_F16
            // Kc/Vc are this iteration's raw f16 K/V (what PREFILL gets as K/V), so they take the
            // same bump. They hold only the q NEW tokens, hence the (key - past_len) row index
            // below.
            Kc += (size_t)subsequence_begin * ldk + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
            Vc += (size_t)subsequence_begin * ldv + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
        #endif
    #endif
    #if BIDIR_MASK
        // Workgroup-uniform, so the branches on it keep the subgroup reductions reachable by every
        // lane. False for an empty token_type_ids, which must not be touched at all -- not even
        // bumped.
        #if USE_BIDIR_GATE
            const bool bidir_active = (token_type_ids_count > 0);
        #else
            // Negative control (SDPA_OCL_BIDIR_GATE=0): read token_type_ids unconditionally.
            const bool bidir_active = true;
        #endif

        // token_type_ids is [B_token], one entry per NEW token of the flattened batch like Q, so it
        // takes the same subsequence bump. Indices into it are then LOCAL ([0, q)), unlike `key` /
        // `causal_k` / `window_k_begin`, which are KEY coordinates (key = query_position_offset +
        // local).
        if (bidir_active)
            token_type_ids += subsequence_begin;
    #endif
#else
    Q += QRY_OFF(b1, b0, 0, 0) + INPUT0_OFFSET;
    K += KEY_OFF(b1, b0_kv, 0, 0) + INPUT1_OFFSET;
    V += VAL_OFF(b1, b0_kv, 0, 0) + INPUT2_OFFSET;
    A += DST_OFF(b1, b0, 0, 0, 0);
#endif
#if WITH_ATTN_MASK
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
#endif
#ifdef KV_COMPRESSED
    // Hoist dynamic compression-layout batch/head pitches out of the hot loops.
    const uint k_comp_base = KEY_COMP_OFF(b1, b0_kv, 0, 0);
    #if USE_2D_BLOCK_IO_V_I8
    const uint v_comp_base = VAL_COMP_OFF(b1, b0_kv, 0, 0);
    #endif
#endif

    const int QD_w = d * (int)sizeof(QRY_DATA_T), QD_h = q, QD_p = (int)ldq * (int)sizeof(QRY_DATA_T);
#if !IS_PA_MIXED
    // Input K/V surfaces. MIXED has none: its K/V are the cache pages (plus Kc/Vc).
    const int KD_w = d * (int)sizeof(KEY_DATA_T), KD_h = k, KD_p = (int)ldk * (int)sizeof(KEY_DATA_T);
    const int VD_w = dv * (int)sizeof(VAL_DATA_T), VD_h = k, VD_p = (int)ldv * (int)sizeof(VAL_DATA_T);
#endif
    const int AD_w = dv * (int)sizeof(OUTPUT_TYPE), AD_h = q, AD_p = (int)lda * (int)sizeof(OUTPUT_TYPE);

#if PA_CUR_KV_F16
    // Surfaces for the NEW-token part of the key range: Kc/Vc are f16 and q rows tall, and rows
    // past q read as zero (those keys are masked anyway).
    const int KcD_w = d * (int)sizeof(half), KcD_h = q, KcD_p = (int)ldk * (int)sizeof(half);
    const int VcD_w = dv * (int)sizeof(half), VcD_h = q, VcD_p = (int)ldv * (int)sizeof(half);
    const global half *Kc_b2d = (const global half *)Kc;
    const global half *Vc_b2d = (const global half *)Vc;
    int KcD_w_b2d = KcD_w, VcD_w_b2d = VcD_w;
    int KcD_x0 = 0, VcD_x0 = 0;
    #if BLOCK2D_KV_CUR_BASE_FIXUP
    // Same repair as BLOCK2D_KV_BASE_FIXUP below: the head offset plus a possibly dynamic feature
    // padding (a crop view of a fused QKV tensor) leave the base not provably 64B-aligned. Round
    // down, widen, shift x.
    {
        const uint kc_prem = (uint)(as_long(Kc_b2d) & 63);
        const uint vc_prem = (uint)(as_long(Vc_b2d) & 63);
        Kc_b2d = (const global half *)((const global char *)Kc_b2d - kc_prem);
        Vc_b2d = (const global half *)((const global char *)Vc_b2d - vc_prem);
        KcD_w_b2d = KcD_w + (int)kc_prem;
        VcD_w_b2d = VcD_w + (int)vc_prem;
        KcD_x0 = (int)(kc_prem / sizeof(half));
        VcD_x0 = (int)(vc_prem / sizeof(half));
    }
    #endif
    #if IS_PA_K_U4
    // The u4 Kc read addresses Kc as a DWORD surface, so its x shift is in dwords. That is exact
    // only if this head's first channel is an even number of halves from the (64B-aligned,
    // fixup-forced) origin; otherwise a dword straddles a (2c, 2c+1) channel pair. That depends on
    // runtime padding, so it is tested here; when it fails the Kc read falls back to a per-lane
    // gather (still exact, just slower).
    const int KcD_x0_dw = KcD_x0 / 2;
    const bool kc_dword_ok = ((KcD_x0 & 1) == 0);
    #endif
#endif

#if USE_2D_BLOCK_IO_KV && !IS_PA_MIXED
    // 2D block IO surface origin for the f16 K/V loads. The builtins need a 64B-aligned base, but
    // the per-head offset is a multiple of the row width (head_size * element size), not of 64 --
    // e.g. 144 B rows at head 72. BLOCK2D_KV_BASE_FIXUP repairs it the way sdpa_micro's
    // block2d_load does: round the base down to 64 B, shift x and widen the surface by the same
    // bytes. The surface still ends at this head's last element, so columns past the head dim stay
    // out of bounds (hardware zero-fill) and nothing from the next head leaks in. Hoisted: neither
    // base moves once the head is fixed.
    const global KEY_DATA_T *K_b2d = K;
    const global VAL_DATA_T *V_b2d = V;
    int KD_w_b2d = KD_w, VD_w_b2d = VD_w;
    int KD_x0 = 0, VD_x0 = 0;
    #if BLOCK2D_KV_BASE_FIXUP
    {
        const uint k_prem = (uint)(as_long(K) & 63);
        const uint v_prem = (uint)(as_long(V) & 63);
        K_b2d = (const global KEY_DATA_T *)((const global char *)K - k_prem);
        V_b2d = (const global VAL_DATA_T *)((const global char *)V - v_prem);
        KD_w_b2d = KD_w + (int)k_prem;
        VD_w_b2d = VD_w + (int)v_prem;
        KD_x0 = (int)(k_prem / sizeof(KEY_DATA_T));
        VD_x0 = (int)(v_prem / sizeof(VAL_DATA_T));
    }
    #endif
#endif

    local uint  Q_slm[DKS_ACTIVE * q_blocks * Q_DWORDS * SUBGROUP_SIZE];
    local uint  S_slm[kq_wg_tile_keys * kq_wg_tile_queries / 2];
    local float S_sum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
#ifdef HAS_SINK_INPUT
        // Seeded, not -INFINITY: this slot is written once here and thereafter only atomic-maxed in
        // the key loop, so it is the running max over every k0 tile AND every subgroup. Seeding it
        // is what counts the sink exactly once.
        S_max_slm[qi] = sink_raw;
#else
        S_max_slm[qi] = -INFINITY;
#endif

    // Cooperative Q->SLM staging: the q_blocks x DKS_ACTIVE (query block, depth tile) tiles are
    // dealt round-robin over all subgroups, so every tile is staged even when there are more tiles
    // than subgroups. The loop bound keeps q_block < q_blocks, so no guard is needed.
    for (int tile = sg_ij; tile < q_blocks * DKS_ACTIVE; tile += sg_per_wg) {
        const int q_block = tile / DKS_ACTIVE;   // 0..q_blocks-1
        const int db      = tile % DKS_ACTIVE;   // 0..DKS_ACTIVE-1
        const int query_base = wg_j0 + q_block * SUBGROUP_SIZE;
        uint8 q_pack;
#if IS_PA_K_U4
        FUNC_CALL(q_chunk_u4)(&q_pack, Q, QD_w, QD_h, QD_p, ldq, q, d, query_base, db, lane);
#else
        FUNC_CALL(q_chunk)(&q_pack, Q, QD_w, QD_h, QD_p, ldq, q, d, query_base, db, lane);
#endif
        intel_sub_group_block_write8(
            (local uint *)&Q_slm[Q_SLM_OFF(db, q_block)], q_pack);
    }

    float S_max_tile[kq_query_blocks];
    float S_sum_tile[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
#ifdef HAS_SINK_INPUT
        // The private half of the same seed. S_max_tile holds the scaled max, so it is seeded from
        // the same sink_raw * scale product the loop computes; the first rescale alpha is then
        // exactly 1.0.
        S_max_tile[qb] = sink_raw * scale;
        // The sink's exp2(sink - sink) = 1 goes to ONE subgroup only. Each subgroup writes its own
        // S_sum_slm[query * kq_sg_per_wg_keys + sg_i_kq] slot and the epilogue sums all of them, so
        // seeding every subgroup would count the sink kq_sg_per_wg_keys times.
        S_sum_tile[qb] = (sg_i_kq == 0) ? 1.0f : 0.0f;
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

#if IS_PA_MIXED
    const int query_position_offset = past_len;
#else
    const int query_position_offset = 0;
#endif

    // Causal upper bound on the key loop: a workgroup owning queries [wg_j0, wg_j0 +
    // kq_wg_tile_queries) never needs a key past its last query, so k0 tiles beyond the diagonal
    // are skipped rather than loaded, multiplied and masked away (sdpa_micro's causal_k).
#if IS_CAUSAL
    #if !IS_PAGED_ATTENTION && CAUSAL_MASK_LOWER_RIGHT
    // Bottom-right-aligned causal mask: query row `query` may attend keys [0, query + (k - q)], the
    // shift sdpa_micro/ref/opt apply for CAUSAL_MASK_LOWER_RIGHT. Without it a single new token (q
    // == 1) would see only the first kq_wg_tile_queries keys.
    const int causal_offset = max(0, k - q);
    int causal_k = min(k, causal_offset + (int)wg_j0 + kq_wg_tile_queries);
    #else
    int causal_k = min(k, query_position_offset + (int)wg_j0 + kq_wg_tile_queries);
    #endif
#else
    const int causal_k = k;
#endif

#if IS_CAUSAL && BIDIR_MASK
    // An image group is bidirectional, so a query inside one also needs its group's FUTURE keys,
    // which the causal bound cuts away: extend it to the end of the group holding the workgroup's
    // last query (groups are contiguous, so that query suffices). The scan runs in LOCAL space
    // bounded by q (groups never leave the new-token region) and relies on pa_kv_cache_update having
    // written the new tokens before this stage, as the reference does.
    {
        const int wg_q_end = min((int)wg_j0 + kq_wg_tile_queries, q) - 1;
        if (bidir_active && wg_q_end >= 0 && token_type_ids[wg_q_end] == 1) {
            const int group_end = FUNC_CALL(bidir_scan_end)(token_type_ids, wg_q_end, q, (int)lane);
            // group_end <= q, so the KEY-space result stays <= query_position_offset + q == k.
            causal_k = max(causal_k, query_position_offset + group_end);
        }
    }
#endif

    // Sliding-window lower bound, the mirror of causal_k (sdpa_micro's window_k0_begin): the mask
    // keeps (query - SLIDING_WINDOW_SIZE, query], so the first key this workgroup needs is for its
    // first query. Rounded down to a k0 tile so key_base stays tile-aligned for the block reads and
    // S_slm indexing.
#if IS_CAUSAL && SLIDING_WINDOW_SIZE
    int window_k_begin = max(0, query_position_offset + (int)wg_j0 - SLIDING_WINDOW_SIZE + 1);
    #if BIDIR_MASK
    // Mirror of the causal_k extension: a query inside an image group also needs the group's PAST
    // keys even when the window has dropped them, so move the window start back to the start of the
    // group straddling it. window_k_begin is a KEY coordinate and token_type_ids is LOCAL, hence
    // the shift; once window_begin_local <= 0 there is nothing below to extend, which also keeps
    // the index in range.
    const int window_begin_local = window_k_begin - query_position_offset;
    if (bidir_active && window_begin_local > 0 && token_type_ids[window_begin_local] == 1) {
        const int group_begin = FUNC_CALL(bidir_scan_begin)(token_type_ids, window_begin_local, (int)lane);
        window_k_begin = query_position_offset + group_begin;
    }
    #endif
    const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;
#else
    const int window_k0_begin = 0;
#endif

#if IS_CAUSAL && BIDIR_MASK
    // Per-query image-group bounds [begin, end), the only state the mask loop needs: a query's
    // allowed keys are (causal n window) u (its own group), so a key's membership never has to be
    // looked up -- one scan per workgroup instead of sdpa_micro's per-(query, key) scan. The scans
    // are clamped to the key loop's own range, which is exact (keys outside it are never visited or
    // already -INFINITY), and (0, 0) encodes "not an image token". Scanned in LOCAL space, stored
    // in KEY coordinates.
    const int bidir_scan_lo = max(0, window_k0_begin - query_position_offset);
    const int bidir_scan_hi = min(q, causal_k - query_position_offset);
    int bidir_group_begin[kq_query_blocks];
    int bidir_group_end[kq_query_blocks];
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        // Same expression the mask loop below uses for `query`, so the per-lane mapping cannot drift.
        const int query = (int)(wg_j0 + sg_j0_kq) + qb * SUBGROUP_SIZE + (int)lane;
        int group_begin = 0;
        int group_end = 0;
        if (bidir_active && query < q && token_type_ids[query] == 1) {
            group_begin = query;
            while (group_begin > bidir_scan_lo && token_type_ids[group_begin - 1] == 1)
                --group_begin;
            group_end = query + 1;
            while (group_end < bidir_scan_hi && token_type_ids[group_end] == 1)
                ++group_end;
            group_begin += query_position_offset;
            group_end += query_position_offset;
        }
        bidir_group_begin[qb] = group_begin;
        bidir_group_end[qb] = group_end;
    }
#endif

#if PA_CUR_KV_F16
    // MIXED with PA_CUR_KV_F16: the cache serves the prefix [0, past_len) and Kc/Vc the exact,
    // uncompressed new rows [past_len, k) -- a compressed cache would not round-trip them. A WG key
    // tile crossing past_len is shortened (k_chunk) so every iteration reads one source; the fixed
    // DPAS tile shape stays and rows past k_chunk are masked (sdpa_micro's boundary handling).
    for (int k0 = window_k0_begin; k0 < causal_k;) {
        int k_chunk = min(causal_k - k0, kq_wg_tile_keys);
        if (k0 < past_len && k0 + k_chunk > past_len)
            k_chunk = past_len - k0;
#else
    for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) {
#endif
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == window_k0_begin);
#if PA_CUR_KV_F16
        const bool last = (k0 + k_chunk >= causal_k);
#else
        const bool last = (k0 + kq_wg_tile_keys >= causal_k);
#endif
#if IS_PA_MIXED
    #if PA_CUR_KV_F16
        // The loop splits at past_len, so this decision is workgroup-uniform.
        const bool from_cache = (k0 < past_len);
    #else
        // Constant: every `if (from_cache)` below folds away.
        const bool from_cache = true;
    #endif
#endif
        float8 S_tile[kq_key_blocks][kq_query_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

#ifdef KV_COMPRESSED
        half k_scale_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        half k_zpb_lane[kq_sg_tile_keys / SUBGROUP_SIZE];   // zp + 1152.0h (bias-trick bias folded in)
        FUNC_CALL(k_comp_per_key)(OPTIONAL_SHAPE_INFO_TENSOR k_scale_lane, k_zpb_lane, K_scales, K_zp, k_comp_base,
                                  key_base, k, lane);
#endif

#if IS_PA_MIXED
        // Everything in this block feeds only the CACHE read, hence `if (from_cache)`: on a
        // PA_CUR_KV_F16 tile it is all dead work, and the branch is uniform.
        uint k_page[kq_key_blocks];
        if (from_cache)
            FUNC_CALL(pa_k_pages)(k_page, block_indices, base_block_index, key_base, k);
    #if IS_PA_K_BY_CHANNEL
        half k_pa_sc_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        half k_pa_zp_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        if (from_cache)
            FUNC_CALL(pa_k_comp_by_channel)(k_pa_sc_ch, k_pa_zp_ch, K, k_page, b0_kv, key_base, k, lane,
                                            (int)lane);
    #elif IS_PA_KV_COMPRESSED
        half k_pa_sc_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        half k_pa_zp_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        if (from_cache)
            FUNC_CALL(pa_k_comp_by_token)(k_pa_sc_lane, k_pa_zp_lane, K, k_page, b0_kv, key_base, k, lane,
                                          (int)lane);
    #endif
    #if USE_1D_BLOCK_IO_K_PA_U4
        uchar16 k_pg[kq_sg_tile_keys / SUBGROUP_SIZE][PA_PAGE_READS(PA_K_ROW_ELEMS)];
        if (from_cache)
            FUNC_CALL(pa_k_page_read_1d)(k_pg, K, k_page, b0_kv);
    #endif
#endif

        #pragma unroll
        for (int db = 0; db < DKS_ACTIVE; ++db) {
            int8 qB[kq_query_blocks];
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int q_block = sg_j0_kq / SUBGROUP_SIZE + qb;
                qB[qb] = as_int8(intel_sub_group_block_read8(
                    (local void *)&Q_slm[Q_SLM_OFF(db, q_block)]));
            }

            ushort8 k_raw[kq_key_blocks];
#if IS_PA_MIXED
            if (from_cache) {
    #if USE_2D_BLOCK_IO_K_PA
            FUNC_CALL(pa_k_tile_b2d16)(k_raw, K, k_page, b0_kv, key_base, k, d, db);
    #elif USE_2D_BLOCK_IO_K_PA_I8
            uint kt[8];
        #if IS_PA_K_BY_CHANNEL
            FUNC_CALL(pa_k_tile_q_b2d)(k_raw, kt, K, k_page, k_pa_sc_ch, k_pa_zp_ch, b0_kv, key_base, k, d, db);
        #else
            FUNC_CALL(pa_k_tile_q_b2d)(k_raw, kt, K, k_page, k_pa_sc_lane, k_pa_zp_lane, b0_kv, key_base, k, d, db);
        #endif
    #elif USE_1D_BLOCK_IO_K_PA_U4
            FUNC_CALL(pa_k_tile_u4_1d)(k_raw, k_pg, k_pa_sc_ch, k_pa_zp_ch, d, db, (int)lane);
    #elif IS_PA_KV_COMPRESSED
        #if IS_PA_K_BY_CHANNEL
            FUNC_CALL(pa_k_tile_q_gather)(k_raw, K, k_page, k_pa_sc_ch, k_pa_zp_ch, b0_kv, key_base, k, d, db, lane,
                                          (int)lane);
        #else
            FUNC_CALL(pa_k_tile_q_gather)(k_raw, K, k_page, k_pa_sc_lane, k_pa_zp_lane, b0_kv, key_base, k, d, db, lane,
                                          (int)lane);
        #endif
    #else
            FUNC_CALL(pa_k_tile_gather)(k_raw, K, k_page, b0_kv, key_base, k, d, db, lane);
    #endif
            }
    #if PA_CUR_KV_F16
            else {
        #if IS_PA_K_U4
                if (kc_dword_ok) {
                    uint kw[DPAS_ROWS];
                    FUNC_CALL(kc_tile_u4_dword)(k_raw, kw, Kc_b2d, KcD_w_b2d, KcD_h, KcD_p, KcD_x0_dw, key_base,
                                                past_len, db);
                } else {
                    FUNC_CALL(kc_tile_u4_gather)(k_raw, Kc, ldk, key_base, past_len, k0, k_chunk, d, db, (int)lane);
                }
        #else
                // f16 / i8 cache: no depth permutation, so this is the plain-SDPA [key, head] read
                // pointed at Kc with a (key - past_len) row origin. Rows past q read as zero, which is
                // what the `key < k` masking already assumes.
                FUNC_CALL(k_tile_b2d16)(k_raw, Kc_b2d, KcD_w_b2d, KcD_h, KcD_p, KcD_x0, db, key_base, past_len);
        #endif
            }
    #endif
#elif USE_2D_BLOCK_IO_K_I8
            uint kt[8];
            FUNC_CALL(k_tile_i8_b2d)(k_raw, kt, K, KD_w, KD_h, KD_p, k_scale_lane, k_zpb_lane, key_base, db);
#elif USE_2D_BLOCK_IO_KV
            FUNC_CALL(k_tile_b2d16)(k_raw, K_b2d, KD_w_b2d, KD_h, KD_p, KD_x0, db, key_base, 0);
#else
    #ifdef KV_COMPRESSED
            FUNC_CALL(k_tile_gather)(k_raw, K, ldk, k_scale_lane, k_zpb_lane, key_base, k, d, db, lane);
    #else
            FUNC_CALL(k_tile_gather)(k_raw, K, ldk, key_base, k, d, db, lane);
    #endif
#endif

            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                #pragma unroll
                for (int qb = 0; qb < kq_query_blocks; ++qb)
                    S_tile[mb][qb] = DPAS_MAD_K16(as_short8(k_raw[mb]), qB[qb], S_tile[mb][qb]);
            }
        }

        half2 mask_tile;
        float2 k_mask;
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
            const int key = key_base + ii * SUBGROUP_SIZE + lane;
            #if WITH_ATTN_MASK
                if (MASK_IS_PER_KEY)
                    mask_tile[ii] = (key < k) ? msk[MSK_OFF(0, 0, 0, key)] : (half)0.0f;
                else
                    mask_tile[ii] = (half)0.0f;
            #else
                mask_tile[ii] = (half)0.0f;
            #endif
            // Mask against the chunk end: a PA_CUR_KV_F16 tile can be shortened at past_len, and
            // this also masks the valid current rows loaded into the unused tail of a shortened
            // cache iteration.
#if PA_CUR_KV_F16
            k_mask[ii] = (key < k0 + k_chunk) ? 0.0f : -INFINITY;
#else
            k_mask[ii] = (key < causal_k) ? 0.0f : -INFINITY;
#endif
        }
        float2 mask_tile_float = MASK_TO_FLOAT2(mask_tile);
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            mask_tile_float[ii] = mask_tile_float[ii] * iscale;

        #if WITH_ATTN_MASK
            float16 mask_full[kq_query_blocks][kq_sg_tile_keys / SUBGROUP_SIZE];
            if (MASK_IS_FULL_2D)
                FUNC_CALL(mask_tile_2d)(OPTIONAL_SHAPE_INFO_TENSOR mask_full, msk, iscale, wg_j0, sg_j0_kq, lane,
                                        key_base);
        #endif

        float alpha[kq_query_blocks];
        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            float lmax = -INFINITY;
            // Block-level causal skip (sdpa_micro's `causal_k_end > causal_q_begin` guard): if the
            // block's last key is <= its first query, every element is inside the causal region and
            // the per-element predicate is a no-op. Uniform across the subgroup (no lane term), so
            // the common case is straight-line code.
#if IS_CAUSAL
            const int blk_key_last = key_base + kq_sg_tile_keys - 1;
            const int blk_query_first =
                LOWER_RIGHT_SHIFT(query_position_offset + (int)(wg_j0 + sg_j0_kq) + qb * SUBGROUP_SIZE);
    #if SLIDING_WINDOW_SIZE
            // With a window the block must also sit fully inside it: the oldest key the block's
            // LAST query may attend is (blk_query_last - SLIDING_WINDOW_SIZE), so the block's
            // FIRST key must be newer than that.
            const int blk_query_last = blk_query_first + SUBGROUP_SIZE - 1;
            const bool causal_block_clear =
                blk_key_last <= blk_query_first && key_base > blk_query_last - SLIDING_WINDOW_SIZE;
    #else
            const bool causal_block_clear = blk_key_last <= blk_query_first;
    #endif
#endif
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                #pragma unroll
                for (int mm = 0; mm < 8; ++mm) {
                    const int key_rel = mb * 8 + mm;
                    const int mask_idx = key_rel / SUBGROUP_SIZE;
                    const int mask_lane = key_rel - mask_idx * SUBGROUP_SIZE;
                    const int query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    const int query_position = query_position_offset + query;
                    const int key = key_base + key_rel;
                    float s = S_tile[mb][qb][mm] + sub_group_broadcast(k_mask[mask_idx], mask_lane);
#ifdef STATIC_SCALAR_ATTN_MASK_VALUE
                    s += STATIC_SCALAR_ATTN_MASK_VALUE * iscale;
#endif
#ifdef HAS_SCALAR_ATTN_MASK
                    // Single-element runtime mask (rank-0 scalar / 1-element 1D): the value broadcasts
                    // to every logit, matching the const-scalar path above.
                    s += MASK_TO_FLOAT(msk[0]) * iscale;
#endif
                    #if WITH_ATTN_MASK
                        if (MASK_IS_PER_KEY) {
                            s += sub_group_broadcast(mask_tile_float[mask_idx], mask_lane);
                        } else if (MASK_IS_FULL_2D) {
                            s += mask_full[qb][mask_idx][mask_lane];
                        } else if (query < q && key < k) {
                            const int mask_query = (MSK_D2 == 1) ? 0 : query;
                            const int mask_key = (MSK_D3 == 1) ? 0 : key;
                            s += MASK_TO_FLOAT(msk[MSK_OFF(0, 0, mask_query, mask_key)]) * iscale;
                        }
                    #endif
#if HAS_QQ_BIAS && IS_PA_MIXED
                    // Speculative tree mask over the NEW keys: query and key_spec = key - past_len
                    // are both subsequence-relative new-token indices, ordered as the reference
                    // builds the mask (openvino/reference/paged_attention.hpp); qq_bias == 0 masks
                    // the pair.
                    if (qq_bias_num > 0 && key >= past_len && key < past_len + (int)spec_num) {
                        const int key_spec = key - past_len;
                        if (query >= 0 && query < (int)spec_num) {
                            const uint qq_off = cumulated_spec_num + (uint)query * spec_num + (uint)key_spec;
                            if (qq_bias[qq_off] == (QQ_BIAS_DATA_T)0)
                                s = -INFINITY;
                        }
                    }
#endif
#if IS_CAUSAL
                    if (!causal_block_clear) {
    #if SLIDING_WINDOW_SIZE
                        // Keys outside (query - SLIDING_WINDOW_SIZE, query] are dropped, matching
                        // sdpa_micro's greater_than() predicate. With a bottom-right-aligned mask
                        // the whole window shifts by (k - q), exactly as micro's col_offset does.
                        if (key > LOWER_RIGHT_SHIFT(query_position) ||
                            key <= LOWER_RIGHT_SHIFT(query_position - SLIDING_WINDOW_SIZE)) {
    #else
                        if (key > LOWER_RIGHT_SHIFT(query_position)) {
    #endif
    #if BIDIR_MASK
                            // ...unless the key is inside this query's own image group, which is
                            // bidirectional. Only reachable when the base predicate above already
                            // masked, so this can only ever un-mask -- which is also why the
                            // causal_block_clear skip stays sound: it proves the block is wholly
                            // INSIDE the causal+window region, where there is nothing to un-mask.
                            if (key < bidir_group_begin[qb] || key >= bidir_group_end[qb])
    #endif
                            s = -INFINITY;
                        }
                    }
#endif
                    S_tile[mb][qb][mm] = s;
                    lmax = fmax(lmax, s);
                }
            }

            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            __builtin_IB_atomic_max_local_f32(&S_max_slm[query], lmax);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#if IS_PA_K_U4 && PA_CUR_KV_F16
        if (!from_cache && sg_i_sv == 0)
            FUNC_CALL(vc_prefetch)(Vc_b2d, VcD_w_b2d, VcD_h, VcD_p, VcD_x0, sg_j0_sv, dv, k0, k_chunk, past_len);
#endif

        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            const float m_new = S_max_slm[query];
            // Required when a query has no valid keys in the current prefix, e.g. future
            // remainder/causal/window masks or a fully masked row. In that case m_new is
            // -inf, and unguarded max rescaling would form -inf - -inf and poison S/A.
            const bool ok = isfinite(m_new);
#if MICRO_MATH
            // sdpa_micro keeps raw QK maxima and subtracts before scaling. The default
            // path stores scaled maxima, which changes rounding in both exp2 arguments.
            const float a = ok ? native_exp2((S_max_tile[qb] - m_new) * scale) : 1.0f;
            float lsum = 0.0f;

            S_max_tile[qb] = ok ? m_new : S_max_tile[qb];
#else
            const float m_log2 = ok ? m_new * scale : 0.0f;
            const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;
            float lsum = 0.0f;

            S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
#endif
            alpha[qb] = a;

            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
#if MICRO_MATH
                float8 exp_tile = ok ? native_exp2((S_tile[mb][qb] - m_new) * scale) : (float8)0.0f;
                // tile_vreduce_add in sdpa_micro accumulates keys in order, including
                // across the two DPAS row blocks of each 16-key subgroup tile.
                #pragma unroll
                for (int mm = 0; mm < DPAS_ROWS; ++mm)
                    lsum += exp_tile[mm];
#else
                float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                lsum += exp_tile[0] + exp_tile[1] + exp_tile[2] + exp_tile[3]
                      + exp_tile[4] + exp_tile[5] + exp_tile[6] + exp_tile[7];
#endif

                const int key = sg_i0_kq + mb * 8;
                const int key_block = key / SUBGROUP_SIZE;
                const int key_lane = key - key_block * SUBGROUP_SIZE;
                const int s_half_offset = (key_block * kq_wg_tile_queries + query) * SUBGROUP_SIZE + key_lane;
                vstore4(as_uint4(PACK_SOFTMAX8(exp_tile)), 0, &S_slm[s_half_offset >> 1]);
            }
#if MICRO_MATH
            if (!first)
                S_sum_tile[qb] *= a;
            S_sum_tile[qb] += lsum;
#else
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
#endif
        }

        if (last) {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                S_sum_slm[query * kq_sg_per_wg_keys + sg_i_kq] = S_sum_tile[qb];
            }
        }

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        if (!first) {
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                float8 av;
                const int rel_query = sg_i0_sv + r * 8 - sg_j0_kq;
                const int alpha_qb = rel_query / SUBGROUP_SIZE;
                const int alpha_lane0 = rel_query - alpha_qb * SUBGROUP_SIZE;
                // alpha_qb is a RUNTIME value (from sg_ij), and indexing a private array at runtime
                // makes IGC move the array to scratch -- inside the k0 loop. kq_query_blocks is a
                // compile-time constant, so select the element with a chain instead; the
                // broadcast's runtime lane is only an indirect register move.
                float alpha_sel = alpha[0];
                #pragma unroll
                for (int t = 1; t < kq_query_blocks; ++t)
                    alpha_sel = (t == alpha_qb) ? alpha[t] : alpha_sel;
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr)
                    av[rr] = sub_group_broadcast(alpha_sel, alpha_lane0 + rr);
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= av;
            }
        }

        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

#if MICRO_MATH
        // ugemm_vs starts each key tile at zero. Adding that partial result after the
        // DPAS loop rounds differently from feeding the previous A_tile into DPAS.
        float8 A_tile1[sv_score_blocks][sv_value_blocks];
        #pragma unroll
        for (int r = 0; r < sv_score_blocks; ++r)
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd)
                A_tile1[r][cd] = (float8)0.0f;
#endif

        #if USE_2D_BLOCK_IO_V_I8
            // Declared outside the cp loop because one read serves two consecutive cp blocks (see below).
            uint vt[8 * sv_value_blocks];
        #endif
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
#if IS_PA_K_U4 && PA_CUR_KV_F16
            // Exact chunks can end before the fixed S*V tile: skip whole blocks past k_chunk (their
            // scores are zero) together with their SLM/V loads, dequant and DPAS. Uniform; the
            // barriers stay outside.
            if (cp * SUBGROUP_SIZE >= k_chunk)
                continue;
#endif
            #if USE_2D_BLOCK_IO_V_I8
                // Issued ahead of the pA (S_slm) reads so the global latency overlaps the SLM
                // traffic; one read serves two cp blocks, and cp is an unroll constant, so the
                // selection folds.
                    const bool vt_do_read = ((cp & 1) == 0);
                    const int vt_half = (cp & 1) * 4;
                if (vt_do_read)
                    FUNC_CALL(v_i8_read)(vt, V, VD_w, VD_h, VD_p, sg_j0_sv, k0, cp);
            #endif

            short8 pA[sv_score_blocks];
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                const int query0 = sg_i0_sv + r * 8;
                pA[r] = as_short8(intel_sub_group_block_read_us8(
                    (local void *)&S_slm[((cp * kq_wg_tile_queries + query0) * SUBGROUP_SIZE) >> 1]));
            }

            #if USE_2D_BLOCK_IO_V_I8
                const half vzb_c = FUNC_CALL(v_i8_comp_fold)(OPTIONAL_SHAPE_INFO_TENSOR pA, V_scales, V_zp, v_comp_base,
                                                             k0, cp, k, lane);
            #elif IS_PA_KV_COMPRESSED && IS_PA_MIXED
                // Cache tiles only: a PA_CUR_KV_F16 tile reads plain f16 Vc, so v_zp_c stays 0.
                half v_zp_c = (half)0.0f;
                if (from_cache)
                    v_zp_c = FUNC_CALL(pa_v_comp_fold)(pA, V, block_indices, base_block_index, k0, cp, k, b0_kv, lane);
            #endif

            int8 vb[sv_value_blocks];
            #if IS_PA_MIXED
            if (from_cache) {
                // A cp block is SUBGROUP_SIZE (== DPAS_K == PAGED_ATTENTION_BLOCK_SIZE) keys
                // starting at a multiple of kq_wg_tile_keys, i.e. exactly one cache page, so the
                // page lookup is hoisted out of the cd and key_pair loops.
                const int cp_key0 = k0 + cp * SUBGROUP_SIZE;
                const size_t v_page_base =
                    FUNC_CALL(pa_v_page_base)(block_indices, base_block_index, cp_key0, k, b0_kv);
                #if IS_PA_KV_COMPRESSED
                    // Compressed V page: lane == token for the per-key comp and key_rel for the
                    // dequant. The scale is already folded into pA, so only the zp subtraction
                    // happens here (broadcast per key).
                    #if USE_2D_BLOCK_IO_V_PA_I8
                    uint vt_pa[8 * sv_value_blocks];
                    half4 vzp4[4];
                    FUNC_CALL(pa_v_tile_q_b2d)(vb, vt_pa, vzp4, V, v_page_base, cp_key0, k, dv, sg_j0_sv, v_zp_c);
                    #elif USE_1D_BLOCK_IO_V_PA_U4
                    uchar16 v_pg[PA_PAGE_READS(PA_V_ROW_ELEMS)];
                    FUNC_CALL(pa_v_tile_u4_1d)(vb, v_pg, V, v_page_base, dv, sg_j0_sv, lane, v_zp_c);
                    #else
                    FUNC_CALL(pa_v_tile_q_gather)(vb, V, v_page_base, cp_key0, k, dv, sg_j0_sv, lane, (int)lane, v_zp_c);
                    #endif
                #elif USE_2D_BLOCK_IO_V_PA
                    FUNC_CALL(pa_v_tile_b2d16)(vb, V, v_page_base, cp_key0, k, dv, sg_j0_sv);
                #else
                    FUNC_CALL(pa_v_tile_gather)(vb, V, v_page_base, cp_key0, k, dv, sg_j0_sv, lane);
                #endif
            }
            #if PA_CUR_KV_F16
            else {
                FUNC_CALL(v_tile_b2d16)(vb, Vc_b2d, VcD_w_b2d, VcD_h, VcD_p, VcD_x0, sg_j0_sv, k0, cp, past_len);
            }
            #endif
            #elif USE_2D_BLOCK_IO_V_I8
                // The paired read above gives a 32-key x 16-value tile (lane = value, 4 keys per
                // uint); this cp block uses the 4 uints at vt_half.
                half4 zpb4[4];
                FUNC_CALL(v_i8_dequant)(vb, zpb4, vt, vt_half, vzb_c);
            #elif USE_2D_BLOCK_IO_KV
                FUNC_CALL(v_tile_b2d16)(vb, V_b2d, VD_w_b2d, VD_h, VD_p, VD_x0, sg_j0_sv, k0, cp, 0);
            #else
        #ifdef KV_COMPRESSED
                FUNC_CALL(v_tile_gather)(OPTIONAL_SHAPE_INFO_TENSOR vb, V, ldv, V_scales, V_zp, b1, b0_kv, k0, cp, k, dv,
                                         sg_j0_sv, lane);
        #else
                FUNC_CALL(v_tile_gather)(OPTIONAL_SHAPE_INFO_TENSOR vb, V, ldv, k0, cp, k, dv, sg_j0_sv, lane);
        #endif
            #endif

            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r)
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
#if MICRO_MATH
                    A_tile1[r][cd] = DPAS_MAD_K16(pA[r], vb[cd], A_tile1[r][cd]);
#else
                    A_tile[r][cd] = DPAS_MAD_K16(pA[r], vb[cd], A_tile[r][cd]);
#endif
        }
#if MICRO_MATH
        #pragma unroll
        for (int r = 0; r < sv_score_blocks; ++r)
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd)
                A_tile[r][cd] += A_tile1[r][cd];
#endif
#if PA_CUR_KV_F16
        k0 += k_chunk;
#endif
    }

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        float8 inv_l;
        #pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            const int query = sg_i0_sv + r * 8 + rr;
            float l = S_sum_slm[query * kq_sg_per_wg_keys + 0];
            #pragma unroll
            for (int p = 1; p < kq_sg_per_wg_keys; ++p)
                l += S_sum_slm[query * kq_sg_per_wg_keys + p];
            inv_l[rr] = (l > 0.0f) ? native_recip(l) : 0.0f;
        }
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            A_tile[r][cd] *= inv_l;
    }

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd) {
            DT_OUT8_T out = ACC_TO_OUT8(A_tile[r][cd]);
            const int col = sg_j0_sv + cd * SUBGROUP_SIZE;
            const int row = wg_j0 + sg_i0_sv + r * 8;
#if USE_2D_BLOCK_IO_A
            if (row + 7 < q && col + SUBGROUP_SIZE <= dv) {
                intel_sub_group_2d_block_write_16b_8r16x1c(
                    (global void *)A, AD_w, AD_h, AD_p,
                    (int2)(col, row),
                    (private ushort *)&out);
            } else {
#endif
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const int out_row = row + rr;
                    const int out_col = col + lane;
                    if (out_row < q && out_col < dv)
                        A[(size_t)out_row * lda + out_col] = out[rr];
                }
#if USE_2D_BLOCK_IO_A
            }
#endif
        }
    }
}
