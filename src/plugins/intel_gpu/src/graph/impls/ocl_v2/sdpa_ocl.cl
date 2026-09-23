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
            float iscale = convert_float(*scale_ptr);
            float scale = native_recip(iscale);
        #else
            float scale = convert_float(*scale_ptr);
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
    const int KD_w = d * (int)sizeof(KEY_DATA_T), KD_h = k, KD_p = (int)ldk * (int)sizeof(KEY_DATA_T);
    const int VD_w = dv * (int)sizeof(VAL_DATA_T), VD_h = k, VD_p = (int)ldv * (int)sizeof(VAL_DATA_T);
    const int AD_w = dv * (int)sizeof(OUTPUT_TYPE), AD_h = q, AD_p = (int)lda * (int)sizeof(OUTPUT_TYPE);

#if PA_CUR_KV_F16
    // Surfaces for the NEW-token part of the key range. Not KD_*/VD_*, which describe the cache
    // here: Kc/Vc are f16 and q rows tall, and rows past q read as zero (those keys are masked
    // anyway).
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

#if USE_2D_BLOCK_IO_KV
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
        // u4: Q adopts the K page's permuted depth labelling (see PA_K_U4_CHANNEL), paid here once
        // per workgroup instead of per k0 tile. A chunk spans its 32-channel window, read as two
        // halves w0/w1, and q_pack dword j takes half `par` of dwords 2j and 2j+1, i.e. channels
        // (win+4j+par, win+4j+2+par).
        const int u4_win = PA_K_U4_WIN(db);
        const int u4_par = PA_K_U4_PAR(db);
        uint8 w0, w1;
    #if USE_2D_BLOCK_IO_Q
        if (query_base + SUBGROUP_SIZE <= q && u4_win + 2 * DPAS_K <= d) {
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(u4_win / 2, query_base), (private uint *)&w0);
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(u4_win / 2 + DPAS_K / 2, query_base), (private uint *)&w1);
        } else
    #endif
        {
            const int query = query_base + lane;
            ushort16 qv0 = (ushort16)0;
            ushort16 qv1 = (ushort16)0;
            if (query < q) {
                const global ushort *q_row = (const global ushort *)(Q + (size_t)query * ldq + u4_win);
                if (u4_win + 2 * DPAS_K <= d) {
                    qv0 = vload16(0, q_row);
                    qv1 = vload16(1, q_row);
                } else {
                    #pragma unroll
                    for (int head_offset = 0; head_offset < DPAS_K; ++head_offset) {
                        if (u4_win + head_offset < d)
                            qv0[head_offset] = q_row[head_offset];
                        if (u4_win + DPAS_K + head_offset < d)
                            qv1[head_offset] = q_row[DPAS_K + head_offset];
                    }
                }
            }
            w0 = as_uint8(as_short16(qv0));
            w1 = as_uint8(as_short16(qv1));
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint a = (j < 4) ? w0[2 * j] : w1[2 * (j - 4)];
            const uint b = (j < 4) ? w0[2 * j + 1] : w1[2 * (j - 4) + 1];
            q_pack[j] = u4_par ? ((a >> 16) | (b & 0xFFFF0000u)) : ((a & 0x0000FFFFu) | (b << 16));
        }
#else
        const int head_base = db * DPAS_K;
#if USE_2D_BLOCK_IO_Q
        if (query_base + SUBGROUP_SIZE <= q && head_base + DPAS_K <= d) {
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                (global void *)Q, QD_w, QD_h, QD_p,
                (int2)(head_base / 2, query_base), (private uint *)&q_pack);
        } else
#endif
        {
            const int query = query_base + lane;
            ushort16 qv = (ushort16)0;
            if (query < q) {
                if (head_base + DPAS_K <= d) {
                    qv = vload16(0, (global ushort *)(Q + (size_t)query * ldq + head_base));
                } else {
                    #pragma unroll
                    for (int head_offset = 0; head_offset < DPAS_K; ++head_offset) {
                        if (head_base + head_offset < d) {
                            qv[head_offset] = as_ushort(Q[(size_t)query * ldq + head_base + head_offset]);
                        }
                    }
                }
            }
            q_pack = as_uint8(as_short16(qv));
        }
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
    // last query (groups are contiguous, so that query suffices). The scan is subgroup-cooperative
    // and uniform, so the break and sub_group_reduce_min() are reached by every lane. It runs in
    // LOCAL space bounded by q (groups never leave the new-token region) and relies on
    // pa_kv_cache_update having written the new tokens before this stage, as the reference does.
    {
        const int wg_q_end = min((int)wg_j0 + kq_wg_tile_queries, q) - 1;
        if (bidir_active && wg_q_end >= 0 && token_type_ids[wg_q_end] == 1) {
            int group_end = wg_q_end + 1;
            while (group_end < q) {
                const int chunk_end = min(q, group_end + SUBGROUP_SIZE);  // exclusive
                const int idx = group_end + (int)lane;
                const bool ends_group = (idx < chunk_end) && (token_type_ids[idx] != 1);
                const int first = sub_group_reduce_min(ends_group ? idx : INT_MAX);
                if (first != INT_MAX) {
                    group_end = first;
                    break;
                }
                group_end = chunk_end;
            }
            // group_end <= q: the loop only ever assigns an index below q or the clamped chunk end,
            // so the KEY-space result stays <= query_position_offset + q == k.
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
        int group_begin = window_begin_local;
        while (group_begin > 0) {
            const int chunk_begin = max(0, group_begin - SUBGROUP_SIZE);
            const int idx = chunk_begin + (int)lane;
            const bool ends_group = (idx < group_begin) && (token_type_ids[idx] != 1);
            const int last = sub_group_reduce_max(ends_group ? idx : -1);
            if (last >= 0) {
                group_begin = last + 1;
                break;
            }
            group_begin = chunk_begin;
        }
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
        // Per-token K scale/zp depend only on the key, so load them once per k0 tile with one
        // 16-wide load (lane L -> key key_base + L) instead of per-key SIMD-1 loads inside the
        // dequant loop. Kept in half for the bias-trick dequant below: zp absorbs the widen bias
        // (+1152.0h), so a byte dequants as (as_half(0x6480 ^ byte) - (zp + 1152)) * scale.
        half k_scale_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        half k_zpb_lane[kq_sg_tile_keys / SUBGROUP_SIZE];   // zp + 1152.0h (bias-trick bias folded in)
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
            const int sc_key = key_base + ii * SUBGROUP_SIZE + lane;
            const uint sc_off = k_comp_base + KEY_COMP_OFF(0, 0, sc_key, 0);
            k_scale_lane[ii] = (sc_key < k) ? convert_half(K_scales[sc_off]) : (half)0.0f;
            #if INPUT0_IS_BF16
            k_zpb_lane[ii] = (sc_key < k) ? convert_half(K_zp[sc_off]) : (half)0.0f;
            #else
            k_zpb_lane[ii] = (sc_key < k) ? (convert_half(K_zp[sc_off]) + (half)1152.0h) : (half)1152.0h;
            #endif
        }
#endif

#if IS_PA_MIXED
        // The block_indices[] page lookup depends only on the key, so it is hoisted out of the db /
        // key loops to one per DPAS row-block (an 8-aligned row-block never straddles a 16-key
        // page). The mb_key0 < k guard keeps it inside this subsequence's blocks (key_base can run
        // past k on the last k0 tile). Everything from here to the end of this block feeds only the
        // CACHE read, hence `if (from_cache)`: on a PA_CUR_KV_F16 tile it is all dead work, and the
        // branch is uniform.
        uint k_page[kq_key_blocks];
        if (from_cache) {
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                const int mb_key0 = key_base + mb * DPAS_ROWS;
                k_page[mb] = (mb_key0 < k) ? block_indices[base_block_index + mb_key0 / PAGED_ATTENTION_BLOCK_SIZE]
                                           : 0u;
            }
        }

    #if IS_PA_K_BY_CHANNEL
        // BY_CHANNEL comp is indexed by CHANNEL, and the KQ A operand is K with lane == head dim,
        // so a channel's (scale, zp) is a plain per-lane scalar: no sub_group_broadcast in the
        // dequant, unlike BY_TOKEN's per-key comp. The pairs are interleaved, so one uint block
        // read at dword db * SUBGROUP_SIZE gives lane L channel db * DPAS_K + L. Two guards, both
        // mandatory:
        //  - a tile reaching past K_HEAD_SIZE (a partial last tile, or u4's even-rounded
        //    DKS_ACTIVE) must not read past the comp region, which is exactly K_HEAD_SIZE dwords;
        //  - a key group at/past k had its page clamped to 0, whose comp bytes are arbitrary: sc =
        //    zp = 0 keeps the dequant finite, and a NaN would survive the -INFINITY mask (NaN +
        //    -INFINITY is NaN).
        half k_pa_sc_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        half k_pa_zp_ch[kq_sg_tile_keys / SUBGROUP_SIZE][DKS_ACTIVE];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            const global uint *k_comp_ch = (const global uint *)(
                K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv) +
                PA_K_COMP_OFF);
            const bool sc_valid = (key_base + kg * SUBGROUP_SIZE) < k;
        #if IS_PA_K_U4
            // u4: tiles 2g and 2g+1 want the comp of channels (win + 2L) and (win + 2L + 1), which
            // are adjacent, so one uint2 per lane covers the pair -- one coalesced 128-byte span.
            // Same two guards as i8.
            #pragma unroll
            for (int g = 0; g < DKS_ACTIVE / 2; ++g) {
                const int u4_win = g * (2 * DPAS_K);
                uint2 pair2 = (uint2)(0u, 0u);
                if (u4_win + 2 * DPAS_K <= K_HEAD_SIZE) {
                    if (sc_valid)
                        pair2 = vload2(lane, k_comp_ch + u4_win);
                } else {
                    const int c0 = u4_win + 2 * (int)lane;
                    pair2.s0 = (sc_valid && c0 < K_HEAD_SIZE) ? k_comp_ch[c0] : 0u;
                    pair2.s1 = (sc_valid && c0 + 1 < K_HEAD_SIZE) ? k_comp_ch[c0 + 1] : 0u;
                }
                const half2 sc_zp0 = as_half2(pair2.s0);
                const half2 sc_zp1 = as_half2(pair2.s1);
                k_pa_sc_ch[kg][2 * g + 0] = sc_zp0.s0;
                k_pa_zp_ch[kg][2 * g + 0] = sc_zp0.s1;
                k_pa_sc_ch[kg][2 * g + 1] = sc_zp1.s0;
                k_pa_zp_ch[kg][2 * g + 1] = sc_zp1.s1;
            }
        #else
            #pragma unroll
            for (int db = 0; db < DKS_ACTIVE; ++db) {
                uint pair = 0u;
                if (db < K_HEAD_SIZE / DPAS_K) {
                    pair = sc_valid ? intel_sub_group_block_read(k_comp_ch + db * SUBGROUP_SIZE) : 0u;
                } else if (db * DPAS_K + (int)lane < K_HEAD_SIZE) {
                    pair = sc_valid ? k_comp_ch[db * SUBGROUP_SIZE + lane] : 0u;
                }
                const half2 sc_zp = as_half2(pair);
                k_pa_sc_ch[kg][db] = sc_zp.s0;
                k_pa_zp_ch[kg][db] = sc_zp.s1;
            }
        #endif
        }
        }
    #elif IS_PA_KV_COMPRESSED
        // BY_TOKEN comp: per-key scale/zp, indexed by token only, loaded once per page with one
        // 16-wide load each (lane L = token L; a 16-key group is exactly one page). The dequant
        // takes each key's value with a sub_group_broadcast at a constant lane, which folds into
        // the consumer. Keys at/past k get sc = zp = 0: their comp bytes were never written and
        // could be NaN, and the block read below has no per-key guard to discard them.
        half k_pa_sc_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        half k_pa_zp_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            const global half *k_comp = (const global half *)(
                K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv) +
                PA_K_COMP_OFF);
            const bool sc_valid = (key_base + kg * SUBGROUP_SIZE + (int)lane) < k;
            k_pa_sc_lane[kg] = sc_valid ? k_comp[lane] : (half)0.0h;
            k_pa_zp_lane[kg] = sc_valid ? k_comp[PAGED_ATTENTION_BLOCK_SIZE + lane] : (half)0.0h;
        }
        }
    #endif

    #if USE_1D_BLOCK_IO_K_PA_U4
        // Whole-page read, hoisted out of the db loop: the page bytes do not depend on db (a byte
        // is a channel pair), so PA_PAGE_READS uc16 reads replace the per-(db, mb, key) gather. The
        // host only enables it where block2d cannot reach (u4 rows of 16 or 32 bytes), which keeps
        // the live page small.
        uchar16 k_pg[kq_sg_tile_keys / SUBGROUP_SIZE][PA_PAGE_READS(PA_K_ROW_ELEMS)];
        if (from_cache) {
        #pragma unroll
        for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
            // Page-aligned like the comp loop above; a group at/past k reads page 0, which is
            // always allocated.
            const global uchar *k_pg_base = (const global uchar *)(
                K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv));
            #pragma unroll
            for (int r = 0; r < PA_PAGE_READS(PA_K_ROW_ELEMS); ++r)
                k_pg[kg][r] = intel_sub_group_block_read_uc16(k_pg_base + r * PA_PAGE_RD_BYTES);
        }
        }
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
            // Token-major f16 K cache: a page is a [PAGED_ATTENTION_BLOCK_SIZE keys, K_HEAD_SIZE]
            // row-major tile, the [key, head] geometry of the input read below with the page as the
            // surface and K_HEAD_SIZE as the pitch, so the same 16b builtin lands the A operand
            // (lane = head, element = key). One read covers one 16-key page, so loop per key group
            // and take its page from k_page[] (indexed per row-block).
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
                const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
                // Height clamped to the keys the page holds: slots at/past k were never written,
                // and a NaN from one would survive the masked-out score. A group entirely at/past k
                // (height <= 0 is not a legal read) is zero-filled; reachable because key_base can
                // run past k on the last k0 tile.
                const int kp_rows = PA_PAGE_ROWS(k, kg_key0);
                if (kp_rows > 0) {
                    // PA_K_PAGE_STRIDE by value (the config #if proves the ADJUSTED_* collapse
                    // here), but spelled out: IGC strength-reduces (x * 16) * K_HEAD_SIZE and x *
                    // (16 * K_HEAD_SIZE) differently at non-power-of-two heads (48/96), and this is
                    // the measured form.
                    const global half *Kp =
                        (const global half *)(K + (((size_t)k_page[kg_mb] * KV_HEADS_NUM + b0_kv) *
                                                   PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE));
                    const int KP_w = d * (int)sizeof(half);
                    const int KP_p = K_HEAD_SIZE * (int)sizeof(half);
                    intel_sub_group_2d_block_read_16b_16r16x1c(
                        (global void *)Kp, KP_w, kp_rows, KP_p,
                        (int2)(db * DPAS_K, 0), (private ushort *)&k_raw[kg_mb]);
                } else {
                    #pragma unroll
                    for (int mb = 0; mb < SUBGROUP_SIZE / DPAS_ROWS; ++mb)
                        k_raw[kg_mb + mb] = (ushort8)0;
                }
            }
    #elif USE_2D_BLOCK_IO_K_PA_I8
            // Token-major i8 K cache, either quant mode (BY_CHANNEL's data region has BY_TOKEN's
            // geometry; only the comp and the dequant index differ): the data is a
            // [PAGED_ATTENTION_BLOCK_SIZE, K_HEAD_SIZE] i8 tile, so the 8-bit VNNI-transform read
            // lands lane = head with 4 keys per uint. The builtin is 32-row only on Xe2 while a
            // page has 16 tokens, so the height is clamped and uints 0..3 are used; two key groups
            // cannot share a read (their pages are not adjacent). The dequant is an explicit (q -
            // zp) * scale in half, identical to the scalar branch, so SDPA_OCL_K_PA_I8_2D=0 is a
            // clean bisection toggle (and the bias trick would round the writer's non-integer zp).
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
                const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
                #pragma unroll
                for (int mb = 0; mb < SUBGROUP_SIZE / DPAS_ROWS; ++mb)
                    k_raw[kg_mb + mb] = (ushort8)0;
                const int kp_rows = PA_PAGE_ROWS(k, kg_key0);
                if (kp_rows > 0) {
                    uint kt[8];
                    #if IS_PA_K_U4
                    // u4: the row is PA_K_ROW_ELEMS bytes and a byte column is a channel pair, so
                    // one read at byte column PA_K_U4_WIN(db)/2 covers the tile pair (db, db^1).
                    // The partner tile re-issues the same read (an L1 hit); sharing it would hoist
                    // k_raw out of the db loop (128 more live ushorts at head 128). x is in bytes
                    // and a multiple of SUBGROUP_SIZE, which meets the 8-bit "multiple of four"
                    // rule.
                    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                        (global void *)(K + PA_K_PAGE_OFF(k_page[kg_mb], b0_kv)),
                        PA_K_ROW_ELEMS, kp_rows, PA_K_ROW_ELEMS, (int2)(PA_K_U4_WIN(db) / 2, 0),
                        (private uint *)&kt[0]);
                    #else
                    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                        (global void *)(K + PA_K_PAGE_OFF(k_page[kg_mb], b0_kv)),
                        d, kp_rows, K_HEAD_SIZE, (int2)(db * DPAS_K, 0), (private uint *)&kt[0]);
                    #endif
                    #if IS_PA_K_BY_CHANNEL
                    // Per-channel scale/zp are per LANE, so they leave the key loop entirely: one pair
                    // for the whole (page, head-dim tile) instead of BY_TOKEN's broadcast per key.
                    const half k_sc = k_pa_sc_ch[kg][db];
                    const half k_zp = k_pa_zp_ch[kg][db];
                    #endif
                    #pragma unroll
                    for (int u = 0; u < SUBGROUP_SIZE / 4; ++u) {
                        const uint w = kt[u];
                        #pragma unroll
                        for (int bb = 0; bb < 4; ++bb) {
                            const int krel = kg * SUBGROUP_SIZE + u * 4 + bb;
                            #if !IS_PA_K_BY_CHANNEL
                            const half k_sc = sub_group_broadcast(k_pa_sc_lane[kg], u * 4 + bb);
                            const half k_zp = sub_group_broadcast(k_pa_zp_lane[kg], u * 4 + bb);
                            #endif
                            #if IS_PA_K_U4
                            // The nibble select is lane-UNIFORM (the parity is the tile's, not the
                            // lane's), so it folds into the shift amount rather than a per-lane sel.
                            // Unsigned by construction: the int4 quantizer clamps to [0, 15] with
                            // zp = -min*scale, so there is no CHAR_MIN and no sign extension.
                            const uint kb_ = (w >> (bb * 8)) & 0xFFu;
                            const half deq_k =
                                PA_DEQ((half)U4_NIBBLE_SEL(kb_, PA_K_U4_PAR(db)), k_zp, k_sc);
                            #else
                            const half deq_k = PA_DEQ((half)(char)((w >> (bb * 8)) & 0xFFu), k_zp, k_sc);
                            #endif
                            k_raw[krel / DPAS_ROWS][krel % DPAS_ROWS] = as_ushort(deq_k);
                        }
                    }
                }
            }
    #elif USE_1D_BLOCK_IO_K_PA_U4
            // Same dequant and k_raw writes as the scalar branch below, only the load differs (a
            // register subscript into the hoisted page), so SDPA_OCL_K_PA_1D=0 bisects the read
            // alone. The column group is db >> 1, a constant, so PA_PAGE_R/I fold. The scalar
            // branch's per-key `head < d && key < k` guard is dropped, as in the block2d branches:
            // keys at/past k only need to be FINITE (the mask adds -INFINITY; a nibble is bounded
            // and sc/zp were zeroed), and `head < d` folds into the scale ((n - zp) * 0 == 0).
            const int head = PA_K_U4_CHANNEL(db, lane);
            const int k_pg_col = (PA_K_U4_WIN(db) / 2) / SUBGROUP_SIZE;
            const bool head_ok = (head < d);
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                const int kg = mb / (SUBGROUP_SIZE / DPAS_ROWS);
                // Per-channel comp is this lane's own and constant across the row-block's keys,
                // exactly as in the scalar branch.
                const half k_sc = head_ok ? k_pa_sc_ch[kg][db] : (half)0.0h;
                const half k_zp = k_pa_zp_ch[kg][db];
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int krel = mb * DPAS_ROWS + key_offset;   // key's subgroup-local index
                    const int tok = krel % PAGED_ATTENTION_BLOCK_SIZE;
                    const uint kb_ = (uint)k_pg[kg][PA_PAGE_R(PA_K_ROW_ELEMS, tok, k_pg_col)]
                                                  [PA_PAGE_I(PA_K_ROW_ELEMS, tok, k_pg_col)];
                    const half deq_k =
                        PA_DEQ((half)U4_NIBBLE_SEL(kb_, PA_K_U4_PAR(db)), k_zp, k_sc);
                    k_raw[mb][key_offset] = as_ushort(deq_k);
                }
            }
    #elif IS_PA_KV_COMPRESSED
            // i8/u4 K page, per-key scalar gather: the d-major page (its 16-byte row is under the
            // block2d minimum) and token-major pages whose pitch misses the host's block2d rule.
            // PA_K_TOKEN_STRIDE / PA_K_HIDDEN_STRIDE select the addressing. The dequant is (q - zp)
            // * scale in half, matching the reference; the plain-SDPA bias trick only pays off over
            // a wide transform read.
            #if IS_PA_K_U4
            // Permuted depth: head is still the channel (so the `head < d` guard is unchanged), but
            // two channels share a byte, so the address is head >> 1 -- lane-contiguous bytes.
            const int head = PA_K_U4_CHANNEL(db, lane);
            const int head_addr = head >> 1;
            #else
            const int head = db * DPAS_K + lane;
            const int head_addr = head;
            #endif
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                // Page base for this row-block, hoisted above; only the intra-page key offset and
                // the head vary here.
                const size_t mb_page_base =
                    PA_K_PAGE_OFF(k_page[mb], b0_kv) +
                    (size_t)head_addr * PA_K_HIDDEN_STRIDE;
                #if IS_PA_K_BY_CHANNEL
                // head == db * DPAS_K + lane here too, so the per-channel pair is this lane's own and
                // is constant across the row-block's keys: no broadcast, and it lifts out of the key
                // loop. A row-block maps to key group mb / (SUBGROUP_SIZE / DPAS_ROWS).
                const half k_sc = k_pa_sc_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
                const half k_zp = k_pa_zp_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
                #endif
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int krel = mb * DPAS_ROWS + key_offset;   // key's subgroup-local index
                    const int key = key_base + krel;
                    #if !IS_PA_K_BY_CHANNEL
                    // sub_group_broadcast is a subgroup COLLECTIVE, so it must run on every lane --
                    // keep it outside the per-lane-divergent (head < d) guard below. krel is a
                    // compile-time constant in this fully unrolled loop, so the broadcast folds into
                    // the consuming add/mul source region rather than emitting a shuffle.
                    const half k_sc = sub_group_broadcast(k_pa_sc_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                    const half k_zp = sub_group_broadcast(k_pa_zp_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                    #endif
                    if (head < d && key < k) {
                        // key_base is a multiple of PAGED_ATTENTION_BLOCK_SIZE, so the key's token
                        // within its page is just its subgroup-local index.
                        const int tok = krel % PAGED_ATTENTION_BLOCK_SIZE;
                        #if IS_PA_K_U4
                        const uint kb_ = (uint)(uchar)K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE];
                        const half deq_k =
                            PA_DEQ((half)U4_NIBBLE_SEL(kb_, PA_K_U4_PAR(db)), k_zp, k_sc);
                        #else
                        const half deq_k = PA_DEQ((half)K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE], k_zp, k_sc);
                        #endif
                        k_raw[mb][key_offset] = as_ushort(deq_k);
                    }
                }
            }
    #else
            const int head = db * DPAS_K + lane;
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                // Page base for this row-block, hoisted above; only the intra-page key offset and
                // the head vary here.
                const size_t mb_page_base =
                    PA_K_PAGE_OFF(k_page[mb], b0_kv) +
                    (size_t)head * PA_K_HIDDEN_STRIDE;
                #pragma unroll
                for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                    const int key = key_base + mb * DPAS_ROWS + key_offset;
                    if (head < d && key < k) {
                        const int tok = key % PAGED_ATTENTION_BLOCK_SIZE;
                        k_raw[mb][key_offset] = as_ushort(K[mb_page_base + (size_t)tok * PA_K_TOKEN_STRIDE]);
                    }
                }
            }
    #endif
            }
    #if PA_CUR_KV_F16
            else {
        #if IS_PA_K_U4
                // u4 Kc: tile db needs lane L = channel PA_K_U4_WIN(db) + 2L + PA_K_U4_PAR(db) (the
                // permuted depth axis Q_slm is staged in), a stride-2 gather no 16b block read can
                // do. Channels (win + 2L, win + 2L + 1) are adjacent halves, i.e. one dword at
                // dword column win/2 + L, so a 32b read over Kc as a DWORD surface lands the pair
                // in lane L and the parity is a half-select. Each window is read twice (once per
                // parity), still far cheaper than the page dequant. Width/pitch stay in bytes, x is
                // in dwords.
                if (kc_dword_ok) {
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb) {
                    uint kw[DPAS_ROWS];
                    intel_sub_group_2d_block_read_32b_8r16x1c(
                        (global void *)Kc_b2d, KcD_w_b2d, KcD_h, KcD_p,
                        (int2)(KcD_x0_dw + PA_K_U4_WIN(db) / 2,
                               key_base + mb * DPAS_ROWS - past_len),
                        (private uint *)&kw[0]);
                    #pragma unroll
                    for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset)
                        k_raw[mb][key_offset] = PA_K_U4_PAR(db) ? (ushort)(kw[key_offset] >> 16)
                                                               : (ushort)kw[key_offset];
                }
                } else {
                    // A dword block read cannot represent an odd half offset from the aligned surface
                    // origin. Stay on the exact Kc source and gather the permuted channels directly;
                    // falling back to the cache here would reintroduce quantization error and would use
                    // page addressing with a potentially non-page-aligned k0.
                    const int current_head = PA_K_U4_CHANNEL(db, lane);
                    #pragma unroll
                    for (int mb = 0; mb < kq_key_blocks; ++mb) {
                        k_raw[mb] = (ushort8)0;
                        #pragma unroll
                        for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
                            const int key = key_base + mb * DPAS_ROWS + key_offset;
                            if (current_head < d && key < k0 + k_chunk) {
                                k_raw[mb][key_offset] =
                                    as_ushort(Kc[(size_t)(key - past_len) * ldk + current_head]);
                            }
                        }
                    }
                }
        #else
                // f16 / i8 cache: no depth permutation, so this is the plain-SDPA [key, head] read
                // verbatim, just pointed at Kc with a (key - past_len) row origin. Rows past q read as
                // zero, which is what the `key < k` masking already assumes.
                #pragma unroll
                for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                    intel_sub_group_2d_block_read_16b_16r16x1c(
                        (global void *)Kc_b2d, KcD_w_b2d, KcD_h, KcD_p,
                        (int2)(KcD_x0 + db * DPAS_K, key_base + kg * SUBGROUP_SIZE - past_len),
                        (private ushort *)&k_raw[kg * (SUBGROUP_SIZE / DPAS_ROWS)]);
                }
        #endif
            }
    #endif
#elif USE_2D_BLOCK_IO_K_I8
            // int8 K via the 8-bit VNNI-transform read: row-major [key, head] read at (x = db *
            // DPAS_K, y = key_base) gives lane = head with 4 consecutive keys per uint, no shuffle.
            // One read spans 32 keys, of which this subgroup uses kq_sg_tile_keys (kq_sg_tile_keys
            // / 4 uints).
            {
                uint kt[8];
                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                    (global void *)K, KD_w, KD_h, KD_p,
                    (int2)(db * DPAS_K, key_base), (private uint *)&kt[0]);
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb)
                    k_raw[mb] = (ushort8)0;
                // Bias-trick dequant, all in half: extract each key byte with shift+mask (as_char4
                // would cost a :b deinterleave), widen as as_half(0x6480 ^ byte) == byte + 1152,
                // then subtract the folded (zp + 1152) and multiply by the scale.
                #pragma unroll
                for (int u = 0; u < kq_sg_tile_keys / 4; ++u) {
                    const uint w = kt[u];
                    #pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int krel = u * 4 + bb;           // key's subgroup-local index 0..kq_sg_tile_keys-1
#if INPUT0_IS_BF16
                        const float wide = convert_float(as_char((uchar)((w >> (bb * 8)) & 0xFFu)));
                        const float k_sc = convert_float(sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                            const float k_zpb = convert_float(sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                            const float deq_k = (wide - k_zpb) * k_sc;
                        k_raw[krel / 8][krel % 8] = _convert_bfloat16_as_ushort(deq_k);
#else
                        const ushort wbits = (ushort)0x6480 ^ (ushort)((w >> (bb * 8)) & 0xFFu);
                        const half wide = as_half(wbits);
                        const half k_sc = sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                            const half k_zpb = sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                            const half deq_k = (wide - k_zpb) * k_sc;
                        k_raw[krel / 8][krel % 8] = as_ushort(deq_k);
#endif
                    }
                }
            }
#elif USE_2D_BLOCK_IO_KV
            // The _16r builtin returns 16 key rows (2 row-blocks), so issue one read per 16-key
            // group: with kq_sg_tile_keys == 32 a single read would leave k_raw[2..3]
            // uninitialised.
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                intel_sub_group_2d_block_read_16b_16r16x1c(
                    (global void *)K_b2d, KD_w_b2d, KD_h, KD_p,
                    (int2)(KD_x0 + db * DPAS_K, key_base + kg * SUBGROUP_SIZE),
                    (private ushort *)&k_raw[kg * (SUBGROUP_SIZE / DPAS_ROWS)]);
            }
#else
            const int head = db * DPAS_K + lane;
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                k_raw[mb] = (ushort8)0;
                #pragma unroll
                for (int key_offset = 0; key_offset < 8; ++key_offset) {
                    const int key = key_base + mb * 8 + key_offset;
                    #ifdef KV_COMPRESSED
                        // i8 compressed K, per-token asymmetric dequant from the hoisted scale/zp.
                        // sub_group_broadcast is a collective, so it must stay outside the per-lane
                        // (head < d) guard; krel is a constant, so it folds.
                        const int krel = mb * 8 + key_offset;
                        const float k_sc = convert_float(sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                            #if INPUT0_IS_BF16
                            const float k_zp = convert_float(sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                            #else
                            // k_zpb_lane holds zp+1152.0h; recover the raw zp for this scalar path.
                            const float k_zp = convert_float(sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE)) - 1152.0f;
                            #endif
                    #endif
                    if (head < d && key < k) {
                        #ifdef KV_COMPRESSED
                            const float deq_k = (convert_float(K[(size_t)key * ldk + head]) - k_zp) * k_sc;
                            k_raw[mb][key_offset] = DT_BITS_FROM_F32(deq_k);
                        #else
                            k_raw[mb][key_offset] = as_ushort(K[(size_t)key * ldk + head]);
                        #endif
                    }
                }
            }
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
            // Full 2D mask [query x key]: each lane loads its own query row, pre-scaled by iscale,
            // so the max loop only adds (sdpa_micro's tile_load_t + unscale). MASK_IS_FULL_2D is a
            // compile-time kind, but for a dynamic mask the host infers kind 2 from the stage, so a
            // [B, H, 1, K] per-key mask can arrive here: clamp its query row to 0 (every query row
            // IS row 0), or the read walks past the single row (OOB -> CL_OUT_OF_RESOURCES or a NaN
            // mask). The selects fold when MSK_D2/MSK_D3 are literals.
            float16 mask_full[kq_query_blocks][kq_sg_tile_keys / SUBGROUP_SIZE];
            if (MASK_IS_FULL_2D) {
                #pragma unroll
                for (int qb = 0; qb < kq_query_blocks; ++qb) {
                    const int mask_query = (MSK_D2 == 1) ? 0
                                                         : (wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane);
                    #pragma unroll
                    for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
                        const int mask_key = key_base + ii * SUBGROUP_SIZE;
                        half16 mv = (half16)0.0f;
                        if (mask_query < MSK_D2) {
                            // Same 1-row guard for the KEY side: a [B, H, q, 1] broadcast mask
                            // compiled as kind 2 would read key columns past its single column.
                            if (MSK_D3 == 1) {
                                mv = (half16)msk[MSK_OFF(0, 0, mask_query, 0)];
                            } else if (mask_key + SUBGROUP_SIZE <= MSK_D3) {
                                mv = vload16(0, msk + MSK_OFF(0, 0, mask_query, mask_key));
                            } else {
                                #pragma unroll
                                for (int kk = 0; kk < SUBGROUP_SIZE; ++kk) {
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
        // Start Vc fetches before softmax without retaining a private payload. One query
        // partition covers all value columns, so it is enough to prefetch each tile once.
        if (!from_cache && sg_i_sv == 0) {
            #pragma unroll
            for (int cp = 0; cp < sv_key_blocks; ++cp) {
                if (cp * SUBGROUP_SIZE < k_chunk) {
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        if (sg_j0_sv + cd * SUBGROUP_SIZE < dv) {
                            intel_sub_group_2d_block_prefetch_16b_16r16x1c(
                                (const global void *)Vc_b2d, VcD_w_b2d, VcD_h, VcD_p,
                                (int2)(VcD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE,
                                       k0 + cp * SUBGROUP_SIZE - past_len));
                        }
                    }
                }
            }
        }
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
#if IS_PA_MIXED
    #if PA_CUR_KV_F16
            // The split at past_len makes the V source workgroup-uniform as well as subgroup-uniform.
            const bool v_from_cache = (k0 < past_len);
    #else
            const bool v_from_cache = true;
    #endif
#endif
            #if USE_2D_BLOCK_IO_V_I8
                // One _8b_32r16x1c read covers 16 value columns and 32 key rows, i.e. two cp
                // blocks, so it is issued on even cp only and serves both (uints 0..3 for cp, 4..7
                // for cp + 1); cp is an unroll constant, so the selection folds. Issued ahead of
                // the pA (S_slm) reads so the global latency overlaps the SLM traffic. Columns past
                // dv read as 0 and the store drops them.
                    const bool vt_do_read = ((cp & 1) == 0);
                    const int vt_half = (cp & 1) * 4;
                if (vt_do_read) {
                    // The x2c / x4c variants fetch the subgroup's 32 / 64 value columns in one
                    // message, into the same block-major layout the x1c loop writes into &vt[cd *
                    // 8], so the dequant indexing is shared. coord.x must be a multiple of 4 for
                    // 8-bit data, which sg_j0_sv is.
                    #if sv_value_blocks == 2
                        intel_sub_group_2d_block_read_transform_8b_32r16x2c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE),
                            (private uint *)&vt[0]);
                    #elif sv_value_blocks == 4
                        intel_sub_group_2d_block_read_transform_8b_32r16x4c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE),
                            (private uint *)&vt[0]);
                    #else
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                                (global void *)V, VD_w, VD_h, VD_p,
                                (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE),
                                (private uint *)&vt[cd * 8]);
                        }
                    #endif
                }
            #endif

            short8 pA[sv_score_blocks];
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                const int query0 = sg_i0_sv + r * 8;
                pA[r] = as_short8(intel_sub_group_block_read_us8(
                    (local void *)&S_slm[((cp * kq_wg_tile_queries + query0) * SUBGROUP_SIZE) >> 1]));
            }

            #if USE_2D_BLOCK_IO_V_I8
                // Per-token V scale depends only on the key, and pA is already lane = key, so the
                // scale folds into pA with a per-lane multiply instead of being broadcast across
                // V's head-dim lanes. zp is a subtraction, so it stays on the V side (broadcast per
                // key there).
                const int vs_key = k0 + cp * SUBGROUP_SIZE + lane;
                const uint vs_co = v_comp_base + VAL_COMP_OFF(0, 0, vs_key, 0);
                // Keep scale/zp in half: V_scales/V_zp are already half, and the dequant is
                // stored as half — half arithmetic is bit-identical to the float path over the
                // int8 range (verified), so this avoids the half->float->half round trips.
                const half vs_c = (vs_key < k) ? V_scales[vs_co] : (half)0.0f;
                    #if INPUT0_IS_BF16
                    const half vzb_c = (vs_key < k) ? convert_half(V_zp[vs_co]) : (half)0.0f;
                    #else
                    // Fold the bias-trick widen bias (+1152.0h) into zp: the V dequant below widens
                    // via as_half(0x6480 ^ byte) (== signed_byte + 1152), so subtracting (zp+1152)
                    // gives (signed_byte - zp) with no convert_half widen. OOB keys -> vzb_c=1152
                    // (zp=0), and the score-side scale (vs_c=0 for OOB) still zeroes the product.
                    const half vzb_c = (vs_key < k) ? (convert_half(V_zp[vs_co]) + (half)1152.0h) : (half)1152.0h;
                    #endif

                #pragma unroll
                for (int r = 0; r < sv_score_blocks; ++r)
#if INPUT0_IS_BF16
                    pA[r] = as_short8(_convert_bfloat168_as_ushort8(
                        _convert_as_bfloat168_float8(as_ushort8(pA[r])) * convert_float(vs_c)));
#else
                    pA[r] = as_short8(as_half8(pA[r]) * vs_c);
#endif
            #elif IS_PA_KV_COMPRESSED && IS_PA_MIXED
                // Same scale/zp split as the plain-SDPA i8 path, with the per-key scale/zp read
                // from the page's comp region (PA_V_COMP_OFF: [token], [block_size + token]); a cp
                // block is one page, so lane == token. OOB keys get scale 0. Cache tiles only: a
                // PA_CUR_KV_F16 tile reads plain f16 Vc, so v_zp_c stays 0.
                half v_zp_c = (half)0.0f;
                if (v_from_cache) {
                    const int vs_key_pa = k0 + cp * SUBGROUP_SIZE + lane;
                    const size_t vs_page_pa =
                        PA_V_PAGE_OFF((vs_key_pa < k) ? block_indices[base_block_index + vs_key_pa / PAGED_ATTENTION_BLOCK_SIZE] : 0u, b0_kv);
                    const global half *v_comp_pa =
                        (const global half *)(V + vs_page_pa + PA_V_COMP_OFF);
                    const int vs_tok_pa = vs_key_pa % PAGED_ATTENTION_BLOCK_SIZE;
                    const half vs_c_pa = (vs_key_pa < k) ? v_comp_pa[vs_tok_pa] : (half)0.0f;
                    v_zp_c = (vs_key_pa < k) ? v_comp_pa[PAGED_ATTENTION_BLOCK_SIZE + vs_tok_pa]
                                             : (half)0.0f;

                    #pragma unroll
                    for (int r = 0; r < sv_score_blocks; ++r)
                        pA[r] = as_short8(as_half8(pA[r]) * vs_c_pa);
                }
            #endif

            int8 vb[sv_value_blocks];
            #if IS_PA_MIXED
            if (v_from_cache) {
                // A cp block is SUBGROUP_SIZE (== DPAS_K == PAGED_ATTENTION_BLOCK_SIZE) keys
                // starting at a multiple of kq_wg_tile_keys, i.e. exactly one cache page, so the
                // page lookup is hoisted out of the cd and key_pair loops.
                const int cp_key0 = k0 + cp * SUBGROUP_SIZE;
                // Page stride is PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE: the comp arrays
                // follow the data rows, so the data row pitch stays PA_V_ROW_ELEMS (V_HEAD_SIZE
                // except for u4).
                const size_t v_page_base =
                    PA_V_PAGE_OFF((cp_key0 < k) ? block_indices[base_block_index + cp_key0 / PAGED_ATTENTION_BLOCK_SIZE] : 0u, b0_kv);
                #if IS_PA_KV_COMPRESSED
                    // Compressed V page: lane == token for the per-key comp and key_rel for the
                    // dequant. The scale is already folded into pA, so only the zp subtraction
                    // happens here (broadcast per key).
                    #if USE_2D_BLOCK_IO_V_PA_I8
                    {
                        // The data region is a [PAGED_ATTENTION_BLOCK_SIZE tokens, PA_V_ROW_ELEMS]
                        // byte tile whose pitch passes the host's block2d rule. The 8b transform is
                        // 32-row only on Xe2 while a page has 16 tokens, so the height is clamped
                        // and uints 0..3 are used; two cp blocks cannot share a read (pages not
                        // adjacent).
                        const int vp_rows = PA_PAGE_ROWS(k, cp_key0);
                        uint vt_pa[8 * sv_value_blocks];
                        if (vp_rows > 0) {
                            #if IS_PA_K_U4
                            // One byte per value PAIR, so the row is PA_V_ROW_ELEMS bytes wide.
                            const int VP_w = PA_V_ROW_ELEMS;
                            const int VP_p = PA_V_ROW_ELEMS;
                            #else
                            const int VP_w = dv;                  // bytes: i8, one byte per value
                            const int VP_p = V_HEAD_SIZE;          // bytes: data row pitch, NOT ADJUSTED
                            #endif
                            #pragma unroll
                            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                                const int vcol = sg_j0_sv + cd * SUBGROUP_SIZE;
                                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                                    (global void *)(V + v_page_base), VP_w, vp_rows, VP_p,
                                    // u4 folds the upper half of the head dim back onto its low twin;
                                    // the nibble select below picks which one this tile wants. Both
                                    // the base and PA_V_ROW_ELEMS are multiples of SUBGROUP_SIZE, so a
                                    // 16-lane tile never straddles the split. Identity for i8.
                                    (int2)(PA_V_U4_COL(vcol), 0),
                                    (private uint *)&vt_pa[cd * 8]);
                            }
                        } else {
                            #pragma unroll
                            for (int u = 0; u < 8 * sv_value_blocks; ++u)
                                vt_pa[u] = 0u;
                        }
                        // zp broadcasts are per-key and independent of the value index, so hoist them
                        // out of the cd loop (once per cp block instead of once per (cd, u) pair).
                        half4 vzp4[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int k0r = u * 4;
                            vzp4[u] = (half4)(sub_group_broadcast(v_zp_c, k0r + 0),
                                              sub_group_broadcast(v_zp_c, k0r + 1),
                                              sub_group_broadcast(v_zp_c, k0r + 2),
                                              sub_group_broadcast(v_zp_c, k0r + 3));
                        }
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            #if IS_PA_K_U4
                            // Which nibble this tile's head dims live in. Uniform across the subgroup
                            // (the split point is a multiple of SUBGROUP_SIZE), so it folds into the
                            // shift amount rather than a per-lane select.
                            const int v_hi = PA_V_U4_HI(sg_j0_sv + cd * SUBGROUP_SIZE);
                            #endif
                            #pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const uint w = vt_pa[cd * 8 + u];
                                // Each uint packs 4 consecutive tokens as signed bytes, token u*4+b
                                // in byte b -- the same packing the plain-SDPA i8 V path decodes.
                                #if IS_PA_K_U4
                                const half4 q4 = (half4)((half)((w >> (v_hi ?  4 :  0)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 12 :  8)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 20 : 16)) & 0x0Fu),
                                                         (half)((w >> (v_hi ? 28 : 24)) & 0x0Fu));
                                #else
                                const half4 q4 = (half4)((half)(char)((w >>  0) & 0xFFu),
                                                         (half)(char)((w >>  8) & 0xFFu),
                                                         (half)(char)((w >> 16) & 0xFFu),
                                                         (half)(char)((w >> 24) & 0xFFu));
                                #endif
                                const half4 deq4 = q4 - vzp4[u];
                                // f16 VNNI operand: vb[cd][key_pair] packs keys (2*kp, 2*kp+1), and
                                // deq4 already holds keys u*4..u*4+3 in order, so .lo/.hi are exactly
                                // key_pairs (u*2, u*2+1).
                                vb[cd][u * 2 + 0] = as_int(deq4.lo);
                                vb[cd][u * 2 + 1] = as_int(deq4.hi);
                            }
                        }
                    }
                    #elif USE_1D_BLOCK_IO_V_PA_U4
                    {
                        // Same dequant and vb writes as the scalar branch below, only the load
                        // differs, so SDPA_OCL_V_PA_1D=0 bisects the read alone. The column group
                        // comes from sg_j0_sv (not a constant), so the base is biased by it and the
                        // index taken at c = 0 (see PA_PAGE_*); that makes the read per cd, which
                        // costs nothing because sv_value_blocks is 1 for the head sizes this path
                        // fires on.
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            vb[cd] = (int8)0;
                            const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                            const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
                            // Nibble select as a uniform shift amount: v_hi is subgroup-uniform but
                            // not a compile-time constant, and the `?:` form would cost a select on
                            // every element.
                            const uint v_sh = PA_V_U4_HI(v_base) ? 4u : 0u;
                            uchar16 v_pg[PA_PAGE_READS(PA_V_ROW_ELEMS)];
                            const global uchar *v_pg_base =
                                (const global uchar *)(V + v_page_base) + PA_V_U4_COL(v_base);
                            #pragma unroll
                            for (int r = 0; r < PA_PAGE_READS(PA_V_ROW_ELEMS); ++r)
                                v_pg[r] = intel_sub_group_block_read_uc16(v_pg_base + r * PA_PAGE_RD_BYTES);
                            // No per-key `key < k` guard (as in the block2d branch): a key at/past
                            // k has a probability of exactly 0, so its V value only has to be
                            // finite -- a nibble is, and v_zp_c is 0 there.
                            if (value < dv) {
                                #pragma unroll
                                for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                                    // The token index IS the key's block-local index (the cp block
                                    // is one page), spelled as the loop constant because
                                    // PA_PAGE_R/I need it at compile time.
                                    const int t0 = key_pair * 2;
                                    const int t1 = t0 + 1;
                                    const uint vb0 = (uint)v_pg[PA_PAGE_R(PA_V_ROW_ELEMS, t0, 0)]
                                                               [PA_PAGE_I(PA_V_ROW_ELEMS, t0, 0)];
                                    const uint vb1 = (uint)v_pg[PA_PAGE_R(PA_V_ROW_ELEMS, t1, 0)]
                                                               [PA_PAGE_I(PA_V_ROW_ELEMS, t1, 0)];
                                    half2 vv;
                                    vv[0] = (half)((vb0 >> v_sh) & 0x0Fu) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    vv[1] = (half)((vb1 >> v_sh) & 0x0Fu) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    vb[cd][key_pair] = as_int(vv);
                                }
                            }
                        }
                    }
                    #else
                    // Scalar-gather fallback for the compressed V page (SDPA_OCL_V_PA_I8_2D=0, or a
                    // pitch that fails the block2d rule): same dequant, one message per value per
                    // key pair.
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        vb[cd] = (int8)0;
                        const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                        #if IS_PA_K_U4
                        // Two head dims share a byte, so the address is the folded byte column plus
                        // the lane; the nibble is the TILE's, hence uniform across the subgroup.
                        const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
                        const int v_hi = PA_V_U4_HI(v_base);
                        const int v_addr = PA_V_U4_COL(v_base) + (int)lane;
                        #endif
                        if (value < dv) {
                            #pragma unroll
                            for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                                const int key0 = cp_key0 + key_pair * 2;
                                const int key1 = key0 + 1;
                                half2 vv = (half2)0.0h;
                                if (key0 < k) {
                                    const int t0 = key0 % PAGED_ATTENTION_BLOCK_SIZE;
                                    #if IS_PA_K_U4
                                    const uint vb0 = (uint)(uchar)V[v_page_base + (size_t)t0 * PA_V_ROW_ELEMS + v_addr];
                                    vv[0] = (half)U4_NIBBLE_SEL(vb0, v_hi) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    #else
                                    vv[0] = (half)(char)V[v_page_base + (size_t)t0 * V_HEAD_SIZE + value] -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 0);
                                    #endif
                                }
                                if (key1 < k) {
                                    const int t1 = key1 % PAGED_ATTENTION_BLOCK_SIZE;
                                    #if IS_PA_K_U4
                                    const uint vb1 = (uint)(uchar)V[v_page_base + (size_t)t1 * PA_V_ROW_ELEMS + v_addr];
                                    vv[1] = (half)U4_NIBBLE_SEL(vb1, v_hi) -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    #else
                                    vv[1] = (half)(char)V[v_page_base + (size_t)t1 * V_HEAD_SIZE + value] -
                                            sub_group_broadcast(v_zp_c, key_pair * 2 + 1);
                                    #endif
                                }
                                vb[cd][key_pair] = as_int(vv);
                            }
                        }
                    }
                    #endif
                #elif USE_2D_BLOCK_IO_V_PA
                    // f16 V page: a [PAGED_ATTENTION_BLOCK_SIZE tokens, V_HEAD_SIZE] row-major
                    // tile, so the 16b VNNI-transform read applies with the page as the surface and
                    // V_HEAD_SIZE as the pitch. The height is clamped to the tokens the page holds
                    // (unwritten slots could be NaN, which would survive the zero score); a block
                    // entirely at/past k (height <= 0 is not a legal read) is zero-filled --
                    // reachable on the last k0 tile, and cp_key0 is constant only in cp, so this
                    // stays a real (uniform) branch.
                    const int vp_rows = PA_PAGE_ROWS(k, cp_key0);
                    if (vp_rows > 0) {
                        const global half *Vp = (const global half *)(V + v_page_base);
                        const int VP_w = dv * (int)sizeof(half);
                        const int VP_p = V_HEAD_SIZE * (int)sizeof(half);
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd) {
                            intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                                (global void *)Vp, VP_w, vp_rows, VP_p,
                                (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, 0), (private uint *)&vb[cd]);
                        }
                    } else {
                        #pragma unroll
                        for (int cd = 0; cd < sv_value_blocks; ++cd)
                            vb[cd] = (int8)0;
                    }
                #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    vb[cd] = (int8)0;
                    const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                    if (value < dv) {
                        #pragma unroll
                        for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                            const int key0 = cp_key0 + key_pair * 2;
                            const int key1 = key0 + 1;
                            half2 vv = (half2)0.0h;
                            if (key0 < k) {
                                vv[0] = V[v_page_base +
                                          (size_t)(key0 % PAGED_ATTENTION_BLOCK_SIZE) * V_HEAD_SIZE + value];
                            }
                            if (key1 < k) {
                                vv[1] = V[v_page_base +
                                          (size_t)(key1 % PAGED_ATTENTION_BLOCK_SIZE) * V_HEAD_SIZE + value];
                            }
                            vb[cd][key_pair] = as_int(vv);
                        }
                    }
                }
                #endif
            }
            #if PA_CUR_KV_F16
            else {
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)Vc_b2d, VcD_w_b2d, VcD_h, VcD_p,
                        (int2)(VcD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE,
                               k0 + cp * SUBGROUP_SIZE - past_len),
                        (private uint *)&vb[cd]);
                }
            }
            #endif
            #elif USE_2D_BLOCK_IO_V_I8
                // int8 V: the paired read above gives a 32-key x 16-value tile (lane = value, 4
                // keys per uint); this cp block uses the 4 uints at vt_half. Dequant each byte and
                // repack into the f16 VNNI operand (two keys per int), with no subgroup shuffle.
                {
                    // Bias-trick dequant (as on the K side): shift+mask byte extract, widen as
                    // as_half(0x6480 ^ byte) == byte + 1152, subtract the folded zp + 1152 (the
                    // scale is already in pA). The zp broadcasts depend on the key only, so they
                    // are hoisted out of the cd loop.
                        half4 zpb4[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int k0r = u * 4;
                            zpb4[u] = (half4)(sub_group_broadcast(vzb_c, k0r + 0),
                                              sub_group_broadcast(vzb_c, k0r + 1),
                                              sub_group_broadcast(vzb_c, k0r + 2),
                                              sub_group_broadcast(vzb_c, k0r + 3));
                        }
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint w = vt[cd * 8 + vt_half + u];
#if INPUT0_IS_BF16
                            const float4 wide4 = (float4)(convert_float(as_char((uchar)((w >>  0) & 0xFFu))),
                                                          convert_float(as_char((uchar)((w >>  8) & 0xFFu))),
                                                          convert_float(as_char((uchar)((w >> 16) & 0xFFu))),
                                                          convert_float(as_char((uchar)((w >> 24) & 0xFFu))));
                                const float4 deq4 = wide4 - convert_float4(zpb4[u]);
                            const ushort4 enc4 = _convert_bfloat164_as_ushort4(deq4);
                            vb[cd][u * 2 + 0] = as_int(enc4.lo);
                            vb[cd][u * 2 + 1] = as_int(enc4.hi);
#else
                            const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >>  8) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 16) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
                                const half4 deq4 = wide4 - zpb4[u];
                            // f16 VNNI operand: vb[cd][key_pair] packs keys (2*key_pair,
                            // 2*key_pair+1), which are exactly deq4.lo / .hi for key_pairs (u*2,
                            // u*2+1).
                            vb[cd][u * 2 + 0] = as_int(deq4.lo);
                            vb[cd][u * 2 + 1] = as_int(deq4.hi);
#endif
                        }
                    }
                }
            #elif USE_2D_BLOCK_IO_KV
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)V_b2d, VD_w_b2d, VD_h, VD_p,
                        (int2)(VD_x0 + sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE),
                        (private uint *)&vb[cd]);
                }
            #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    vb[cd] = (int8)0;
                    const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                    if (value < dv) {
                        #pragma unroll
                        for (int key_pair = 0; key_pair < 8; ++key_pair) {
                            const int key0 = k0 + cp * SUBGROUP_SIZE + key_pair * 2;
                            const int key1 = key0 + 1;
                            DT_ELEM2_T vv = DT_ELEM2_ZERO;
                            if (key0 < k) {
                                #ifdef KV_COMPRESSED
                                    // i8 compressed V: per-token (per-kv-head) asymmetric dequant.
                                    // Scale/zp vary per key (token), so they must be indexed by
                                    // key0/key1 here, not by the value (head-dim) index.
                                    const uint v_comp_off0 = VAL_COMP_OFF(b1, b0_kv, key0, 0);
                                    vv[0] = DT_FROM_F32((convert_float(V[(size_t)key0 * ldv + value]) - convert_float(V_zp[v_comp_off0])) * convert_float(V_scales[v_comp_off0]));
                                #else
                                    vv[0] = DT_FROM_RAW(V[(size_t)key0 * ldv + value]);
                                #endif
                            }
                            if (key1 < k) {
                                #ifdef KV_COMPRESSED
                                    const uint v_comp_off1 = VAL_COMP_OFF(b1, b0_kv, key1, 0);
                                    vv[1] = DT_FROM_F32((convert_float(V[(size_t)key1 * ldv + value]) - convert_float(V_zp[v_comp_off1])) * convert_float(V_scales[v_comp_off1]));
                                #else
                                    vv[1] = DT_FROM_RAW(V[(size_t)key1 * ldv + value]);
                                #endif
                            }
                            vb[cd][key_pair] = as_int(vv);
                        }
                    }
                }
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
