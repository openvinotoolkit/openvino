#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io               : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                         : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                   : enable

#include "include/batch_headers/sdpa_utils.cl"

float __builtin_IB_atomic_max_local_f32(__local float *, float);

#define kq_wg_tile_keys      (kq_sg_tile_keys * kq_sg_per_wg_keys)
#define kq_wg_tile_queries   (kq_sg_tile_queries * kq_sg_per_wg_queries)
#define kq_key_blocks        (kq_sg_tile_keys / DPAS_ROWS)
#define kq_query_blocks      (kq_sg_tile_queries / SUBGROUP_SIZE)

#define sg_per_wg (kq_sg_per_wg_keys * kq_sg_per_wg_queries)

#define sv_score_blocks      (sv_sg_tile_scores / DPAS_ROWS)
#define sv_value_blocks      (sv_sg_tile_values / SUBGROUP_SIZE)
#define sv_key_blocks        (kq_wg_tile_keys / DPAS_K)
#define q_blocks             (kq_wg_tile_queries / SUBGROUP_SIZE)

// Mask-kind predicates. When the host proved the mask shape at compile time
// (MASK_KIND in {0,1,2}) these fold to compile-time constants so IGC drops the
// dead mask branches; MASK_KIND == -1 keeps the original runtime MSK_D2/MSK_D3
// checks. 2 = full 2D [q>1,k>1], 1 = per-key [q==1,k>1], 0 = scalar/broadcast.
#if MASK_KIND == -1
#  define MASK_IS_PER_KEY  (MSK_D2 == 1 && MSK_D3 > 1)
#  define MASK_IS_FULL_2D  (MSK_D2 > 1 && MSK_D3 > 1)
#else
#  define MASK_IS_PER_KEY  (MASK_KIND == 1)
#  define MASK_IS_FULL_2D  (MASK_KIND == 2)
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)))
KERNEL(sdpa_ocl)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
        global half *A,
#if IS_PAGED_ATTENTION
        const __global INPUT3_TYPE* subsequence_begins,
    #if !IS_PREFILL
        const __global INPUT3_TYPE* past_lens,
        const __global INPUT3_TYPE* block_indices,
        const __global INPUT3_TYPE* block_indices_begins,
    #endif
#endif
#if WITH_ATTN_MASK
        const global half *msk,
#endif
#if WITH_SCALE
        global SCALE_DATA_T *scale_ptr,
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
    #if KEY_ZERO_POINTS
        , const global KEY_ATTR_ZP_DATA_T *K_zp
    #endif
        , const global VAL_ATTR_SCALES_DATA_T *V_scales
    #if VAL_ZERO_POINTS
        , const global VAL_ATTR_ZP_DATA_T *V_zp
    #endif
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
    #if HAS_QQ_BIAS
        const uint qq_bias_num = qq_bias_begins[gws_mapping + 1] - qq_bias_begins[gws_mapping];
        const uint cumulated_spec_num = qq_bias_begins[gws_mapping];
    #endif
    #if IS_PREFILL
        const int past_len = 0;
        const int k = q;
    #else
        const int past_len = past_lens[gws_mapping];
        const int k = q + past_len;
    #endif
    const int d = HEAD_SIZE;
#endif

    const size_t lane  = get_sub_group_local_id();
    const size_t sg_ij = get_local_id(1);
#if IS_PAGED_ATTENTION
    // Query blocks are handed out through blocked_indexes_start_and_gws_mapping, so the block this
    // workgroup owns is not a plain function of get_group_id(0): it is the block start recorded in
    // that buffer, expressed relative to the beginning of its own subsequence.
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
            float iscale = sqrt(convert_float(HEAD_SIZE));
            float scale = native_recip(iscale);
        #endif
    #endif

    scale *= LOG2E;

    /* Row stride (in elements) of the Q/K/V/A matrices. */
#if IS_PAGED_ATTENTION
    // Paged attention Q/K/V/output are 2D [total_tokens, num_heads * head_size]: there is no Y
    // dimension, so the generic QRY_S2/KEY_S2/VAL_S2/DST_S2 (Y pitch) macros are all 0 here and the
    // token stride has to be derived from the head layout instead.
    const uint ldq = HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint ldk = HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    const uint lda = HEAD_SIZE * HEADS_NUM;
#else
    const uint ldq = QRY_S2;
    const uint ldk = KEY_S2;
    const uint ldv = VAL_S2;
    const uint lda = DST_S2;
#endif

#if IS_PAGED_ATTENTION
    // Tokens of all subsequences are packed into one matrix, so a batch index does not exist:
    // seek to the first token of this workgroup's subsequence and to this head's column slice.
    Q += (size_t)subsequence_begin * ldq + b0 * HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM;
    K += (size_t)subsequence_begin * ldk + b0_kv * HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    V += (size_t)subsequence_begin * ldv + b0_kv * HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    A += (size_t)subsequence_begin * lda + b0 * HEAD_SIZE;
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
    const int VD_w = d * (int)sizeof(VAL_DATA_T), VD_h = k, VD_p = (int)ldv * (int)sizeof(VAL_DATA_T);
    const int AD_w = d * (int)sizeof(half), AD_h = q, AD_p = (int)lda * (int)sizeof(half);
    local uint  Q_slm[DKS * q_blocks * Q_DWORDS * SUBGROUP_SIZE];
    local uint  S_slm[kq_wg_tile_keys * kq_wg_tile_queries / 2];
    local float S_sum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
        S_max_slm[qi] = -INFINITY;

    // Cooperative Q->SLM staging: the Q tile is q_blocks query-blocks x DKS head-dim
    // chunks = q_blocks*DKS independent (q_block, db) tiles. Distribute them round-robin
    // across the workgroup subgroups so all subgroups load Q and every tile is staged even
    // when q_blocks*DKS exceeds sg_per_wg (e.g. D_MAX >= 256), shrinking the prologue
    // Q-load latency. The loop bound guarantees q_block < q_blocks, so no guard is needed.
    for (int tile = sg_ij; tile < q_blocks * DKS; tile += sg_per_wg) {
        const int q_block = tile / DKS;   // 0..q_blocks-1
        const int db      = tile % DKS;   // 0..DKS-1
        const int query_base = wg_j0 + q_block * SUBGROUP_SIZE;
        const int head_base = db * DPAS_K;
        uint8 q_pack;
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
        intel_sub_group_block_write8(
            (local uint *)&Q_slm[((db * q_blocks + q_block) * Q_DWORDS) * SUBGROUP_SIZE], q_pack);
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

    // Causal upper bound on the key loop. With IS_CAUSAL every key > query is masked to -inf, so a
    // workgroup owning queries [wg_j0, wg_j0 + kq_wg_tile_queries) can never need a key beyond
    // wg_j0 + kq_wg_tile_queries - 1. Without this bound the loop walks the FULL key range and
    // every k0 tile past the diagonal is pure waste: it loads K/V, runs the DPAS, then throws the
    // result away in the causal mask. At q = k = 1024 with kq_wg_tile_queries = 32 that is 256
    // tile-iterations instead of 144 (1.78x), and it is why sdpa_micro -- which has had this bound
    // since day one (causal_k = min(k, wg_j0 + wg_tile_n)) -- wins despite a less efficient
    // per-tile inner loop.
    // The non-causal case keeps the full range, and the bound is a no-op there.
#if IS_CAUSAL
    const int causal_k = min(k, (int)wg_j0 + kq_wg_tile_queries);
#else
    const int causal_k = k;
#endif

    // Sliding-window lower bound, the mirror of causal_k and the counterpart of sdpa_micro's
    // window_k0_begin. The mask below keeps only (query - SLIDING_WINDOW_SIZE, query], so the
    // smallest key this workgroup can need is for its smallest query, wg_j0:
    //   key > wg_j0 - SLIDING_WINDOW_SIZE  =>  first needed key = wg_j0 - SLIDING_WINDOW_SIZE + 1.
    // Every k0 tile below that is entirely outside the window and would be masked away wholesale,
    // exactly the waste causal_k removes at the top end. Round down to a k0 tile boundary so the
    // loop keeps its kq_wg_tile_keys stride and key_base stays tile-aligned (the 2D block reads
    // and the S_slm indexing both assume that).
#if IS_CAUSAL && SLIDING_WINDOW_SIZE
    const int window_k_begin = max(0, (int)wg_j0 - SLIDING_WINDOW_SIZE + 1);
    const int window_k0_begin = (window_k_begin / kq_wg_tile_keys) * kq_wg_tile_keys;
#else
    const int window_k0_begin = 0;
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

#ifdef KV_COMPRESSED
        // Per-token K scale/zp depend only on the key (shared across the head dim), so load them
        // ONCE per k0-tile with a subgroup-cooperative wide load (lane L -> key key_base+L) and
        // keep them in registers. The old code fetched K_scales[KEY_COMP_OFF(...)]/K_zp[...] inside
        // the db x mb x key_offset dequant loop; since that offset is key-only (db-independent) and
        // lane-uniform, IGC emitted a per-key SIMD-1 (1|M0) scalar load, reloaded every db -> ~128
        // such loads per k0 iter (measured in the GEN ISA). This collapses them to one 16-wide load
        // each (mirrors the V vs_c/vz_c pattern and the k_mask lane=key layout below).
        // scale/zp kept in HALF for the bias-trick dequant below. NOTE: half here does NOT hit
        // the GEN <2> widen penalty that made an earlier all-half K dequant slower — that penalty
        // is in the convert_float/(half) WIDEN of the int8 byte, which the bias trick eliminates
        // (it never widens via convert). zp folds the widen bias (+1152.0h) so the per-byte dequant
        // is just: reinterpret (0x6480 ^ byte) as half, subtract (zp+1152), multiply scale.
        half k_scale_lane[kq_sg_tile_keys / SUBGROUP_SIZE];
        #if KEY_ZERO_POINTS
        half k_zpb_lane[kq_sg_tile_keys / SUBGROUP_SIZE];   // zp + 1152.0h (bias-trick bias folded in)
        #endif
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
            const int sc_key = key_base + ii * SUBGROUP_SIZE + lane;
            const uint sc_off = k_comp_base + KEY_COMP_OFF(0, 0, sc_key, 0);
            k_scale_lane[ii] = (sc_key < k) ? convert_half(K_scales[sc_off]) : (half)0.0f;
            #if KEY_ZERO_POINTS
            k_zpb_lane[ii] = (sc_key < k) ? (convert_half(K_zp[sc_off]) + (half)1152.0h) : (half)1152.0h;
            #endif
        }
#endif

        #pragma unroll
        for (int db = 0; db < DKS; ++db) {
            int8 qB[kq_query_blocks];
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb) {
                const int q_block = sg_j0_kq / SUBGROUP_SIZE + qb;
                qB[qb] = as_int8(intel_sub_group_block_read8(
                    (local void *)&Q_slm[((db * q_blocks + q_block) * Q_DWORDS) * SUBGROUP_SIZE]));
            }

            ushort8 k_raw[kq_key_blocks];
#if USE_2D_BLOCK_IO_K_I8
            // int8 K via the 8-bit VNNI-transform read (same builtin as V). Reading K's row-major
            // [key, head] memory at (x=db*DPAS_K head-col, y=key_base) gives lane=head with each
            // uint packing 4 consecutive keys as bytes (GPU-probed: lane==head exactly, key order
            // u*4+b linear -> no subgroup shuffle, unlike the earlier non-transform K attempt).
            // One read spans 32 keys; this subgroup uses the first kq_sg_tile_keys (16) = uints 0..3.
            // Dequant reuses the hoisted per-key scale/zp broadcasts (Step 1); result is the f16
            // A operand k_raw[mb][key_offset], mb=krel/8, key_offset=krel%8.
            {
                uint kt[8];
                intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                    (global void *)K, KD_w, KD_h, KD_p,
                    (int2)(db * DPAS_K, key_base), (private uint *)&kt[0]);
                #pragma unroll
                for (int mb = 0; mb < kq_key_blocks; ++mb)
                    k_raw[mb] = (ushort8)0;
                // kq_sg_tile_keys keys = kq_sg_tile_keys/4 uints. head=64/128 => 16 keys => uints 0..3.
                // Bias-trick dequant (microbench-validated, mov -69% vs the convert_float widen):
                // extract each key byte with shift+mask (NO as_char4 -> no <4;1,0> :b deinterleave),
                // widen via the denormal-bias reinterpret (0x6480 ^ byte) as half, then the folded
                // (zp+1152) subtract and scale multiply -- all in half.
                #pragma unroll
                for (int u = 0; u < kq_sg_tile_keys / 4; ++u) {
                    const uint w = kt[u];
                    #pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int krel = u * 4 + bb;           // key's subgroup-local index 0..kq_sg_tile_keys-1
                        const ushort wbits = (ushort)0x6480 ^ (ushort)((w >> (bb * 8)) & 0xFFu);
                        const half wide = as_half(wbits);
                        const half k_sc = sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                        #if KEY_ZERO_POINTS
                            const half k_zpb = sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
                            const half deq_k = (wide - k_zpb) * k_sc;
                        #else
                            const half deq_k = (wide - (half)1152.0h) * k_sc;
                        #endif
                        k_raw[krel / 8][krel % 8] = as_ushort(deq_k);
                    }
                }
            }
#elif USE_2D_BLOCK_IO_KV
            // The _16r builtin returns exactly 16 key rows == 2 key-blocks of 8. A subgroup owns
            // kq_sg_tile_keys keys == kq_key_blocks blocks, so issue one read per 16-key group
            // instead of assuming a single read covers the whole tile: with kq_sg_tile_keys == 32
            // (kq_key_blocks == 4) a lone read filled only k_raw[0..1] and left k_raw[2..3]
            // uninitialised, silently corrupting S for the upper half of the tile.
            #pragma unroll
            for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
                intel_sub_group_2d_block_read_16b_16r16x1c(
                    (global void *)K, KD_w, KD_h, KD_p,
                    (int2)(db * DPAS_K, key_base + kg * SUBGROUP_SIZE),
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
                        // i8 compressed K: per-token (per-kv-head) asymmetric dequant. Scale/zp
                        // are the hoisted per-key values (lane=key wide load above); recover this
                        // key's scalar with a subgroup broadcast. krel is the key's subgroup-local
                        // index (compile-time constant here), so the broadcast folds to a register
                        // move. It is a subgroup collective, so it MUST run on all lanes -> keep it
                        // OUTSIDE the per-lane-divergent (head < d) guard below.
                        const int krel = mb * 8 + key_offset;
                        const float k_sc = convert_float(sub_group_broadcast(k_scale_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE));
                        #if KEY_ZERO_POINTS
                            // k_zpb_lane holds zp+1152.0h; recover the raw zp for this scalar path.
                            const float k_zp = convert_float(sub_group_broadcast(k_zpb_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE)) - 1152.0f;
                        #endif
                    #endif
                    if (head < d && key < k) {
                        #ifdef KV_COMPRESSED
                            #if KEY_ZERO_POINTS
                                const float deq_k = (convert_float(K[(size_t)key * ldk + head]) - k_zp) * k_sc;
                            #else
                                const float deq_k = convert_float(K[(size_t)key * ldk + head]) * k_sc;
                            #endif
                            k_raw[mb][key_offset] = as_ushort((half)deq_k);
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
                    S_tile[mb][qb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(k_raw[mb]), qB[qb], S_tile[mb][qb]);
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
            k_mask[ii] = (key < k) ? 0.0f : -INFINITY;
        }
        float2 mask_tile_float = convert_float2(mask_tile);
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            mask_tile_float[ii] = mask_tile_float[ii] * iscale;

        #if WITH_ATTN_MASK
            // Full 2D mask [query x key]: each lane loads its own query row (strided,
            // same access pattern as sdpa_micro's tile_load_t). Pre-scale by iscale at
            // load time and keep it as float so the softmax max-loop below only does a
            // branchless add (mirrors micro's tile_elementwise(unscale)+tile_binary add).
            float16 mask_full[kq_query_blocks][kq_sg_tile_keys / SUBGROUP_SIZE];
            if (MASK_IS_FULL_2D) {
                #pragma unroll
                for (int qb = 0; qb < kq_query_blocks; ++qb) {
                    const int mask_query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    #pragma unroll
                    for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii) {
                        const int mask_key = key_base + ii * SUBGROUP_SIZE;
                        half16 mv = (half16)0.0f;
                        if (mask_query < q) {
                            if (mask_key + SUBGROUP_SIZE <= k) {
                                mv = vload16(0, msk + MSK_OFF(0, 0, mask_query, mask_key));
                            } else {
                                #pragma unroll
                                for (int kk = 0; kk < SUBGROUP_SIZE; ++kk) {
                                    if (mask_key + kk < k)
                                        mv[kk] = msk[MSK_OFF(0, 0, mask_query, mask_key + kk)];
                                }
                            }
                        }
                        mask_full[qb][ii] = convert_float16(mv) * iscale;
                    }
                }
            }
        #endif

        float alpha[kq_query_blocks];
        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            float lmax = -INFINITY;
            // Whether this subgroup's (key x query) block can touch the causal boundary at all.
            // Keys run [key_base, key_base + kq_sg_tile_keys), queries run
            // [wg_j0 + sg_j0_kq + qb*SUBGROUP_SIZE, + SUBGROUP_SIZE), so if the block's LAST key is
            // <= its FIRST query every element is inside the causal region and the per-element
            // predicate is a no-op. Measured at q = k = 4096 with the default tiling: 96.6% of
            // blocks are in that class, yet the old code still issued cmp+sel for all 16 keys in
            // every one of them -- 32 sel + 8 cmp per (qb, k0) iteration, and the causal-mask
            // region was 16% of the whole loop body. sdpa_micro has always had this block-level
            // skip (its `if (causal_k_end > causal_q_begin)` guard around
            // tile_predicated_assignment_t); this is the ocl counterpart.
            // causal_block_clear is uniform across the subgroup (no lane term), so IGC turns the
            // branch into straight-line code for the common case rather than a per-lane select.
#if IS_CAUSAL
    #if BLOCK_SKIP_CAUSAL
            const int blk_key_last = key_base + kq_sg_tile_keys - 1;
            const int blk_query_first = (int)(wg_j0 + sg_j0_kq) + qb * SUBGROUP_SIZE;
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
    #else
            // BLOCK_SKIP_CAUSAL=0 keeps the original always-mask behaviour, so a wrong-result
            // config can be bisected against this optimisation.
            const bool causal_block_clear = false;
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
                    const int key = key_base + key_rel;
                    float s = S_tile[mb][qb][mm] + sub_group_broadcast(k_mask[mask_idx], mask_lane);
#ifdef STATIC_SCALAR_ATTN_MASK_VALUE
                    s += STATIC_SCALAR_ATTN_MASK_VALUE * iscale;
#endif
                    #if WITH_ATTN_MASK
                        if (MASK_IS_PER_KEY) {
                            s += sub_group_broadcast(mask_tile_float[mask_idx], mask_lane);
                        } else if (MASK_IS_FULL_2D) {
                            s += mask_full[qb][mask_idx][mask_lane];
                        } else if (query < q && key < k) {
                            const int mask_query = (MSK_D2 == 1) ? 0 : query;
                            const int mask_key = (MSK_D3 == 1) ? 0 : key;
                            s += convert_float(msk[MSK_OFF(0, 0, mask_query, mask_key)]) * iscale;
                        }
                    #endif
#if IS_CAUSAL
                    if (!causal_block_clear) {
    #if SLIDING_WINDOW_SIZE
                        // Keys outside (query - SLIDING_WINDOW_SIZE, query] are dropped, matching
                        // sdpa_micro's greater_than() predicate.
                        if (key > query || key <= query - SLIDING_WINDOW_SIZE) {
    #else
                        if (key > query) {
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

    #if MAX_BARRIER_V_PREFETCH && USE_2D_BLOCK_IO_KV
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                intel_sub_group_2d_block_prefetch_16b_16r16x1c(
                    (const global void *)V, VD_w, VD_h, VD_p,
                    (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE));
            }
        }
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    #else
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif

        #pragma unroll
        for (int qb = 0; qb < kq_query_blocks; ++qb) {
            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            const float m_new = S_max_slm[query];
            // Required when a query has no valid keys in the current prefix, e.g. future
            // remainder/causal/window masks or a fully masked row. In that case m_new is
            // -inf, and unguarded max rescaling would form -inf - -inf and poison S/A.
            const bool ok = isfinite(m_new);
            const float m_log2 = ok ? m_new * scale : 0.0f;
            const float a = ok ? native_exp2(S_max_tile[qb] - m_log2) : 1.0f;
            float lsum = 0.0f;

            S_max_tile[qb] = ok ? m_log2 : S_max_tile[qb];
            alpha[qb] = a;

            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
                float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                lsum += exp_tile[0] + exp_tile[1] + exp_tile[2] + exp_tile[3]
                      + exp_tile[4] + exp_tile[5] + exp_tile[6] + exp_tile[7];

                const int key = sg_i0_kq + mb * 8;
                const int key_block = key / SUBGROUP_SIZE;
                const int key_lane = key - key_block * SUBGROUP_SIZE;
                const int s_half_offset = (key_block * kq_wg_tile_queries + query) * SUBGROUP_SIZE + key_lane;
                vstore4(as_uint4(convert_half8(exp_tile)), 0, &S_slm[s_half_offset >> 1]);
            }
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
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
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr)
                    av[rr] = sub_group_broadcast(alpha[alpha_qb], alpha_lane0 + rr);
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= av;
            }
        }

        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        #if USE_2D_BLOCK_IO_V_I8
            // Declared outside the cp loop because with V_I8_PAIRED_READ one read serves two
            // consecutive cp blocks (see below). Unpaired, the def-use pattern is unchanged.
            uint vt[8 * sv_value_blocks];
        #endif
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
            #if USE_2D_BLOCK_IO_V_I8
                // One _8b_32r16x1c read covers a fixed 16 value columns, while a subgroup owns
                // sv_sg_tile_values == sv_value_blocks * SUBGROUP_SIZE of them, so issue one read
                // per cd, stepping x by SUBGROUP_SIZE. Reads are kept ahead of the S_slm (pA)
                // reads below so the global-memory latency overlaps with the SLM traffic.
                // Value columns past d only exist when d < D_MAX; the block read clamps them to 0
                // and the store guard (out_col < d) drops the corresponding A_tile columns.
                //
                // The builtin returns 32 key rows (uints 0..7, 4 rows each) but one cp block is
                // only SUBGROUP_SIZE == 16 keys, so a per-cp read discards half of every message
                // and consecutive cp reads overlap by 16 rows. V_I8_PAIRED_READ issues the read
                // on even cp only and lets it serve two blocks -- uints 0..3 for cp, uints 4..7
                // for cp+1 -- halving the V message count. cp is a full-unroll constant, so both
                // vt_do_read and vt_half fold at compile time: no branch and no dynamic vt index
                // survive. An odd sv_key_blocks needs no tail case; the last even cp simply uses
                // uints 0..3 exactly as the unpaired form does.
                #if V_I8_PAIRED_READ
                    const bool vt_do_read = ((cp & 1) == 0);
                    const int vt_half = (cp & 1) * 4;
                #else
                    const bool vt_do_read = true;
                    const int vt_half = 0;
                #endif
                if (vt_do_read) {
                    // The multi-block x2c / x4c variants fetch this subgroup's whole 32 / 64
                    // value columns in ONE message. GPU-probed on B580 (see
                    // test/microbench/probe_v_multiblock) their destination is BLOCK-MAJOR --
                    // uint u carries block u/8, key (u%8)*4+b, value (u/8)*16+lane -- which is
                    // bit-identical to what the x1c loop below writes into &vt[cd * 8], so the
                    // dequant indexing needs no change. coord.x must be a multiple of 4 for
                    // 8-bit data: sg_j0_sv is a multiple of sv_sg_tile_values (16/32/64), so
                    // that holds. The x1c loop stays as the fallback for any sv_value_blocks
                    // the extension has no single-message variant for.
                    #if V_I8_MULTIBLOCK_READ && sv_value_blocks == 2
                        intel_sub_group_2d_block_read_transform_8b_32r16x2c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE),
                            (private uint *)&vt[0]);
                    #elif V_I8_MULTIBLOCK_READ && sv_value_blocks == 4
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
                // int8 V uses the 8-bit VNNI-transform read below. Per-token V scale depends only
                // on the key (not the value/head index), and pA (the score operand just read above)
                // is already lane=key — same layout as vs_c below — so the scale is folded into pA
                // directly with a per-lane multiply (no broadcast) instead of into V, which would
                // require broadcasting vs_c across the value/head-dim lanes in the dequant loop.
                // zp is a subtraction, not a scalar factor, so it stays on the V side (still needs
                // its per-key value broadcast into the value/head-dim lanes there).
                const int vs_key = k0 + cp * SUBGROUP_SIZE + lane;
                const uint vs_co = v_comp_base + VAL_COMP_OFF(0, 0, vs_key, 0);
                // Keep scale/zp in half: V_scales/V_zp are already half, and the dequant is
                // stored as half — half arithmetic is bit-identical to the float path over the
                // int8 range (verified), so this avoids the half->float->half round trips.
                const half vs_c = (vs_key < k) ? V_scales[vs_co] : (half)0.0f;
                #if VAL_ZERO_POINTS
                    // Fold the bias-trick widen bias (+1152.0h) into zp: the V dequant below widens
                    // via as_half(0x6480 ^ byte) (== signed_byte + 1152), so subtracting (zp+1152)
                    // gives (signed_byte - zp) with no convert_half widen. OOB keys -> vzb_c=1152
                    // (zp=0), and the score-side scale (vs_c=0 for OOB) still zeroes the product.
                    const half vzb_c = (vs_key < k) ? (convert_half(V_zp[vs_co]) + (half)1152.0h) : (half)1152.0h;
                #endif

                #pragma unroll
                for (int r = 0; r < sv_score_blocks; ++r)
                    pA[r] = as_short8(as_half8(pA[r]) * vs_c);
            #endif

            int8 vb[sv_value_blocks];
            #if USE_2D_BLOCK_IO_V_I8
                // int8 V via 8-bit VNNI-transform read: one coalesced read gives a 32-key x
                // 16-value tile (lane=value, each uint packs 4 consecutive keys as bytes). We
                // need this cp-block's 16 keys, which are the 4 uints at vt_half (0 for an even
                // cp, 4 for the odd cp that reuses the previous read). Dequant each byte (per-key
                // scale via the cached vs_c broadcast) and repack into the f16 VNNI operand
                // (2 half-keys per int), matching the scalar vb layout with no subgroup shuffle.
                {
                    // this cp-block = 16 keys = 4 uints (4 keys each). key_rel = u*4 + b.
                    // Bias-trick dequant (microbench-validated, mov -66% vs convert_half4):
                    // extract each key byte with shift+mask (NO as_char4 -> no <4;1,0> :b
                    // deinterleave), widen via the denormal-bias reinterpret (0x6480 ^ byte)
                    // == signed_byte+1152 as half. scale is folded into pA above (score side),
                    // so only the bias-folded zp subtraction (vzb=zp+1152) remains here.
                    #if VAL_ZERO_POINTS
                        // zp is per-key and independent of the value/head index, so the broadcasts
                        // are hoisted out of the cd loop: issued once per cp-block instead of once
                        // per (cd, u) pair.
                        half4 zpb4[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int k0r = u * 4;
                            zpb4[u] = (half4)(sub_group_broadcast(vzb_c, k0r + 0),
                                              sub_group_broadcast(vzb_c, k0r + 1),
                                              sub_group_broadcast(vzb_c, k0r + 2),
                                              sub_group_broadcast(vzb_c, k0r + 3));
                        }
                    #endif
                    #pragma unroll
                    for (int cd = 0; cd < sv_value_blocks; ++cd) {
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint w = vt[cd * 8 + vt_half + u];
                            const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >>  8) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 16) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
                            #if VAL_ZERO_POINTS
                                const half4 deq4 = wide4 - zpb4[u];
                            #else
                                const half4 deq4 = wide4 - (half4)((half)1152.0h);
                            #endif
                            // f16 VNNI operand: vb[cd][key_pair] packs keys (2*key_pair, 2*key_pair+1).
                            // deq4 already holds keys u*4..u*4+3 in order, so its .lo/.hi halves
                            // are exactly the two key_pairs (u*2, u*2+1) for this u — store
                            // straight into vb instead of round-tripping through an array.
                            vb[cd][u * 2 + 0] = as_int(deq4.lo);
                            vb[cd][u * 2 + 1] = as_int(deq4.hi);
                        }
                    }
                }
            #elif V_F16_MULTIBLOCK_READ && USE_2D_BLOCK_IO_KV && sv_value_blocks == 2
                intel_sub_group_2d_block_read_transform_16b_16r16x2c(
                    (global void *)V, VD_w, VD_h, VD_p,
                    (int2)(sg_j0_sv, k0 + cp * SUBGROUP_SIZE), (private uint *)&vb[0]);
            #elif USE_2D_BLOCK_IO_KV
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)V, VD_w, VD_h, VD_p,
                        (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE), (private uint *)&vb[cd]);
                }
            #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd) {
                    vb[cd] = (int8)0;
                    const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
                    if (value < d) {
                        #pragma unroll
                        for (int key_pair = 0; key_pair < 8; ++key_pair) {
                            const int key0 = k0 + cp * SUBGROUP_SIZE + key_pair * 2;
                            const int key1 = key0 + 1;
                            half2 vv = (half2)0.0h;
                            if (key0 < k) {
                                #ifdef KV_COMPRESSED
                                    // i8 compressed V: per-token (per-kv-head) asymmetric dequant.
                                    // Scale/zp vary per key (token), so they must be indexed by
                                    // key0/key1 here, not by the value (head-dim) index.
                                    const uint v_comp_off0 = VAL_COMP_OFF(b1, b0_kv, key0, 0);
                                    #if VAL_ZERO_POINTS
                                        vv[0] = (half)((convert_float(V[(size_t)key0 * ldv + value]) - convert_float(V_zp[v_comp_off0])) * convert_float(V_scales[v_comp_off0]));
                                    #else
                                        vv[0] = (half)(convert_float(V[(size_t)key0 * ldv + value]) * convert_float(V_scales[v_comp_off0]));
                                    #endif
                                #else
                                    vv[0] = V[(size_t)key0 * ldv + value];
                                #endif
                            }
                            if (key1 < k) {
                                #ifdef KV_COMPRESSED
                                    const uint v_comp_off1 = VAL_COMP_OFF(b1, b0_kv, key1, 0);
                                    #if VAL_ZERO_POINTS
                                        vv[1] = (half)((convert_float(V[(size_t)key1 * ldv + value]) - convert_float(V_zp[v_comp_off1])) * convert_float(V_scales[v_comp_off1]));
                                    #else
                                        vv[1] = (half)(convert_float(V[(size_t)key1 * ldv + value]) * convert_float(V_scales[v_comp_off1]));
                                    #endif
                                #else
                                    vv[1] = V[(size_t)key1 * ldv + value];
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
                    A_tile[r][cd] = intel_sub_group_f16_f16_matrix_mad_k16(pA[r], vb[cd], A_tile[r][cd]);
        }
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
            half8 out = convert_half8(A_tile[r][cd]);
            const int col = sg_j0_sv + cd * SUBGROUP_SIZE;
            const int row = wg_j0 + sg_i0_sv + r * 8;
#if USE_2D_BLOCK_IO_A
            if (row + 7 < q && col + SUBGROUP_SIZE <= d) {
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
                    if (out_row < q && out_col < d)
                        A[(size_t)out_row * lda + out_col] = out[rr];
                }
#if USE_2D_BLOCK_IO_A
            }
#endif
        }
    }
}
