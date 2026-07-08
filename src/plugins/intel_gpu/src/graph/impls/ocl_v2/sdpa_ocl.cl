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
#if WITH_ATTN_MASK
        const global half *msk,
#endif
#if WITH_SCALE
        global SCALE_DATA_T *scale_ptr,
#endif
        const int d,
        const int k,
        const int q
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
    const size_t lane  = get_sub_group_local_id();
    const size_t sg_ij = get_local_id(1);
    const size_t wg_j0 = get_group_id(0) * kq_wg_tile_queries;
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

    Q += QRY_OFF(b1, b0, 0, 0) + INPUT0_OFFSET;
    K += KEY_OFF(b1, b0_kv, 0, 0) + INPUT1_OFFSET;
    V += VAL_OFF(b1, b0_kv, 0, 0) + INPUT2_OFFSET;
    A += DST_OFF(b1, b0, 0, 0, 0);
#if WITH_ATTN_MASK
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
#endif

    const int KD_w = d * (int)sizeof(KEY_DATA_T), KD_h = k, KD_p = KEY_S2 * (int)sizeof(KEY_DATA_T);
    const int VD_w = d * (int)sizeof(VAL_DATA_T), VD_h = k, VD_p = VAL_S2 * (int)sizeof(VAL_DATA_T);
    const int AD_w = d * (int)sizeof(half), AD_h = q, AD_p = DST_S2 * (int)sizeof(half);
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
        const int query = wg_j0 + q_block * SUBGROUP_SIZE + lane;
        const int head_base = db * DPAS_K;
        ushort16 qv = (ushort16)0;
        if (query < q) {
            if (head_base + DPAS_K <= d) {
                qv = vload16(0, (global ushort *)(Q + (size_t)query * QRY_S2 + head_base));
            } else {
                #pragma unroll
                for (int head_offset = 0; head_offset < DPAS_K; ++head_offset) {
                    if (head_base + head_offset < d) {
                        qv[head_offset] = as_ushort(Q[(size_t)query * QRY_S2 + head_base + head_offset]);
                    }
                }
            }
        }
        uint8 q_pack = as_uint8(as_short16(qv));
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

    for (int k0 = 0; k0 < k; k0 += kq_wg_tile_keys) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == 0);
        const bool last = (k0 + kq_wg_tile_keys >= k);

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
            const uint sc_off = KEY_COMP_OFF(b1, b0_kv, sc_key, 0);
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
#elif USE_2D_BLOCK_IO
            intel_sub_group_2d_block_read_16b_16r16x1c(
                (global void *)K, KD_w, KD_h, KD_p,
                (int2)(db * DPAS_K, key_base), (private ushort *)&k_raw[0]);
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
                                const float deq_k = (convert_float(K[(size_t)key * KEY_S2 + head]) - k_zp) * k_sc;
                            #else
                                const float deq_k = convert_float(K[(size_t)key * KEY_S2 + head]) * k_sc;
                            #endif
                            k_raw[mb][key_offset] = as_ushort((half)deq_k);
                        #else
                            k_raw[mb][key_offset] = as_ushort(K[(size_t)key * KEY_S2 + head]);
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

#if 0
        // V prefetch: disabled. Ablation showed it costs ~11us net (it prefetches the
        // same V tiles consumed by the transform-load in this same iteration, with no
        // intervening compute to hide latency, so it is pure overhead here).
        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                intel_sub_group_2d_block_prefetch_16b_16r16x1c(
                    (const global void *)V, VD_w, VD_h, VD_p,
                    (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE));
            }
        }
#endif

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
                    if (key > query) {
                        s = -INFINITY;
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

        #pragma unroll
        for (int cp = 0; cp < sv_key_blocks; ++cp) {
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
                const uint vs_co = VAL_COMP_OFF(b1, b0_kv, vs_key, 0);
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
            #pragma unroll
            for (int cd = 0; cd < sv_value_blocks; ++cd) {
                #if USE_2D_BLOCK_IO_V_I8
                    // int8 V via 8-bit VNNI-transform read: one coalesced read gives a 32-key x
                    // 16-value tile (lane=value, each uint packs 4 consecutive keys as bytes). We
                    // need this cp-block's 16 keys, which are uints 0..3 (keys cp*16 .. +15 when read
                    // at row k0+cp*16). Dequant each byte (per-key scale via the cached vs_c
                    // broadcast) and repack into the f16 VNNI operand (2 half-keys per int), matching
                    // the scalar vb layout with no subgroup shuffle.
                    {
                        uint vt[8];
                        intel_sub_group_2d_block_read_transform_8b_32r16x1c(
                            (global void *)V, VD_w, VD_h, VD_p,
                            (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE), (private uint *)&vt[0]);
                        // this cp-block = 16 keys = uints 0..3 (4 keys each). key_rel = u*4 + b.
                        // Bias-trick dequant (microbench-validated, mov -66% vs convert_half4):
                        // extract each key byte with shift+mask (NO as_char4 -> no <4;1,0> :b
                        // deinterleave), widen via the denormal-bias reinterpret (0x6480 ^ byte)
                        // == signed_byte+1152 as half. scale is folded into pA above (score side),
                        // so only the bias-folded zp subtraction (vzb=zp+1152) remains here.
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const uint w = vt[u];
                            const int k0r = u * 4;
                            const half4 wide4 = (half4)(as_half((ushort)(0x6480 ^ ((w >>  0) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >>  8) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 16) & 0xFFu))),
                                                        as_half((ushort)(0x6480 ^ ((w >> 24) & 0xFFu))));
                            #if VAL_ZERO_POINTS
                                const half4 zpb4 = (half4)(sub_group_broadcast(vzb_c, k0r + 0),
                                                           sub_group_broadcast(vzb_c, k0r + 1),
                                                           sub_group_broadcast(vzb_c, k0r + 2),
                                                           sub_group_broadcast(vzb_c, k0r + 3));
                                const half4 deq4 = wide4 - zpb4;
                            #else
                                const half4 deq4 = wide4 - (half4)((half)1152.0h);
                            #endif
                            // f16 VNNI operand: vb[key_pair] packs keys (2*key_pair, 2*key_pair+1).
                            // deq4 already holds keys k0r..k0r+3 in order, so its .lo/.hi halves
                            // are exactly the two key_pairs (u*2, u*2+1) for this u — store
                            // straight into vb instead of round-tripping through an array.
                            vb[cd][u * 2 + 0] = as_int(deq4.lo);
                            vb[cd][u * 2 + 1] = as_int(deq4.hi);
                        }
                    }
                #elif USE_2D_BLOCK_IO
                    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                        (global void *)V, VD_w, VD_h, VD_p,
                        (int2)(sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE), (private uint *)&vb[cd]);
                #else
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
                                        vv[0] = (half)((convert_float(V[(size_t)key0 * VAL_S2 + value]) - convert_float(V_zp[v_comp_off0])) * convert_float(V_scales[v_comp_off0]));
                                    #else
                                        vv[0] = (half)(convert_float(V[(size_t)key0 * VAL_S2 + value]) * convert_float(V_scales[v_comp_off0]));
                                    #endif
                                #else
                                    vv[0] = V[(size_t)key0 * VAL_S2 + value];
                                #endif
                            }
                            if (key1 < k) {
                                #ifdef KV_COMPRESSED
                                    const uint v_comp_off1 = VAL_COMP_OFF(b1, b0_kv, key1, 0);
                                    #if VAL_ZERO_POINTS
                                        vv[1] = (half)((convert_float(V[(size_t)key1 * VAL_S2 + value]) - convert_float(V_zp[v_comp_off1])) * convert_float(V_scales[v_comp_off1]));
                                    #else
                                        vv[1] = (half)(convert_float(V[(size_t)key1 * VAL_S2 + value]) * convert_float(V_scales[v_comp_off1]));
                                    #endif
                                #else
                                    vv[1] = V[(size_t)key1 * VAL_S2 + value];
                                #endif
                            }
                            vb[cd][key_pair] = as_int(vv);
                        }
                    }
                #endif
            }

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
#if USE_2D_BLOCK_IO
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
                        A[(size_t)out_row * DST_S2 + out_col] = out[rr];
                }
#if USE_2D_BLOCK_IO
            }
#endif
        }
    }
}
