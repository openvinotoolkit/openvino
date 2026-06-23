#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io               : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                         : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                   : enable

#include "include/batch_headers/sdpa_utils.cl"

float __builtin_IB_atomic_max_local_f32(__local float *, float);

// #ifndef KQ_SG_TILE_KEYS
// #define KQ_SG_TILE_KEYS      16
// #endif
// #ifndef KQ_SG_TILE_QUERIES
// #define KQ_SG_TILE_QUERIES   16
// #endif
// #ifndef KQ_SG_PER_WG_KEYS
// #define KQ_SG_PER_WG_KEYS    8
// #endif
// #ifndef KQ_SG_PER_WG_QUERIES
// #define KQ_SG_PER_WG_QUERIES 2
// #endif
#define KQ_WG_TILE_KEYS      (KQ_SG_TILE_KEYS * KQ_SG_PER_WG_KEYS)
#define KQ_WG_TILE_QUERIES   (KQ_SG_TILE_QUERIES * KQ_SG_PER_WG_QUERIES)
#define KQ_MB                (KQ_SG_TILE_KEYS / 8)       // 2
#define KQ_QB                (KQ_SG_TILE_QUERIES / 16)      // 1

#define sg_per_wg (KQ_SG_PER_WG_KEYS * KQ_SG_PER_WG_QUERIES)

// S*V subgroup layout is supplied by the host config.
// #ifndef SV_SG_PER_WG_VALUES
// #define SV_SG_PER_WG_VALUES ((D_MAX <= 64) ? 4 : 8)
// #endif
// #ifndef SV_SG_PER_WG_SCORES
// #define SV_SG_PER_WG_SCORES (sg_per_wg / SV_SG_PER_WG_VALUES)
// #endif
#define sv_sg_per_wg_values SV_SG_PER_WG_VALUES
#define sv_sg_per_wg_scores SV_SG_PER_WG_SCORES
#define sv_sg_tile_values   SV_SG_TILE_VALUES
#define sv_sg_tile_scores   SV_SG_TILE_SCORES
#define SV_VALUE_BLOCKS     (sv_sg_tile_values / 16)      // 2
#define SV_SCORE_BLOCKS     (sv_sg_tile_scores / 8)
#define SV_KEY_BLOCKS       (KQ_WG_TILE_KEYS / 16)
#define Q_BLOCKS             (KQ_WG_TILE_QUERIES / SUBGROUP_SIZE)

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
        const global half *K,
        const global half *Q,
        const global half *V,
        global half *A,
#if WITH_ATTN_MASK
        const global half *msk,
#endif
        const int d,
        const int k,
        const int q,
        float scale)
{
    const size_t lane  = get_sub_group_local_id();
    const size_t sg_ij = get_local_id(1);
    const size_t wg_j0 = get_group_id(0) * KQ_WG_TILE_QUERIES;
    const size_t b0 = get_group_id(1);     // heads_num
    const size_t b1 = get_group_id(2);     // batch
    const size_t b0_kv = b0 / (HEADS_NUM / KV_HEADS_NUM);

    const size_t sg_i_kq  = sg_ij % KQ_SG_PER_WG_KEYS;
    const size_t sg_j_kq  = sg_ij / KQ_SG_PER_WG_KEYS;
    const size_t sg_i0_kq = sg_i_kq * KQ_SG_TILE_KEYS;
    const size_t sg_j0_kq = sg_j_kq * KQ_SG_TILE_QUERIES;

    const size_t sv_sg_values = sg_ij % sv_sg_per_wg_values;
    const size_t sv_sg_scores = sg_ij / sv_sg_per_wg_values;
    const size_t sv_value0 = sv_sg_values * sv_sg_tile_values;
    const size_t sv_score0 = sv_sg_scores * sv_sg_tile_scores;

    const float LOG2E = 1.4426950408889634f;

    float iscale = native_recip(scale);
    scale *= LOG2E;

    K += KEY_OFF(b1, b0_kv, 0, 0) + INPUT1_OFFSET;
    Q += QRY_OFF(b1, b0, 0, 0) + INPUT0_OFFSET;
    V += VAL_OFF(b1, b0_kv, 0, 0) + INPUT2_OFFSET;
    A += DST_OFF(b1, b0, 0, 0, 0);
#if WITH_ATTN_MASK
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
#endif

    const int KD_w = d * (int)sizeof(half), KD_h = k, KD_p = KEY_S2 * (int)sizeof(half);
    const int VD_w = d * (int)sizeof(half), VD_h = k, VD_p = VAL_S2 * (int)sizeof(half);
    const int AD_w = d * (int)sizeof(half), AD_h = q, AD_p = DST_S2 * (int)sizeof(half);
    local uint  Q_slm[DKS * Q_BLOCKS * Q_DWORDS * SUBGROUP_SIZE];
    local uint  S_slm[KQ_WG_TILE_KEYS * KQ_WG_TILE_QUERIES / 2];
    local float S_sum_slm[KQ_WG_TILE_QUERIES * KQ_SG_PER_WG_KEYS];
    local float S_max_slm[KQ_WG_TILE_QUERIES];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < KQ_WG_TILE_QUERIES; qi += sg_per_wg * SUBGROUP_SIZE)
        S_max_slm[qi] = -INFINITY;

    // Cooperative Q->SLM staging: the Q tile is Q_BLOCKS(=2) query-blocks x DKS(=8)
    // head-dim chunks = 16 independent (q_block, db) tiles. Distribute them 1:1 across
    // the 16 subgroups so all subgroups load Q (instead of only the first Q_BLOCKS),
    // shrinking the prologue Q-load latency.
    {
        const int q_block = sg_ij / DKS;   // 0..Q_BLOCKS-1
        const int db      = sg_ij % DKS;   // 0..DKS-1
        if (q_block < Q_BLOCKS) {
            const int query = wg_j0 + q_block * SUBGROUP_SIZE + lane;
            ushort16 qv = (ushort16)0;
            if (query < q)
                qv = vload16(0, (global ushort *)(Q + (size_t)query * QRY_S2 + db * KSTEP));
            uint8 q_pack = as_uint8(as_short16(qv));
            intel_sub_group_block_write8(
                (local uint *)&Q_slm[((db * Q_BLOCKS + q_block) * Q_DWORDS) * SUBGROUP_SIZE], q_pack);
        }
    }

    float S_max_tile[KQ_QB];
    float S_sum_tile[KQ_QB];
    #pragma unroll
    for (int qb = 0; qb < KQ_QB; ++qb) {
        S_max_tile[qb] = -INFINITY;
        S_sum_tile[qb] = 0.0f;
    }

    float8 A_tile[SV_SCORE_BLOCKS][SV_VALUE_BLOCKS];
    #pragma unroll
    for (int r = 0; r < SV_SCORE_BLOCKS; ++r)
        #pragma unroll
        for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd)
            A_tile[r][cd] = (float8)0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k0 = 0; k0 < k; k0 += KQ_WG_TILE_KEYS) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == 0);
        const bool last = (k0 + KQ_WG_TILE_KEYS >= k);

        float8 S_tile[KQ_MB][KQ_QB];
        #pragma unroll
        for (int mb = 0; mb < KQ_MB; ++mb)
            #pragma unroll
            for (int qb = 0; qb < KQ_QB; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

        #pragma unroll
        for (int db = 0; db < DKS; ++db) {
            int8 qB[KQ_QB];
            #pragma unroll
            for (int qb = 0; qb < KQ_QB; ++qb) {
                const int q_block = sg_j0_kq / SUBGROUP_SIZE + qb;
                qB[qb] = as_int8(intel_sub_group_block_read8(
                    (local void *)&Q_slm[((db * Q_BLOCKS + q_block) * Q_DWORDS) * SUBGROUP_SIZE]));
            }

            ushort8 k_raw[KQ_MB];
            intel_sub_group_2d_block_read_16b_16r16x1c(
                (global void *)K, KD_w, KD_h, KD_p,
                (int2)(db * KSTEP, key_base), (private ushort *)&k_raw[0]);

            #pragma unroll
            for (int mb = 0; mb < KQ_MB; ++mb) {
                #pragma unroll
                for (int qb = 0; qb < KQ_QB; ++qb)
                    S_tile[mb][qb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(k_raw[mb]), qB[qb], S_tile[mb][qb]);
            }
        }

#if 0
        // V prefetch: disabled. Ablation showed it costs ~11us net (it prefetches the
        // same V tiles consumed by the transform-load in this same iteration, with no
        // intervening compute to hide latency, so it is pure overhead here).
        #pragma unroll
        for (int cp = 0; cp < SV_KEY_BLOCKS; ++cp) {
            #pragma unroll
            for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd) {
                intel_sub_group_2d_block_prefetch_16b_16r16x1c(
                    (const global void *)V, VD_w, VD_h, VD_p,
                    (int2)(sv_value0 + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE));
            }
        }
#endif

        half2 mask_tile;
        float2 k_mask;
        #pragma unroll
        for (int ii = 0; ii < KQ_SG_TILE_KEYS / SUBGROUP_SIZE; ++ii) {
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
        for (int ii = 0; ii < KQ_SG_TILE_KEYS / SUBGROUP_SIZE; ++ii)
            mask_tile_float[ii] = mask_tile_float[ii] * iscale;

        #if WITH_ATTN_MASK
            // Full 2D mask [query x key]: each lane loads its own query row (strided,
            // same access pattern as sdpa_micro's tile_load_t). Pre-scale by iscale at
            // load time and keep it as float so the softmax max-loop below only does a
            // branchless add (mirrors micro's tile_elementwise(unscale)+tile_binary add).
            float16 mask_full[KQ_QB][KQ_SG_TILE_KEYS / SUBGROUP_SIZE];
            if (MASK_IS_FULL_2D) {
                #pragma unroll
                for (int qb = 0; qb < KQ_QB; ++qb) {
                    const int mask_query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    #pragma unroll
                    for (int ii = 0; ii < KQ_SG_TILE_KEYS / SUBGROUP_SIZE; ++ii) {
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

        float alpha[KQ_QB];
        #pragma unroll
        for (int qb = 0; qb < KQ_QB; ++qb) {
            float lmax = -INFINITY;
            #pragma unroll
            for (int mb = 0; mb < KQ_MB; ++mb) {
                #pragma unroll
                for (int mm = 0; mm < 8; ++mm) {
                    const int key_rel = mb * 8 + mm;
                    const int mask_idx = key_rel / SUBGROUP_SIZE;
                    const int mask_lane = key_rel - mask_idx * SUBGROUP_SIZE;
                    const int query = wg_j0 + sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                    const int key = key_base + key_rel;
                    float s = S_tile[mb][qb][mm] + sub_group_broadcast(k_mask[mask_idx], mask_lane);
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
                    S_tile[mb][qb][mm] = s;
                    lmax = fmax(lmax, s);
                }
            }

            const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
            __builtin_IB_atomic_max_local_f32(&S_max_slm[query], lmax);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int qb = 0; qb < KQ_QB; ++qb) {
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
            for (int mb = 0; mb < KQ_MB; ++mb) {
                float8 exp_tile = ok ? native_exp2(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                lsum += exp_tile[0] + exp_tile[1] + exp_tile[2] + exp_tile[3]
                      + exp_tile[4] + exp_tile[5] + exp_tile[6] + exp_tile[7];

                const int key = sg_i0_kq + mb * 8;
                const int key_block = key / SUBGROUP_SIZE;
                const int key_lane = key - key_block * SUBGROUP_SIZE;
                const int s_half_offset = (key_block * KQ_WG_TILE_QUERIES + query) * SUBGROUP_SIZE + key_lane;
                vstore4(as_uint4(convert_half8(exp_tile)), 0, &S_slm[s_half_offset >> 1]);
            }
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
        }

        if (last) {
            #pragma unroll
            for (int qb = 0; qb < KQ_QB; ++qb) {
                const int query = sg_j0_kq + qb * SUBGROUP_SIZE + lane;
                S_sum_slm[query * KQ_SG_PER_WG_KEYS + sg_i_kq] = S_sum_tile[qb];
            }
        }

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        if (!first) {
            #pragma unroll
            for (int r = 0; r < SV_SCORE_BLOCKS; ++r) {
                float8 av;
                const int rel_query = sv_score0 + r * 8 - sg_j0_kq;
                const int alpha_qb = rel_query / SUBGROUP_SIZE;
                const int alpha_lane0 = rel_query - alpha_qb * SUBGROUP_SIZE;
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr)
                    av[rr] = sub_group_broadcast(alpha[alpha_qb], alpha_lane0 + rr);
                #pragma unroll
                for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd)
                    A_tile[r][cd] *= av;
            }
        }

        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int cp = 0; cp < SV_KEY_BLOCKS; ++cp) {
            short8 pA[SV_SCORE_BLOCKS];
            #pragma unroll
            for (int r = 0; r < SV_SCORE_BLOCKS; ++r) {
                const int query0 = sv_score0 + r * 8;
                pA[r] = as_short8(intel_sub_group_block_read_us8(
                    (local void *)&S_slm[((cp * KQ_WG_TILE_QUERIES + query0) * SUBGROUP_SIZE) >> 1]));
            }

            int8 vb[SV_VALUE_BLOCKS];
            #pragma unroll
            for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd) {
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    (global void *)V, VD_w, VD_h, VD_p,
                    (int2)(sv_value0 + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE), (private uint *)&vb[cd]);
            }

            #pragma unroll
            for (int r = 0; r < SV_SCORE_BLOCKS; ++r)
                #pragma unroll
                for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd)
                    A_tile[r][cd] = intel_sub_group_f16_f16_matrix_mad_k16(pA[r], vb[cd], A_tile[r][cd]);
        }
    }

    #pragma unroll
    for (int r = 0; r < SV_SCORE_BLOCKS; ++r) {
        float8 inv_l;
        #pragma unroll
        for (int rr = 0; rr < 8; ++rr) {
            const int query = sv_score0 + r * 8 + rr;
            float l = S_sum_slm[query * KQ_SG_PER_WG_KEYS + 0];
            #pragma unroll
            for (int p = 1; p < KQ_SG_PER_WG_KEYS; ++p)
                l += S_sum_slm[query * KQ_SG_PER_WG_KEYS + p];
            inv_l[rr] = (l > 0.0f) ? native_recip(l) : 0.0f;
        }
        #pragma unroll
        for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd)
            A_tile[r][cd] *= inv_l;
    }

    #pragma unroll
    for (int r = 0; r < SV_SCORE_BLOCKS; ++r) {
        #pragma unroll
        for (int cd = 0; cd < SV_VALUE_BLOCKS; ++cd) {
            half8 out = convert_half8(A_tile[r][cd]);
            const int col = sv_value0 + cd * SUBGROUP_SIZE;
            const int row = wg_j0 + sv_score0 + r * 8;
            if (row + 7 < q && col + SUBGROUP_SIZE <= d) {
                intel_sub_group_2d_block_write_16b_8r16x1c(
                    (global void *)A, AD_w, AD_h, AD_p,
                    (int2)(col, row),
                    (private ushort *)&out);
            } else {
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const int out_row = row + rr;
                    const int out_col = col + lane;
                    if (out_row < q && out_col < d)
                        A[(size_t)out_row * DST_S2 + out_col] = out[rr];
                }
            }
        }
    }
}
