// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// xe_hpg (SG8) PA PREFILL kernel, f16 head 64 / 96 / 128 / 256 / 512 (k == v), with or without a sliding window, a sink
// and token_type_ids, which also serves PA MIXED on an f16 cache (without token_type_ids) and plain SDPA (heads 32-256
// in steps of 8 and 512) without causality, with or without an attention mask, or causal without a mask
// (HPG_PA_PREFILL, set by the host: SDPAOclGenerator's hpg_pa_prefill_ok() / hpg_plain_prefill_ok()). Included by
// sdpa_ocl.cl in place of its own kernel, after the config and loader headers. The loop is sdpa_ocl.cl's (online
// softmax over workgroup key tiles, causal and window bounds, S_max_slm / S_sum_slm exchange); what differs is built
// around the SG8 operand forms:
//  - KQ is DPAS(Q, K): A = Q from Q_slm (staged so one ulong4 SLM block read is the operand), B = K straight from the
//    input (lane = key, 16 depth halves per lane: one 32 B read per key row). The scores come out lane = key; a
//    per-subgroup 8 x 8 SLM transpose turns them to lane = query for the mask and softmax.
//  - SV is DPAS(V, P): A = V (lane = key pair, component = value), B = P (lane = query), so the output tile is
//    lane = query and the rescale by alpha needs no broadcast.
//  - V of the first HPG_PA_V_LEAD key blocks is loaded before the wait of the P barrier and kept HPG_PA_V_LEAD blocks
//    ahead in the S*V loop, which pins the loads (IGC otherwise sinks them toward their use).
//  - A transposed V (V_TRANSPOSED: plain SDPA, value order {..,3,2}, stored [value][key]) is read by value rows, lane =
//    value and dword = key pair (hpg_v_rows()), the B operand: SV is DPAS(P, V) there, with P stored in S_slm as the A
//    operand (lane = key pair, component = query). The output tile is then lane = value (alpha by broadcast) and goes
//    back to lane = query through the KQ transpose scratch before the store.
//  - A subgroup tile entirely below the diagonal and inside the window (every key <= its first query and in the window
//    of its last query, all keys < k) skips the mask.
// The sink and the image groups of token_type_ids follow sdpa_ocl.cl: the sink seeds the online-softmax state (max = the
// sink logit, sum = 1 in one subgroup), and an image group is bidirectional.
// Plain SDPA takes its rows from the 4D layout (batch = get_group_id(2), a workgroup per 32 queries). Without causality
// only the keys past k are masked; causal, query j sits at key position j + max(0, k - q) when lower-right aligned
// (wg_p0), as in sdpa_ocl.cl. A mask is added in the raw score domain, as in sdpa_ocl.cl (no tile skips it).
// Out-of-range rows are clamped to the last valid key or query, so no read leaves the subsequence, and their scores are
// masked to -INFINITY (P = 0 times a finite V).
// PA MIXED (IS_PA_MIXED) reads every key from the token-major f16 pages (an f16 page holds the new tokens bit for bit),
// with the queries at key positions past_len + query (wg_p0). On a compressed cache (HPG_COMP: i8 BY_TOKEN, i8 / u4
// BY_CHANNEL) the past keys come from the pages and the new ones from the f16 Kc / Vc (PA_CUR_KV_F16): the key loop
// splits its tile at past_len (k_chunk), so a tile reads one source. Page rows are dequantized per lane, (q - zp) * scale
// in half (PA_DEQ); the V scale of each key is folded into P where P is stored (lsum takes the unscaled P). Rows are
// clamped to the tile's last key, so every read is of a written slot. Design: "xe_hpg PA PREFILL kernel" in the docs.

// The S*V subgroup of sg_ij owns the 16 queries of its KQ subgroup, 32 for a compressed MIXED at heads 64 / 96 / 128 /
// 256 (sv_sg_per_wg_values == kq_sg_per_wg_keys, the score tile is the KQ query tile), and each holds 16 or 32 value
// columns of the head, 8 in that MIXED at head 64 (hpg_pa_prefill_tiling()). The head need not be a power of two nor a
// multiple of 16: plain SDPA at heads such as 72 and 40 (HPG_TAIL = 8) has a last depth tile of 8 depths (K and Q read
// 16 B there and the rest is zero) and a last value subgroup of 8 columns, which reads the 16 columns that end the head
// and stores its upper 8 (sv_col below), so no load leaves the head.
#define HPG_TAIL (K_HEAD_SIZE % DPAS_K)
#if !(SG8 && (IS_PREFILL || IS_PA_MIXED) && K_HEAD_SIZE == V_HEAD_SIZE && (HPG_TAIL == 0 || HPG_TAIL == DPAS_K / 2) && \
      (kq_sg_tile_queries == 16 || kq_sg_tile_queries == 32) && sv_sg_tile_scores == kq_sg_tile_queries && \
      sv_sg_per_wg_scores == kq_sg_per_wg_queries && \
      sv_sg_per_wg_values == kq_sg_per_wg_keys && \
      (sv_sg_tile_values == 8 || sv_sg_tile_values == 16 || sv_sg_tile_values == 32) && \
      sv_sg_tile_values * sv_sg_per_wg_values == V_HEAD_SIZE + HPG_TAIL)
#  error "sdpa_ocl_hpg_pa_prefill.cl: the host gate (hpg_pa_prefill_ok()) and this kernel disagree on the shape or the tiling"
#endif
// A static scalar mask (plain SDPA only, finite: hpg_plain_prefill_ok()) adds the same value to every score of a row, which
// the softmax ignores, so the kernel leaves it out. Plain causal SDPA comes without a mask (its row would be the shifted
// query) and without a window.
// Plain SDPA with a per-channel compressed K / V (HPG_PC, the GQA decomposition's fused cache: i8, or 4-bit with
// PLAIN_KV_U4, two channels per byte, the even one in the low nibble): each K element is code - zp of its channel in half,
// the K scale folded into Q; each V element is its code in half, the V scale and zp applied to the output column,
// vscale * (acc / l - vzp) (exact: l is the sum of P; as sdpa_micro). The channels' scales and zps are staged once per
// workgroup (K zp in SLM, V in registers).
// i8 with a [key][value] V (HPG_PC_XOR) widens K and V by the 0x6480 XOR alone (half 1152 + code, as sdpa_micro's copy):
// the K offset and zp shift every logit of a query alike (softmax-invariant, no sink here), the V offset leaves the
// accumulator once per key tile (pc_v_off), the V zp in the epilogue. A transposed i8 V keeps the code - zp / code reads
// (docs/sdpa_ocl.md "Per-channel compressed KV on xe_hpg").
#define HPG_PC (KV_COMP_PER_CHANNEL && !IS_PAGED_ATTENTION)
#define HPG_PC_XOR (HPG_PC && !PLAIN_KV_U4 && !V_TRANSPOSED)
#if defined(HAS_SCALAR_ATTN_MASK) || (defined(STATIC_SCALAR_ATTN_MASK_VALUE) && IS_PAGED_ATTENTION) || WITH_SCALE || \
    (defined(KV_COMPRESSED) && !HPG_PC) || MICRO_MATH || INPUT0_IS_BF16 || (IS_PAGED_ATTENTION && !IS_CAUSAL) || \
    (!IS_PAGED_ATTENTION && (defined(HAS_SINK_INPUT) || (IS_CAUSAL && (WITH_ATTN_MASK || SLIDING_WINDOW_SIZE))))
#  error "sdpa_ocl_hpg_pa_prefill.cl: an op the host gate (hpg_pa_prefill()) should have sent to the generic kernel"
#endif
#if HPG_TAIL && IS_PAGED_ATTENTION
#  error "sdpa_ocl_hpg_pa_prefill.cl: a head that is not a multiple of 16 is plain SDPA's (the K page reads have no tail)"
#endif
#if V_TRANSPOSED && IS_PAGED_ATTENTION
#  error "sdpa_ocl_hpg_pa_prefill.cl: a transposed V is plain SDPA's"
#endif
#if IS_PA_MIXED && (HAS_QQ_BIAS || HAS_TOKEN_TYPE_IDS || PAGED_ATTENTION_BLOCK_SIZE != DPAS_K)
#  error "sdpa_ocl_hpg_pa_prefill.cl: MIXED reads token-major pages of 16 tokens, without qq_bias or token_type_ids"
#endif
// An f16 page holds the new tokens bit for bit; a compressed one rounded them, so they come from Kc / Vc.
#if IS_PA_MIXED && (IS_PA_KV_COMPRESSED != PA_CUR_KV_F16)
#  error "sdpa_ocl_hpg_pa_prefill.cl: MIXED reads the new tokens from Kc / Vc exactly when the cache is compressed"
#endif
#define HPG_COMP (IS_PA_MIXED && IS_PA_KV_COMPRESSED)
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
#if sv_sg_tile_values == 8
#  define HPG_V_T       uint4
#  define HPG_V_LOAD(p) vload4(0, p)
#elif sv_sg_tile_values == 16
#  define HPG_V_T       uint8
#  define HPG_V_LOAD(p) vload8(0, p)
#else
#  define HPG_V_T       uint16
#  define HPG_V_LOAD(p) vload16(0, p)
#endif

// A transposed V: hpg_v_rows() also takes V's start in halves off a dword (2: every row through the two-half path) and
// vt_mode, a literal of the caller: 0 checks each block against causal_k and the row alignment, 1 reads a tile wholly
// below causal_k with every row on a dword without checks. With an even lead and key block count the ring takes two
// blocks per read (hpg_vt_rows2(): a value row's 64 B line at once).
// A per-channel one (HPG_PC: i8, or 4-bit by byte rows of channel pairs) reads one block at a time, and with vt_mode 2 a
// whole tile whose rows start off a dword by aligned dwords.
#if V_TRANSPOSED && HPG_PC
#  define HPG_VT_ARG , const uint vt_par0, const int vt_mode
#  define HPG_VT     , vt_par0, vt_mode
#  define HPG_VT_PAIRS 0
#elif V_TRANSPOSED
#  define HPG_VT_ARG , const uint vt_par0, const int vt_mode
#  define HPG_VT     , vt_par0, vt_mode
#  define HPG_VT_PAIRS (HPG_PA_V_LEAD % 2 == 0 && sv_key_blocks % 2 == 0)
#else
#  define HPG_VT_ARG
#  define HPG_VT
#  define HPG_VT_PAIRS 0
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
    // Unrolled whole at head 256, IGC would load K of every depth tile ahead (the host's HPG_PA_KQ_UNROLL).
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

#if HPG_COMP
// The depth loops of hpg_kq_comp() run one depth tile per iteration (HPG_COMP_KQ_UNROLL overrides it for offline A/B).
#  ifndef HPG_COMP_KQ_UNROLL
#    define HPG_COMP_KQ_UNROLL 1
#  endif

// One depth tile of a compressed MIXED's KQ, hpg_kq()'s DPAS(Q, K): scores += Q (A, from Q_slm) x K (B, lane = key).
SDPA_OCL_INLINE void FUNC(hpg_kq_comp_dpas)(__private float8 (*scores)[kq_query_blocks], const __private int8 *k_b,
                                            const __local uint *q_slm, const int db) {
    #pragma unroll
    for (int qb = 0; qb < kq_query_blocks; ++qb) {
        const int8 q_a = as_int8(intel_sub_group_block_read_ul4((const __local ulong *)(q_slm + Q_SLM_OFF(db, qb))));
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            scores[mb][qb] = DPAS_MAD_K16(q_a, k_b[mb], scores[mb][qb]);
    }
}

// Scores of one subgroup tile of a compressed MIXED, as hpg_kq(): lane = query, component = key. A tile of past keys
// (from_cache) reads each key's row of its token-major page (hpg_kq()'s MIXED addressing) and dequantizes it, (q - zp) *
// scale in half: BY_TOKEN with the key's (scale, zp), [token] and [PAGED_ATTENTION_BLOCK_SIZE + token] of the page's comp,
// read once per tile; BY_CHANNEL with the page's (scale, zp) dwords of the tile's 16 depths, the same for every key. A tile
// of new keys reads Kc (row key - past_len) as hpg_kq() reads PREFILL's K. The rows are clamped to lim - 1, so every read
// is of a written slot; the caller masks the keys at/past lim. A subgroup whose first key is at/past lim leaves its scores 0.
SDPA_OCL_INLINE void FUNC(hpg_kq_comp)(__private float8 (*scores)[kq_query_blocks], const __global INPUT3_TYPE *block_indices,
                                       const __global KEY_DATA_T *K, const __global half *Kc, const uint ldkc,
                                       const bool from_cache, const int key_base, const int lim, const int past_len,
                                       const __local uint *q_slm, __local uint *transpose, const uint lane) {
    if (key_base >= lim)
        return;
    if (from_cache) {
        // key_base is a multiple of 16 (a tile of past keys starts at a workgroup tile), so page pg holds keys
        // page_key0[pg]..; a page past lim takes the page of the last key, where the clamped rows of its row blocks land.
        const __global KEY_DATA_T *k_page[kq_key_blocks / 2];
        int page_key0[kq_key_blocks / 2];
        #pragma unroll
        for (int pg = 0; pg < kq_key_blocks / 2; ++pg) {
            page_key0[pg] = min(key_base + pg * DPAS_K, (lim - 1) & ~(PAGED_ATTENTION_BLOCK_SIZE - 1));
            k_page[pg] = K + PA_K_PAGE_OFF(block_indices[page_key0[pg] / PAGED_ATTENTION_BLOCK_SIZE], 0);
        }
        int page_row[kq_key_blocks];
    #if !IS_PA_K_BY_CHANNEL
        half k_sc[kq_key_blocks];
        half k_zp[kq_key_blocks];
    #endif
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb) {
            page_row[mb] = min(key_base + mb * DPAS_ROWS + (int)lane, lim - 1) - page_key0[mb / 2];
    #if !IS_PA_K_BY_CHANNEL
            const __global half *k_comp = (const __global half *)(k_page[mb / 2] + PA_K_COMP_OFF);
        #if defined(NEG_SG8) && NEG_SG8 == 19
            const int comp_tok = page_row[mb] % DPAS_ROWS;  // negative control: tokens 8..15 take the comp of tokens 0..7
        #else
            const int comp_tok = page_row[mb];
        #endif
            k_sc[mb] = k_comp[comp_tok];
            k_zp[mb] = k_comp[PAGED_ATTENTION_BLOCK_SIZE + comp_tok];
    #endif
        }
        __attribute__((opencl_unroll_hint(HPG_COMP_KQ_UNROLL)))
        for (int db = 0; db < DKS_ACTIVE; ++db) {
            int8 k_b[kq_key_blocks];
            #pragma unroll
            for (int pg = 0; pg < kq_key_blocks / 2; ++pg) {
    #if IS_PA_K_BY_CHANNEL
                // (scale, zp) of depths db * 16 + j: dword db * 16 + j of the page's comp.
                const uint16 cc = vload16(0, (const __global uint *)(k_page[pg] + PA_K_COMP_OFF) + db * DPAS_K);
                half16 sc;
                half16 zp;
                #pragma unroll
                for (int j = 0; j < DPAS_K; ++j) {
        #if defined(NEG_SG8) && NEG_SG8 == 23
                    const half2 sc_zp = as_half2(cc[j ^ 1]);  // negative control: the pairs of two channels swapped
        #else
                    const half2 sc_zp = as_half2(cc[j]);
        #endif
        #if defined(NEG_SG8) && NEG_SG8 == 24
                    sc[j] = sc_zp.s1;  // negative control: scale and zp of a pair swapped
                    zp[j] = sc_zp.s0;
        #else
                    sc[j] = sc_zp.s0;
                    zp[j] = sc_zp.s1;
        #endif
                }
    #endif
                #pragma unroll
                for (int hb = 0; hb < 2; ++hb) {
                    const int mb = 2 * pg + hb;
                    const __global KEY_DATA_T *k_row = k_page[pg] + (size_t)page_row[mb] * PA_K_ROW_ELEMS;
    #if IS_PA_K_U4
                    // Depths db * 16 + 2j and + 1: the low and the high nibble of byte db * 8 + j (unsigned), four
                    // per 16-bit word of the read (U4_TO_HALF, exact).
                    const ushort4 kw = as_ushort4(vload2(0, (const __global uint *)(k_row + db * DPAS_K / 2)));
                    half16 kq;
                    #pragma unroll
                    for (int i = 0; i < DPAS_K / 4; ++i) {
        #if defined(NEG_SG8) && NEG_SG8 == 25
                        // Negative control: the two nibbles of a byte swapped.
                        kq[4 * i] = U4_TO_HALF(kw[i], 4);
                        kq[4 * i + 1] = U4_TO_HALF(kw[i], 0);
                        kq[4 * i + 2] = U4_TO_HALF(kw[i], 12);
                        kq[4 * i + 3] = U4_TO_HALF(kw[i], 8);
        #else
                        kq[4 * i] = U4_TO_HALF(kw[i], 0);
                        kq[4 * i + 1] = U4_TO_HALF(kw[i], 4);
                        kq[4 * i + 2] = U4_TO_HALF(kw[i], 8);
                        kq[4 * i + 3] = U4_TO_HALF(kw[i], 12);
        #endif
                    }
    #else
                    const char16 kb = as_char16(vload4(0, (const __global uint *)(k_row + db * DPAS_K)));
        #if defined(NEG_SG8) && NEG_SG8 == 22
                    // Negative control: the two channels of each dword swapped.
                    const half16 kq = convert_half16(kb.s1032547698badcfe);
        #else
                    const half16 kq = convert_half16(kb);
        #endif
    #endif
    #if IS_PA_K_BY_CHANNEL
                    k_b[mb] = as_int8(PA_DEQ(kq, zp, sc));
    #else
                    k_b[mb] = as_int8(PA_DEQ(kq, (half16)k_zp[mb], (half16)k_sc[mb]));
    #endif
                }
            }
            FUNC_CALL(hpg_kq_comp_dpas)(scores, k_b, q_slm, db);
        }
    } else {
        // New keys: Kc rows key - past_len, n = lim - past_len of them (dword rows: the host gate).
        const int row0 = key_base - past_len;
        const int n = lim - past_len;
        __attribute__((opencl_unroll_hint(HPG_COMP_KQ_UNROLL)))
        for (int db = 0; db < DKS_ACTIVE; ++db) {
            int8 k_b[kq_key_blocks];
            #pragma unroll
            for (int mb = 0; mb < kq_key_blocks; ++mb) {
    #if defined(NEG_SG8) && NEG_SG8 == 27
                const int row = (row0 + mb * DPAS_ROWS + (int)lane + 1) % n;  // negative control: the next new token's row
    #else
                const int row = min(row0 + mb * DPAS_ROWS + (int)lane, n - 1);
    #endif
                k_b[mb] = as_int8(vload8(0, (const __global uint *)(Kc + (size_t)row * ldkc + db * DPAS_K)));
            }
            FUNC_CALL(hpg_kq_comp_dpas)(scores, k_b, q_slm, db);
        }
    }
    // lane = key -> lane = query, as in hpg_kq().
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

// V scale of key key_base + mb * DPAS_ROWS + lane of a tile of past keys, folded into P where P is stored: [token] of the
// page's comp. 0 at/past lim, where the slot may never have been written (a NaN scale times P = 0 is NaN).
SDPA_OCL_INLINE void FUNC(hpg_v_scales)(__private half *v_sc, const __global INPUT3_TYPE *block_indices,
                                        const __global VAL_DATA_T *V, const int key_base, const int lim, const uint lane) {
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        const int key = key_base + mb * DPAS_ROWS + (int)lane;
        half sc = 0.0h;
        if (key < lim) {
            const __global VAL_DATA_T *page = V + PA_V_PAGE_OFF(block_indices[key / PAGED_ATTENTION_BLOCK_SIZE], 0);
            sc = ((const __global half *)(page + PA_V_COMP_OFF))[key % PAGED_ATTENTION_BLOCK_SIZE];
        }
        v_sc[mb] = sc;
    }
}
#endif

// The V rows of one 16-key block for this lane: keys key0 + 2 lane and key0 + 2 lane + 1 (clamped to causal_k - 1),
// value columns col..col + sv_sg_tile_values - 1. MIXED: rows of the block's page (key0 is a multiple of 16), or of the
// last key's page when the block is past causal_k, so the clamped rows stay in the page. A compressed MIXED reads its new
// tokens' rows from Vc here (key0 and causal_k are then Vc rows); its page rows are hpg_vq_rows()'.
SDPA_OCL_INLINE void FUNC(hpg_v_rows)(HPG_PAGES_ARG const __global half *V, const uint ldv, const int key0,
                                      const int causal_k, const uint col, const uint lane, __private HPG_V_T *even,
                                      __private HPG_V_T *odd HPG_VT_ARG) {
#if V_TRANSPOSED && HPG_PC
    // A per-channel i8 V by value rows (ldv in bytes): lane l of group g reads keys key0..key0 + 15 of value column
    // col + 8 g + l, one 16 B load when the block is below causal_k and every row starts on a dword (vt_par0 0), five
    // aligned dwords shifted down by the row's offset in a whole tile of rows off a dword (vt_mode 2: the fifth dword is
    // inside the pitch, the caller's vt_whole_mis), else a byte per key clamped to causal_k - 1; the codes in half give
    // hpg_sv() the f16 rows' dwords (dword i = keys 2i, 2i + 1; the zp is in the epilogue). A block past causal_k reads
    // nothing (P is 0 there). A 4-bit V (PLAIN_KV_U4): column c reads byte row c / 2 the same way and takes nibble c % 2
    // of each key's byte (SDPA_OCL_NEG_X5=4: the other one, negative control).
    #pragma unroll
    for (int g = 0; g < sv_sg_tile_values / SUBGROUP_SIZE; ++g) {
    #if PLAIN_KV_U4
        const uint c = col + g * SUBGROUP_SIZE + lane;
        const __global char *row = (const __global char *)V + (size_t)(c / 2) * ldv + key0;
        #if NEG_X5 == 4
        const uint nsh = (c & 1u) ? 0u : 4u;
        #else
        const uint nsh = (c & 1u) ? 4u : 0u;
        #endif
        uint4 w = (uint4)0;
        if (vt_mode == 2) {
            const uint mis = (uint)as_long(row) & 3u;
            const __global uint *base = (const __global uint *)(row - mis);
            const uint4 lo = vload4(0, base);
            const uint4 hi = (uint4)(lo.s123, base[4]);
            const uint sh = 8u * mis;
            w = (lo >> sh) | ((hi << 1u) << (31u - sh));
        } else if (vt_mode == 1 || (key0 + DPAS_K <= causal_k && vt_par0 == 0)) {
            w = vload4(0, (const __global uint *)row);
        } else if (key0 < causal_k) {
            #pragma unroll
            for (int i = 0; i < DPAS_K; ++i)
                w[i / 4] |= (uint)(uchar)row[min(key0 + i, causal_k - 1) - key0] << (8 * (i % 4));
        }
        half16 h;
        #pragma unroll
        for (int i = 0; i < DPAS_K; ++i)
            h[i] = PLAIN_U4_TO_HALF(w[i / 4], 8 * (i % 4) + nsh);
    #else
        const __global char *row = (const __global char *)V + (size_t)(col + g * SUBGROUP_SIZE + lane) * ldv + key0;
        half16 h = (half16)0.0h;
        if (vt_mode == 2) {
            const uint mis = (uint)as_long(row) & 3u;
            const __global uint *base = (const __global uint *)(row - mis);
            const uint4 lo = vload4(0, base);
            const uint4 hi = (uint4)(lo.s123, base[4]);
            const uint sh = 8u * mis;
            // (x << 1) << (31 - sh) is x << (32 - sh), and 0 at sh 0 (one shift by 32 would shift by 0)
            h = convert_half16(as_char16((lo >> sh) | ((hi << 1u) << (31u - sh))));
        } else if (vt_mode == 1 || (key0 + DPAS_K <= causal_k && vt_par0 == 0)) {
            h = convert_half16(as_char16(vload4(0, (const __global uint *)row)));
        } else if (key0 < causal_k) {
            #pragma unroll
            for (int i = 0; i < DPAS_K; ++i)
                h[i] = convert_half(row[min(key0 + i, causal_k - 1) - key0]);
        }
    #endif
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint w = as_uint((half2)(h[2 * i], h[2 * i + 1]));
    #if defined(NEG_SG8) && NEG_SG8 == 40
            w = rotate(w, 16u);  // negative control: each dword's two keys swapped (transposed-V cases must fail)
    #endif
    #if defined(NEG_SG8) && NEG_SG8 == 45
            if (vt_mode == 1)
                w = rotate(w, 16u);  // negative control: the same in unchecked tiles only
    #endif
    #if defined(NEG_SG8) && NEG_SG8 == 52
            if (vt_mode == 2)
                w = rotate(w, 16u);  // negative control: the same in the aligned-dword tiles only
    #endif
            const int d = g * 8 + i;
            if (d < sv_sg_tile_values / 2)
                ((__private uint *)even)[d] = w;
            else
                ((__private uint *)odd)[d - sv_sg_tile_values / 2] = w;
        }
    }
#elif V_TRANSPOSED
    // ldv is the pitch of a value row. Lane l of group g reads value column col + 8 g + l: keys key0..key0 + 15 of its
    // row as 8 dwords (dword i = keys 2i, 2i + 1), the B operand of hpg_sv(); a lane's dwords 8 g + i go to even (the
    // first sv_sg_tile_values / 2) and odd. A row starting 2 B off a dword reads from the dword below and takes its
    // 16th key alone (vt_par0: V's start in halves off a dword, shared by every row with an even ldv). A block
    // reaching causal_k clamps each key to causal_k - 1, as the rows below (vt_par0 == 2: every block); one past it
    // is zero.
    const bool whole = vt_mode == 1 || (key0 + DPAS_K <= causal_k && vt_par0 < 2);  // uniform
    const bool any_mis = vt_mode == 0 && (vt_par0 != 0 || (ldv & 1) != 0);          // uniform
    #pragma unroll
    for (int g = 0; g < sv_sg_tile_values / SUBGROUP_SIZE; ++g) {
        const __global ushort *row = (const __global ushort *)V + (size_t)(col + g * SUBGROUP_SIZE + lane) * ldv + key0;
        uint w[8];
        if (whole) {
            const uint mis = (uint)as_long(row) & 2u;
            const uint8 a = vload8(0, (const __global uint *)((const __global char *)row - mis));
            if (any_mis) {
                const uint e = mis ? (uint)row[DPAS_K - 1] : 0u;
                const uint sh = mis ? 16u : 0u;
                const uint hm = mis ? 0xFFFF0000u : 0u;
                const uint an[9] = {a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7, e};
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    w[i] = (an[i] >> sh) | ((an[i + 1] << 16) & hm);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    w[i] = a[i];
            }
        } else if (key0 < causal_k) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ke = min(key0 + 2 * i, causal_k - 1) - key0;
                const int ko = min(key0 + 2 * i + 1, causal_k - 1) - key0;
                w[i] = (uint)row[ke] | ((uint)row[ko] << 16);
            }
        } else {
            // A block past causal_k (the rest of the last tile; P is 0 there) reads nothing.
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                w[i] = 0u;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
    #if defined(NEG_SG8) && NEG_SG8 == 40
            w[i] = rotate(w[i], 16u);  // negative control: each dword's two keys swapped (transposed-V cases must fail)
    #endif
    #if defined(NEG_SG8) && NEG_SG8 == 45
            if (vt_mode == 1)
                w[i] = rotate(w[i], 16u);  // negative control: the same in unchecked tiles only
    #endif
            const int d = g * 8 + i;
            if (d < sv_sg_tile_values / 2)
                ((__private uint *)even)[d] = w[i];
            else
                ((__private uint *)odd)[d - sv_sg_tile_values / 2] = w[i];
        }
    }
#else
#if HPG_COMP && defined(NEG_SG8) && NEG_SG8 == 28
    // Negative control: the next new token's Vc row, wrapping.
    const int even_row = (key0 + (int)lane * 2 + 1) % causal_k;
    const int odd_row = (key0 + (int)lane * 2 + 2) % causal_k;
#else
    const int even_row = min(key0 + (int)lane * 2, causal_k - 1);
    const int odd_row = min(even_row + 1, causal_k - 1);
#endif
#if IS_PA_MIXED && !HPG_COMP
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
#endif  // V_TRANSPOSED (&& HPG_PC)
}

#if HPG_VT_PAIRS
// hpg_v_rows() of two key blocks, key0 and key0 + 16 (the lane's 32 keys of its row: a 64 B line when the row starts on
// 64 B), into (even0, odd0) and (even1, odd1). Two blocks wholly below causal_k (uniform) take one read per row, the
// others hpg_v_rows() each.
SDPA_OCL_INLINE void FUNC(hpg_vt_rows2)(const __global half *V, const uint ldv, const int key0, const int causal_k,
                                        const uint col, const uint lane, __private HPG_V_T *even0,
                                        __private HPG_V_T *odd0, __private HPG_V_T *even1, __private HPG_V_T *odd1,
                                        const uint vt_par0, const int vt_mode) {
    if (vt_mode == 0 && !(key0 + 2 * DPAS_K <= causal_k && vt_par0 < 2)) {
        FUNC_CALL(hpg_v_rows)(V, ldv, key0, causal_k, col, lane, even0, odd0, vt_par0, vt_mode);
        FUNC_CALL(hpg_v_rows)(V, ldv, key0 + DPAS_K, causal_k, col, lane, even1, odd1, vt_par0, vt_mode);
        return;
    }
    const bool any_mis = vt_mode == 0 && (vt_par0 != 0 || (ldv & 1) != 0);  // uniform
    #pragma unroll
    for (int g = 0; g < sv_sg_tile_values / SUBGROUP_SIZE; ++g) {
        const __global ushort *row = (const __global ushort *)V + (size_t)(col + g * SUBGROUP_SIZE + lane) * ldv + key0;
        const uint mis = (uint)as_long(row) & 2u;
        const uint16 a = vload16(0, (const __global uint *)((const __global char *)row - mis));
        uint w[16];
        if (any_mis) {
            const uint e = mis ? (uint)row[2 * DPAS_K - 1] : 0u;
            const uint sh = mis ? 16u : 0u;
            const uint hm = mis ? 0xFFFF0000u : 0u;
            const uint an[17] = {a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7,
                                 a.s8, a.s9, a.sa, a.sb, a.sc, a.sd, a.se, a.sf, e};
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                w[i] = (an[i] >> sh) | ((an[i + 1] << 16) & hm);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                w[i] = a[i];
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
    #if defined(NEG_SG8) && NEG_SG8 == 40
            w[i] = rotate(w[i], 16u);  // negative control, as in hpg_v_rows()
    #endif
    #if defined(NEG_SG8) && NEG_SG8 == 45
            if (vt_mode == 1)
                w[i] = rotate(w[i], 16u);
    #endif
            const int d = g * 8 + i % 8;
            __private uint *even = (__private uint *)(i < 8 ? even0 : even1);
            __private uint *odd = (__private uint *)(i < 8 ? odd0 : odd1);
            if (d < sv_sg_tile_values / 2)
                even[d] = w[i];
            else
                odd[d - sv_sg_tile_values / 2] = w[i];
        }
    }
}
#endif

#if V_TRANSPOSED
// The first HPG_PA_V_LEAD key blocks of the tile at k0 into the ring of hpg_sv().
SDPA_OCL_INLINE void FUNC(hpg_vt_fill)(const __global half *V, const uint ldv, const int k0, const int causal_k,
                                       const uint col, const uint lane, __private HPG_V_T *ring_even,
                                       __private HPG_V_T *ring_odd HPG_VT_ARG) {
#if HPG_VT_PAIRS
    #pragma unroll
    for (int pb = 0; pb < HPG_PA_V_LEAD; pb += 2)
        FUNC_CALL(hpg_vt_rows2)(V, ldv, k0 + pb * DPAS_K, causal_k, col, lane, &ring_even[pb], &ring_odd[pb],
                                &ring_even[pb + 1], &ring_odd[pb + 1], vt_par0, vt_mode);
#else
    #pragma unroll
    for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
        FUNC_CALL(hpg_v_rows)(V, ldv, k0 + pb * DPAS_K, causal_k, col, lane, &ring_even[pb], &ring_odd[pb] HPG_VT);
#endif
}
#endif

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
// (the caller fills slots 0..HPG_PA_V_LEAD - 1 with blocks 0..HPG_PA_V_LEAD - 1). A transposed V: acc[qb][vb] += P^T V,
// lane = value (the same block read of p_slm gives the A operand, as P is stored for it).
SDPA_OCL_INLINE void FUNC(hpg_sv)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                  HPG_PAGES_ARG const __global half *V, const uint ldv, const int k0, const int causal_k,
                                  const uint col, const uint lane, __private HPG_V_T *ring_even,
                                  __private HPG_V_T *ring_odd HPG_VT_ARG) {
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
#if HPG_VT_PAIRS
        // At an odd kb the slots of kb - 1 and kb are free: blocks kb + lead - 1 and kb + lead go there.
        if (kb % 2 == 1 && kb + HPG_PA_V_LEAD - 1 < sv_key_blocks)
            FUNC_CALL(hpg_vt_rows2)(V, ldv, k0 + (kb + HPG_PA_V_LEAD - 1) * DPAS_K, causal_k, col, lane,
                                    &ring_even[(kb - 1) % HPG_PA_V_LEAD], &ring_odd[(kb - 1) % HPG_PA_V_LEAD],
                                    &ring_even[slot], &ring_odd[slot], vt_par0, vt_mode);
#else
        if (kb + HPG_PA_V_LEAD < sv_key_blocks)
            FUNC_CALL(hpg_v_rows)(HPG_PAGES V, ldv, k0 + (kb + HPG_PA_V_LEAD) * DPAS_K, causal_k, col, lane,
                                  &ring_even[slot], &ring_odd[slot] HPG_VT);
#endif
        #pragma unroll
        for (int vb = 0; vb < sv_value_blocks; ++vb) {
#if V_TRANSPOSED
            // B operand: the lane's dwords 8 vb..8 vb + 7 (value column 8 vb + lane, hpg_v_rows()); P is the A operand.
            int8 v_b;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = vb * 8 + j;
                v_b[j] = as_int(d < sv_sg_tile_values / 2 ? ((const __private uint *)&even)[d]
                                                          : ((const __private uint *)&odd)[d - sv_sg_tile_values / 2]);
            }
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                acc[qb][vb] = DPAS_MAD_K16(p_cur[qb], v_b, acc[qb][vb]);
#else
            // A operand: component = value, lane = (key 2 lane, key 2 lane + 1). Slice the 8 values first, then
            // interleave the two keys halfword by halfword.
            const int w = vb * 4;
            const ushort8 e = as_ushort8((uint4)(even[w], even[w + 1], even[w + 2], even[w + 3]));
            const ushort8 o = as_ushort8((uint4)(odd[w], odd[w + 1], odd[w + 2], odd[w + 3]));
            const int8 v_a = as_int8(shuffle2(e, o, (ushort16)(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)));
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                acc[qb][vb] = DPAS_MAD_K16(v_a, p_cur[qb], acc[qb][vb]);
#endif
        }
        if (kb + 1 < sv_key_blocks) {
            #pragma unroll
            for (int qb = 0; qb < sv_score_blocks; ++qb)
                p_cur[qb] = p_next[qb];
        }
    }
}

#if HPG_COMP || HPG_PC
// A compressed row slice of this S*V subgroup: sv_sg_tile_values values, one byte each (i8 the value; u4 the byte of the
// split row whose low or high nibble holds it, PA_V_U4_COL() / PA_V_U4_HI(): a slice never straddles the split, as
// PA_V_ROW_ELEMS and col are multiples of the slice width: 32 and 8 at head 64); a plain 4-bit V (HPG_PC) two values per
// byte, in order.
#  if HPG_PC && PLAIN_KV_U4
#    define HPG_VQ_BYTES (sv_sg_tile_values / 2)
#  else
#    define HPG_VQ_BYTES sv_sg_tile_values
#  endif
#  if HPG_VQ_BYTES == 4
#    define HPG_VQ_T       uint
#    define HPG_VQ_LOAD(p) (*(const __global uint *)(p))
#  elif HPG_VQ_BYTES == 8
#    define HPG_VQ_T       uint2
#    define HPG_VQ_LOAD(p) vload2(0, (const __global uint *)(p))
#  elif HPG_VQ_BYTES == 16
#    define HPG_VQ_T       uint4
#    define HPG_VQ_LOAD(p) vload4(0, (const __global uint *)(p))
#  else
#    define HPG_VQ_T       uint8
#    define HPG_VQ_LOAD(p) vload8(0, (const __global uint *)(p))
#  endif
#endif

#if HPG_PC
// hpg_kq() of a per-channel K: each key's 16 channels of the depth tile (i8 16 bytes, 4-bit 8, byte j = channels 2j and
// 2j + 1, i.e. dword j of the B operand), code - zp of the channel in half (k_zp in SLM, read only with zero points; the
// scale is in Q); HPG_PC_XOR 1152 + code, dword 2j = channels 4j, 4j + 2 and dword 2j + 1 = 4j + 1, 4j + 3 (Q is staged
// in that order). Rows of whole dwords (hpg_plain_prefill_ok()); head 72's last tile reads half, zero past it (Q is 0
// there).
// SDPA_OCL_NEG_SG8=48: each dword's two channels swapped (negative control: only the per-channel cases on this kernel must
// fail).
SDPA_OCL_INLINE void FUNC(hpg_kq_pc)(__private float8 (*scores)[kq_query_blocks], const __global KEY_DATA_T *K,
                                     const uint ldk, const int key_base, const int causal_k, const __local uint *q_slm,
                                     __local uint *transpose, const __local half *k_zp, const uint lane) {
    if (key_base >= causal_k)
        return;
#ifdef HPG_PA_KQ_UNROLL
    __attribute__((opencl_unroll_hint(HPG_PA_KQ_UNROLL)))
#else
    #pragma unroll
#endif
    for (int db = 0; db < DKS_ACTIVE; ++db) {
    #if KEY_ZERO_POINTS && !HPG_PC_XOR
        const half16 zp = vload16(0, k_zp + db * DPAS_K);
    #endif
        int8 k_b[kq_key_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb) {
            const int row = min(key_base + mb * DPAS_ROWS + (int)lane, causal_k - 1);
    #if PLAIN_KV_U4
            const __global KEY_DATA_T *k_row = K + (size_t)row * ldk + db * (DPAS_K / 2);
            uint2 kw;
        #if HPG_TAIL
            if (db == DKS_ACTIVE - 1)
                kw = (uint2)(*(const __global uint *)k_row, 0u);
            else
        #endif
                kw = vload2(0, (const __global uint *)k_row);
            half16 kh;
            #pragma unroll
            for (int j = 0; j < DPAS_K / 2; ++j) {
                kh[2 * j] = PLAIN_U4_TO_HALF(kw[j / 4], 8 * (j % 4));
                kh[2 * j + 1] = PLAIN_U4_TO_HALF(kw[j / 4], 8 * (j % 4) + 4);
            }
    #else
            const __global KEY_DATA_T *k_row = K + (size_t)row * ldk + db * DPAS_K;
            char16 kb;
        #if HPG_TAIL
            if (db == DKS_ACTIVE - 1)
                kb = (char16)(as_char8(vload2(0, (const __global uint *)k_row)), (char8)0);
            else
        #endif
                kb = as_char16(vload4(0, (const __global uint *)k_row));
        #if HPG_PC_XOR
            const uint4 kw = as_uint4(kb);
            uint8 kd;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kd[2 * j] = (kw[j] & 0x00FF00FFu) ^ 0x64806480u;
                kd[2 * j + 1] = ((kw[j] >> 8) & 0x00FF00FFu) ^ 0x64806480u;
            }
            #if defined(NEG_SG8) && NEG_SG8 == 48
            kd = rotate(kd, (uint8)16u);
            #endif
            k_b[mb] = as_int8(kd);
        #else
            half16 kh = convert_half16(kb);
        #endif
    #endif
    #if !HPG_PC_XOR
        #if defined(NEG_SG8) && NEG_SG8 == 48
            kh = kh.s1032547698badcfe;
        #endif
        #if KEY_ZERO_POINTS
            k_b[mb] = as_int8(kh - zp);
        #else
            k_b[mb] = as_int8(kh);
        #endif
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
    // lane = key -> lane = query, as in hpg_kq().
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

// hpg_v_rows() of a per-channel V: the slices of keys key0 + 2 lane and key0 + 2 lane + 1 (clamped to causal_k - 1),
// value columns col.., as bytes (4-bit: from byte col / 2).
SDPA_OCL_INLINE void FUNC(hpg_vpc_rows)(const __global VAL_DATA_T *V, const uint ldv, const int key0, const int causal_k,
                                        const uint col, const uint lane, __private HPG_VQ_T *even, __private HPG_VQ_T *odd) {
    const int even_row = min(key0 + (int)lane * 2, causal_k - 1);
    const int odd_row = min(even_row + 1, causal_k - 1);
#if PLAIN_KV_U4
    *even = HPG_VQ_LOAD(V + (size_t)even_row * ldv + col / 2);
    *odd = HPG_VQ_LOAD(V + (size_t)odd_row * ldv + col / 2);
#else
    *even = HPG_VQ_LOAD(V + (size_t)even_row * ldv + col);
    *odd = HPG_VQ_LOAD(V + (size_t)odd_row * ldv + col);
#endif
}

#if HPG_PC_XOR
// pc_v_off of the kernel: 1152 times this tile's sum of P of each accumulator row over the key subgroups (tsum, written
// before the arrive, rewritten after the next tile's first barrier). SDPA_OCL_NEG_SG8=50: 0, the 1152 stays in (negative
// control).
SDPA_OCL_INLINE void FUNC(hpg_pc_v_off)(__private float *v_off, const __local float *tsum, const int q0, const uint lane) {
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
    #if defined(NEG_SG8) && NEG_SG8 == 50
        v_off[r] = 0.0f;
    #else
        const int query = q0 + r * SUBGROUP_SIZE + (int)lane;
        float tl = 0.0f;
        #pragma unroll
        for (int p = 0; p < kq_sg_per_wg_keys; ++p)
            tl += tsum[query * kq_sg_per_wg_keys + p];
        v_off[r] = 1152.0f * tl;
    #endif
    }
}
#endif

// hpg_sv() of a per-channel V: the A operand is the two rows' codes of each value column in half; the column's zp and scale
// apply to the output in the epilogue.
SDPA_OCL_INLINE void FUNC(hpg_sv_pc)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                     const __global VAL_DATA_T *V, const uint ldv, const int k0, const int causal_k,
                                     const uint col, const uint lane, __private HPG_VQ_T *ring_even,
                                     __private HPG_VQ_T *ring_odd) {
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
        const HPG_VQ_T even = ring_even[slot];
        const HPG_VQ_T odd = ring_odd[slot];
        if (kb + HPG_PA_V_LEAD < sv_key_blocks)
            FUNC_CALL(hpg_vpc_rows)(V, ldv, k0 + (kb + HPG_PA_V_LEAD) * DPAS_K, causal_k, col, lane, &ring_even[slot],
                                    &ring_odd[slot]);
        #pragma unroll
        for (int vb = 0; vb < sv_value_blocks; ++vb) {
            // Values vb * 8.. of the slice for the two keys, then interleaved halfword by halfword as hpg_sv() does.
    #if PLAIN_KV_U4
            const uint we = ((const __private uint *)&even)[vb];
            const uint wo = ((const __private uint *)&odd)[vb];
            half8 he;
            half8 ho;
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                he[m] = PLAIN_U4_TO_HALF(we, 4 * m);
                ho[m] = PLAIN_U4_TO_HALF(wo, 4 * m);
            }
    #else
            const uint2 we = (uint2)(even[2 * vb], even[2 * vb + 1]);
            const uint2 wo = (uint2)(odd[2 * vb], odd[2 * vb + 1]);
        #if HPG_PC_XOR
            const half8 he = as_half8(convert_ushort8(as_uchar8(we)) ^ (ushort8)0x6480);  // 1152 + code
            const half8 ho = as_half8(convert_ushort8(as_uchar8(wo)) ^ (ushort8)0x6480);
        #else
            const half8 he = convert_half8(as_char8(we));
            const half8 ho = convert_half8(as_char8(wo));
        #endif
    #endif
            const int8 v_a = as_int8(shuffle2(he, ho, (ushort16)(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)));
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
#endif

#if HPG_COMP
// S*V of new keys (Vc rows row0.., n of them): the ring of the first blocks, the wait for P, hpg_sv().
SDPA_OCL_INLINE void FUNC(hpg_sv_vc)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                     const __global INPUT3_TYPE *block_indices, const __global half *Vc, const uint ldvc,
                                     const int row0, const int n, const uint col, const uint lane) {
    HPG_V_T v_even[HPG_PA_V_LEAD];
    HPG_V_T v_odd[HPG_PA_V_LEAD];
    #pragma unroll
    for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
        FUNC_CALL(hpg_v_rows)(HPG_PAGES Vc, ldvc, row0 + pb * DPAS_K, n, col, lane, &v_even[pb], &v_odd[pb]);
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    FUNC_CALL(hpg_sv)(acc, p_slm, HPG_PAGES Vc, ldvc, row0, n, col, lane, v_even, v_odd);
}

// hpg_v_rows() of a tile of past keys: the page rows of keys key0 + 2 lane and key0 + 2 lane + 1 (clamped to lim - 1, in
// the page of the last key when the block is past it), value columns col.., and the two keys' zps
// ([PAGED_ATTENTION_BLOCK_SIZE + token] of the page's comp). The scales are in P.
SDPA_OCL_INLINE void FUNC(hpg_vq_rows)(const __global INPUT3_TYPE *block_indices, const __global VAL_DATA_T *V, const int key0,
                                       const int lim, const uint col, const uint lane, __private HPG_VQ_T *even,
                                       __private HPG_VQ_T *odd, __private half2 *zp) {
    const int even_key = min(key0 + (int)lane * 2, lim - 1);
    const int odd_key = min(even_key + 1, lim - 1);
    const int page_key0 = min(key0, (lim - 1) & ~(PAGED_ATTENTION_BLOCK_SIZE - 1));
    const __global VAL_DATA_T *page = V + PA_V_PAGE_OFF(block_indices[page_key0 / PAGED_ATTENTION_BLOCK_SIZE], 0);
    const int te = even_key - page_key0;
    const int to = odd_key - page_key0;
    #if defined(NEG_SG8) && NEG_SG8 == 21
    // Negative control: the zps of the next key pair.
    const int ze = (te + 2) % PAGED_ATTENTION_BLOCK_SIZE;
    const int zo = (to + 2) % PAGED_ATTENTION_BLOCK_SIZE;
    #else
    const int ze = te;
    const int zo = to;
    #endif
    *even = HPG_VQ_LOAD(page + (size_t)te * PA_V_ROW_ELEMS + PA_V_U4_COL((int)col));
    *odd = HPG_VQ_LOAD(page + (size_t)to * PA_V_ROW_ELEMS + PA_V_U4_COL((int)col));
    // One dword: the zps of the pair (ze & ~1, + 1); ze is even unless clamped, and then zo == ze (the pair's other zp may be
    // an unwritten slot's, never taken).
    const __global half *zps = (const __global half *)(page + PA_V_COMP_OFF) + PAGED_ATTENTION_BLOCK_SIZE;
    const half2 zz = as_half2(*(const __global uint *)(zps + (ze & ~1)));
    const half z_e = (ze & 1) ? zz.s1 : zz.s0;
    *zp = (half2)(z_e, zo == ze ? z_e : zz.s1);
}

// hpg_sv() of a tile of past keys: the A operand is the two rows dequantized, (q - zp) in half (the scale is in P).
SDPA_OCL_INLINE void FUNC(hpg_sv_comp)(__private float8 (*acc)[sv_value_blocks], const __local uint *p_slm,
                                       const __global INPUT3_TYPE *block_indices, const __global VAL_DATA_T *V, const int k0,
                                       const int lim, const uint col, const uint lane, __private HPG_VQ_T *ring_even,
                                       __private HPG_VQ_T *ring_odd, __private half2 *ring_zp) {
    #if IS_PA_K_U4
        #if defined(NEG_SG8) && NEG_SG8 == 26
    const uchar v_sh = PA_V_U4_HI((int)col) ? 0 : 4;  // negative control: the other nibble
        #else
    const uchar v_sh = PA_V_U4_HI((int)col) ? 4 : 0;
        #endif
    #endif
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
        const HPG_VQ_T even = ring_even[slot];
        const HPG_VQ_T odd = ring_odd[slot];
        const half2 zp = ring_zp[slot];
        if (kb + HPG_PA_V_LEAD < sv_key_blocks)
            FUNC_CALL(hpg_vq_rows)(block_indices, V, k0 + (kb + HPG_PA_V_LEAD) * DPAS_K, lim, col, lane, &ring_even[slot],
                                   &ring_odd[slot], &ring_zp[slot]);
        #pragma unroll
        for (int vb = 0; vb < sv_value_blocks; ++vb) {
            // Values vb * 8.. of the slice for the two keys, then interleaved halfword by halfword as hpg_sv() does.
            const uint2 we = (uint2)(even[2 * vb], even[2 * vb + 1]);
            const uint2 wo = (uint2)(odd[2 * vb], odd[2 * vb + 1]);
    #if IS_PA_K_U4
            // u4: each nibble goes straight to its slot of the interleaved operand, two values per 16-bit word of each
            // row (U4_TO_HALF, exact).
            const ushort4 ue = as_ushort4(we);
            const ushort4 uo = as_ushort4(wo);
            half16 va;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                va[4 * i] = U4_TO_HALF(ue[i], v_sh) - zp.s0;
                va[4 * i + 1] = U4_TO_HALF(uo[i], v_sh) - zp.s1;
                va[4 * i + 2] = U4_TO_HALF(ue[i], 8 + v_sh) - zp.s0;
                va[4 * i + 3] = U4_TO_HALF(uo[i], 8 + v_sh) - zp.s1;
            }
            const int8 v_a = as_int8(va);
    #else
            const half8 he = convert_half8(as_char8(we));
            const half8 ho = convert_half8(as_char8(wo));
            const int8 v_a = as_int8(shuffle2(he - (half8)zp.s0, ho - (half8)zp.s1,
                                              (ushort16)(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)));
    #endif
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
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, sg_per_wg, 1)))
KERNEL(sdpa_ocl)(OPTIONAL_SHAPE_INFO_ARG
        const global KEY_DATA_T *K,
        const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V,
#if IS_PA_MIXED
        const global QRY_DATA_T *Kc,  // the new tokens' K / V: read by a compressed MIXED only (an f16 page holds them)
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
        const __global int* blocked_indexes_start_and_gws_mapping
#else
        const int d, const int k, const int q
#endif
#if HPG_PC
        // As sdpa_ocl.cl's (get_arguments_desc()): no zero points for a symmetric cache.
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
    #if IS_CAUSAL && CAUSAL_MASK_LOWER_RIGHT && !(defined(NEG_SG8) && NEG_SG8 == 39)
    // NEG_SG8=39 drops the shift: a negative control (the causal cases with kv > q must fail).
    const int wg_p0 = wg_j0 + max(0, k - q);
    #else
    const int wg_p0 = wg_j0;
    #endif
    const uint b1 = get_group_id(2);  // batch
#endif
    // wg_j0 is the workgroup's first query in the subsequence (Q and output rows), wg_p0 its key position (the causal and
    // window bounds and masks): they differ by MIXED's past keys, or by plain SDPA's lower-right shift.

    const uint lane = get_sub_group_local_id();
    const uint sg_ij = get_local_id(1);
    const uint b0 = get_group_id(1);
    const uint b0_kv = b0 / KV_GROUP_SIZE;

    const uint sg_i_kq = sg_ij % kq_sg_per_wg_keys;
    const uint sg_j_kq = sg_ij / kq_sg_per_wg_keys;
    const int sg_i0_kq = sg_i_kq * kq_sg_tile_keys;
    const int sg_j0_kq = sg_j_kq * kq_sg_tile_queries;
    // The S*V subgroup of sg_ij owns the same queries as its KQ subgroup (sg_i_sv == sg_j_kq), so alpha[] applies to
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
#if WITH_ATTN_MASK
    const float mask_sat = SDPA_MASK_SAT_FOR(scale);  // the clamp of every mask term (sdpa_ocl_config.cl)
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
        #if HPG_COMP
    // Kc / Vc: the q new tokens, as PREFILL's K / V (row key - past_len), read with dword loads only, on the host's dword
    // rows (hpg_pa_prefill_ok(): a dynamic padding taken on trust, as PREFILL's K / V).
    const uint ldkc = K_HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldvc = V_HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    Kc += (size_t)subsequence_begin * ldkc + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    Vc += (size_t)subsequence_begin * ldvc + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
        #endif
    #else
    const uint ldk = K_HEAD_SIZE * KV_HEADS_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM;
    const uint ldv = V_HEAD_SIZE * KV_HEADS_NUM + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM;
    K += (size_t)subsequence_begin * ldk + b0_kv * K_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM;
    V += (size_t)subsequence_begin * ldv + b0_kv * V_HEAD_SIZE + INPUT2_PAD_BEFORE_FEATURE_NUM;
    #endif
#else
    // The host admits only layouts whose head dimension is innermost and unpadded, so every row starts at a multiple
    // of the (even) head size; a transposed V is read by value rows (ldv = their pitch, hpg_v_rows()).
    const uint ldq = QRY_S2;
    const uint ldk = KEY_S2;
    #if V_TRANSPOSED
    const uint ldv = VAL_S3;
    #else
    const uint ldv = VAL_S2;
    #endif
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
    // A dynamic padding is taken on trust by the host (dword_token_matrix_ok()); Q's and the mask's are checked here, with
    // two-byte-load fallbacks (an unaligned dword load returns wrong data without an error).
#if defined(NEG_SG8) && NEG_SG8 == 4
    // Not a negative control: forces the fallbacks so the suites cover them (they must still PASS).
    const bool q_dword_ok = false;
    #if WITH_ATTN_MASK
    const bool mask_dword_ok = false;
    #endif
    #if V_TRANSPOSED
    const uint vt_par0 = 2;
    #endif
#else
    const bool q_dword_ok = ((ldq & 1u) == 0) && ((as_long(Q) & 3) == 0);
    #if WITH_ATTN_MASK
    const bool mask_dword_ok = (MSK_D2 == 1 || (MSK_S2 & 1) == 0) && ((as_long(msk) & 3) == 0);
    #endif
    #if V_TRANSPOSED && HPG_PC
    const uint vt_par0 = (((uint)as_long(V) | ldv) & 3u) ? 2u : 0u;  // 2: some i8 value row starts off a dword
    #elif V_TRANSPOSED
    const uint vt_par0 = ((uint)as_long(V) >> 1) & 1;
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
#if HPG_PC_XOR
    local float S_tsum_slm[kq_wg_tile_queries * kq_sg_per_wg_keys];  // this tile's lsum, laid out as S_sum_slm
#endif
    local float S_max_slm[kq_wg_tile_queries];

    for (int qi = sg_ij * SUBGROUP_SIZE + lane; qi < kq_wg_tile_queries; qi += sg_per_wg * SUBGROUP_SIZE)
#ifdef HAS_SINK_INPUT
        S_max_slm[qi] = sink_raw;  // the running max over every tile and subgroup starts at the sink
#else
        S_max_slm[qi] = -INFINITY;
#endif

#if HPG_PC
    const uint k_ch_base = KEY_COMP_OFF((KEY_SCALE_BATCH_NUM == 1) ? 0 : b1_k, (KEY_SCALE_FEATURE_NUM == 1) ? 0 : b0_kv, 0, 0);
    const uint v_ch_base = VAL_COMP_OFF((VAL_SCALE_BATCH_NUM == 1) ? 0 : b1_v, (VAL_SCALE_FEATURE_NUM == 1) ? 0 : b0_kv, 0, 0);
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
#if HPG_PC
        // Times the K scale of each channel (0 past the head), rounded to half once: the K reads take the code.
        {
            half16 q_lo = as_half16(qq.lo);
            half16 q_hi = as_half16(qq.hi);
            #pragma unroll
            for (int e = 0; e < DPAS_K; ++e) {
                const int c_lo = db * DPAS_K + e;
                const int c_hi = c_lo + DPAS_K;
                q_lo[e] *= (c_lo < K_HEAD_SIZE) ? K_COMP_SCALE_TO_HALF(K_scales[k_ch_base + K_COMP_CH_OFF(c_lo)]) : (half)0.0h;
                q_hi[e] *= (c_hi < K_HEAD_SIZE) ? K_COMP_SCALE_TO_HALF(K_scales[k_ch_base + K_COMP_CH_OFF(c_hi)]) : (half)0.0h;
            }
    #if HPG_PC_XOR && !(defined(NEG_SG8) && NEG_SG8 == 51)
            // The channel order of hpg_kq_pc()'s i8 dwords (SDPA_OCL_NEG_SG8=51: left out, negative control).
            q_lo = q_lo.s021346578a9bcedf;
            q_hi = q_hi.s021346578a9bcedf;
    #endif
            qq = (uint16)(as_uint8(q_lo), as_uint8(q_hi));
        }
#endif
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

#if HPG_PC
    // The per-channel K zp of this KV head in SLM (0 past the head; the K scale is in Q), and the V scale / zp of this
    // subgroup's value columns (8 per value block) in registers, as half (an absent zp is 0). SDPA_OCL_NEG_X5 (negative
    // controls, as sdpa_ocl.cl): 1 = the K scale / zp of the neighbouring channel (K_COMP_CH_OFF()), 2 = the V scale / zp of
    // the neighbouring column.
    local half pc_k_zp[DKS_ACTIVE * DPAS_K] __attribute__((aligned(16)));
    #if KEY_ZERO_POINTS && !HPG_PC_XOR
    for (int c = sg_ij * SUBGROUP_SIZE + lane; c < DKS_ACTIVE * DPAS_K; c += sg_per_wg * SUBGROUP_SIZE)
        pc_k_zp[c] = (c < K_HEAD_SIZE) ? convert_half(K_zp[k_ch_base + K_COMP_CH_OFF(c)]) : (half)0.0h;
    #endif
    float8 pc_vsc[sv_value_blocks];
    half8 pc_vzp[sv_value_blocks];
    #pragma unroll
    for (int vb = 0; vb < sv_value_blocks; ++vb) {
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
    #if NEG_X5 == 2
            const uint off = v_ch_base + VAL_COMP_OFF(0, 0, 0, (sv_col + vb * 8 + m) ^ 1);
    #else
            const uint off = v_ch_base + VAL_COMP_OFF(0, 0, 0, sv_col + vb * 8 + m);
    #endif
            pc_vsc[vb][m] = V_COMP_SCALE_TO_FLOAT(V_scales[off]);
    #if VAL_ZERO_POINTS
            pc_vzp[vb][m] = convert_half(V_zp[off]);
    #else
            pc_vzp[vb][m] = (half)0.0h;
    #endif
        }
    }
#endif

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
#if HPG_PC_XOR
    // 1152 times the last tile's sum of P of each accumulator row (lane = query): hpg_sv_pc() added it, the next
    // rescale or the epilogue takes it off, so the accumulator never holds 1152 l (whose f32 rounding grows with k).
    float pc_v_off[sv_score_blocks];
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r)
        pc_v_off[r] = 0.0f;
#endif

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
    // as both hold the query, so the mask stays one unsigned compare (a separate group test spills). The group is scanned
    // within the keys the loop visits, as sdpa_ocl.cl does.
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

#if HPG_COMP
    // The tile crossing past_len ends there (k_chunk), so a tile reads the pages or Kc / Vc, never both; its keys at/past
    // lim are masked as keys past k (sdpa_ocl.cl's split). Workgroup-uniform.
    for (int k0 = window_k0_begin; k0 < causal_k;) {
        int k_chunk = min(causal_k - k0, kq_wg_tile_keys);
    #if !(defined(NEG_SG8) && NEG_SG8 == 29)  // negative control: no split, the new tokens read back from the pages
        if (k0 < past_len && k0 + k_chunk > past_len)
            k_chunk = past_len - k0;
    #endif
        const int lim = k0 + k_chunk;
        const bool from_cache = k0 < past_len;
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == window_k0_begin);
        const bool last = (lim >= causal_k);
#else
    for (int k0 = window_k0_begin; k0 < causal_k; k0 += kq_wg_tile_keys) {
        const int key_base = k0 + sg_i0_kq;
        const bool first = (k0 == window_k0_begin);
        const bool last = (k0 + kq_wg_tile_keys >= causal_k);
#endif

        float8 S_tile[kq_key_blocks][kq_query_blocks];
        #pragma unroll
        for (int mb = 0; mb < kq_key_blocks; ++mb)
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_tile[mb][qb] = (float8)0.0f;

#if HPG_COMP
        // The V scales of a tile of past keys go into P below; 1 for new keys (Vc is f16).
        half v_sc[kq_key_blocks];
        if (from_cache)
            FUNC_CALL(hpg_v_scales)(v_sc, block_indices, V, key_base, lim, lane);
        // A subgroup whose queries are all past q contributes nothing that is stored.
        if (wg_j0 + sg_j0_kq < q)
            FUNC_CALL(hpg_kq_comp)(S_tile, block_indices, K, (const __global half *)Kc, ldkc, from_cache, key_base, lim,
                                   past_len, Q_slm + Q_SLM_OFF(0, sg_j0_kq / SUBGROUP_SIZE),
                                   transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE, lane);

        float k_mask[kq_sg_tile_keys / SUBGROUP_SIZE];
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            k_mask[ii] = (key_base + ii * SUBGROUP_SIZE + (int)lane < lim) ? 0.0f : -INFINITY;
#else
        // A subgroup whose queries are all past q contributes nothing that is stored.
        if (wg_j0 + sg_j0_kq < q)
    #if HPG_PC
            FUNC_CALL(hpg_kq_pc)(S_tile, K, ldk, key_base, causal_k, Q_slm + Q_SLM_OFF(0, sg_j0_kq / SUBGROUP_SIZE),
                                 transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE, pc_k_zp, lane);
    #else
            FUNC_CALL(hpg_kq)(S_tile, HPG_PAGES (const global half *)K, ldk, key_base, causal_k,
                              Q_slm + Q_SLM_OFF(0, sg_j0_kq / SUBGROUP_SIZE),
                              transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE, lane);
    #endif

        float k_mask[kq_sg_tile_keys / SUBGROUP_SIZE];
        #pragma unroll
        for (int ii = 0; ii < kq_sg_tile_keys / SUBGROUP_SIZE; ++ii)
            k_mask[ii] = (key_base + ii * SUBGROUP_SIZE + (int)lane < causal_k) ? 0.0f : -INFINITY;
#endif

        float alpha[kq_query_blocks];
#if !WITH_ATTN_MASK
        // Uniform per subgroup: a tile wholly below the diagonal (every key <= the subgroup's first query, and all keys
        // < k, or < lim for a compressed MIXED) and inside the window of the subgroup's last query has nothing to mask.
        if (
#if IS_CAUSAL
            key_base + kq_sg_tile_keys - 1 <= wg_p0 + sg_j0_kq &&
#endif
#if HPG_COMP
            key_base + kq_sg_tile_keys <= lim
#else
            key_base + kq_sg_tile_keys <= k
#endif
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
                    const float8 mask8 = SDPA_SAT_MASK(FUNC_CALL(hpg_mask8)(OPTIONAL_SHAPE_INFO_TENSOR msk, query,
                                                                            key_base + mb * DPAS_ROWS, mask_dword_ok) * iscale,
                                                       mask_sat);
#endif
                    #pragma unroll
                    for (int mm = 0; mm < DPAS_ROWS; ++mm) {
                        const int key_rel = mb * DPAS_ROWS + mm;
                        float s = S_tile[mb][qb][mm] +
                                  sub_group_broadcast(k_mask[key_rel / SUBGROUP_SIZE], key_rel % SUBGROUP_SIZE);
#if WITH_ATTN_MASK
                        s += mask8[mm];
#endif
#if IS_CAUSAL  // without it only the keys past k are masked (k_mask)
    #if BIDIR_MASK
                        if (!causal_block_clear && (uint)(key_base + key_rel - keep_begin[qb]) >= keep_len[qb])
    #elif SLIDING_WINDOW_SIZE
                        // 0 <= query - key < window as one unsigned compare (the two signed ones spill).
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
                // SDPA_SOFTMAX_EXP (sdpa_ocl_config.cl): with a mask, exp2 saturated to [0, 1] (else a '-inf' mask over
                // the first 128 or more keys of a row and logits near 100 give NaN); native_exp2 without one.
#if HPG_PC_XOR
                // l sums P as S * V reads it (half): the 1152 taken off per tile (pc_v_off) is the accumulator's.
                const float8 e = ok ? SDPA_SOFTMAX_EXP(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
                const float8 p = convert_float8(convert_half8(e));
#else
                const float8 p = ok ? SDPA_SOFTMAX_EXP(S_tile[mb][qb] * scale - m_log2) : (float8)0.0f;
#endif
                lsum += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
#if HPG_COMP
                // Past keys: P times each key's V scale (rounded once); a key at/past lim has scale 0 and P 0.
                float8 pv = p;
                if (from_cache) {
                    #pragma unroll
                    for (int mm = 0; mm < DPAS_ROWS; ++mm)
    #if defined(NEG_SG8) && NEG_SG8 == 20
                        pv[mm] *= convert_float(sub_group_broadcast(v_sc[mb], mm ^ 1));  // negative control: key pairs' scales swapped
    #else
                        pv[mm] *= convert_float(sub_group_broadcast(v_sc[mb], mm));
    #endif
                }
    #define HPG_P_STORE pv
#else
    #define HPG_P_STORE p
#endif
                // Keys mb * 8.. of the subgroup tile: dwords 0..3 (mb even) or 4..7 (mb odd) of the operand rows of
                // 16-key block 2 sg_i_kq + mb / 2. A transposed V: of this lane's (query's) row of the block instead,
                // so hpg_sv()'s block read gives the A operand (lane = key pair, component = query).
#if V_TRANSPOSED
                vstore4(as_uint4(convert_half8(HPG_P_STORE)), 0,
                        S_slm + P_SLM_OFF(sg_i_kq * (kq_sg_tile_keys / DPAS_K) + mb / 2,
                                          sg_j_kq * kq_query_blocks + qb) + lane * 8 + (mb % 2) * 4);
#else
                intel_sub_group_block_write4(
                    S_slm + P_SLM_OFF(sg_i_kq * (kq_sg_tile_keys / DPAS_K) + mb / 2, sg_j_kq * kq_query_blocks + qb) +
                        (mb % 2) * (SUBGROUP_SIZE * 4),
                    as_uint4(convert_half8(HPG_P_STORE)));
#endif
#undef HPG_P_STORE
            }
            S_sum_tile[qb] = a * S_sum_tile[qb] + lsum;
#if HPG_PC_XOR
            S_tsum_slm[(sg_j0_kq + qb * SUBGROUP_SIZE + lane) * kq_sg_per_wg_keys + sg_i_kq] = lsum;
#endif
        }

        if (last) {
            #pragma unroll
            for (int qb = 0; qb < kq_query_blocks; ++qb)
                S_sum_slm[(sg_j0_kq + qb * SUBGROUP_SIZE + lane) * kq_sg_per_wg_keys + sg_i_kq] = S_sum_tile[qb];
        }

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        if (!first) {
#if V_TRANSPOSED
            // The tile is lane = value, component = query (hpg_sv()): alpha of query 8 r + m by broadcast.
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r) {
                float8 a8;
                #pragma unroll
                for (int m = 0; m < 8; ++m)
                    a8[m] = sub_group_broadcast(alpha[r], m);
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= a8;
            }
#else
            #pragma unroll
            for (int r = 0; r < sv_score_blocks; ++r)
    #if HPG_PC_XOR
            {
                const float c = pc_v_off[r] * alpha[r];
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] = A_tile[r][cd] * alpha[r] - c;
            }
    #else
                #pragma unroll
                for (int cd = 0; cd < sv_value_blocks; ++cd)
                    A_tile[r][cd] *= alpha[r];
    #endif
#endif
        }

        // V does not depend on P, so the first blocks are loaded before the wait (see the header).
#if HPG_COMP
        if (from_cache) {
            HPG_VQ_T vq_even[HPG_PA_V_LEAD];
            HPG_VQ_T vq_odd[HPG_PA_V_LEAD];
            half2 vq_zp[HPG_PA_V_LEAD];
            #pragma unroll
            for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
                FUNC_CALL(hpg_vq_rows)(block_indices, V, k0 + pb * DPAS_K, lim, sv_col, lane, &vq_even[pb], &vq_odd[pb],
                                       &vq_zp[pb]);
            intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
            FUNC_CALL(hpg_sv_comp)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), block_indices, V, k0, lim,
                                   sv_col, lane, vq_even, vq_odd, vq_zp);
        } else {
            const int vc_row0 = k0 - past_len;  // Vc rows: key - past_len
            FUNC_CALL(hpg_sv_vc)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), block_indices,
                                 (const global half *)Vc, ldvc, vc_row0, lim - past_len, sv_col, lane);
        }
        k0 += k_chunk;
#elif V_TRANSPOSED
        // Uniform: a tile wholly below causal_k with every V row on a dword reads them without hpg_v_rows()'s per-block
        // checks (vt_mode 1), branches that inside the S*V loop cost more than the reads; the others keep them (0).
        const bool vt_whole = k0 + kq_wg_tile_keys <= causal_k && vt_par0 < 2;
        const bool vt_mis = vt_par0 != 0 || (ldv & 1u) != 0;
    #if HPG_PC
        // A per-channel i8 V with rows off a dword: a whole tile by aligned dwords (vt_mode 2), its last block reading up
        // to 4 bytes past its keys, inside the pitch (SDPA_OCL_NEG_SG8=4 keeps the checked reads).
        #if defined(NEG_SG8) && NEG_SG8 == 4
        const bool vt_whole_mis = false;
        #else
        const bool vt_whole_mis =
            vt_par0 == 2 && k0 + kq_wg_tile_keys <= causal_k && k0 + kq_wg_tile_keys + 4 <= (int)ldv;
        #endif
    #endif
        const __global half *v_in = (const global half *)V;
        const __local uint *p_sv = S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks);
        HPG_V_T v_even[HPG_PA_V_LEAD];
        HPG_V_T v_odd[HPG_PA_V_LEAD];
        if (vt_whole && !vt_mis)
            FUNC_CALL(hpg_vt_fill)(v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 1);
    #if HPG_PC
        else if (vt_whole_mis)
            FUNC_CALL(hpg_vt_fill)(v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 2);
    #endif
        else
            FUNC_CALL(hpg_vt_fill)(v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 0);
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        if (vt_whole && !vt_mis)
            FUNC_CALL(hpg_sv)(A_tile, p_sv, v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 1);
    #if HPG_PC
        else if (vt_whole_mis)
            FUNC_CALL(hpg_sv)(A_tile, p_sv, v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 2);
    #endif
        else
            FUNC_CALL(hpg_sv)(A_tile, p_sv, v_in, ldv, k0, causal_k, sv_col, lane, v_even, v_odd, vt_par0, 0);
#elif HPG_PC
        HPG_VQ_T vq_even[HPG_PA_V_LEAD];
        HPG_VQ_T vq_odd[HPG_PA_V_LEAD];
        #pragma unroll
        for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
            FUNC_CALL(hpg_vpc_rows)(V, ldv, k0 + pb * DPAS_K, causal_k, sv_col, lane, &vq_even[pb], &vq_odd[pb]);
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    #if HPG_PC_XOR
        FUNC_CALL(hpg_pc_v_off)(pc_v_off, S_tsum_slm, sg_i0_sv, lane);
    #endif

        FUNC_CALL(hpg_sv_pc)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), V, ldv, k0, causal_k, sv_col, lane,
                             vq_even, vq_odd);
#else
        HPG_V_T v_even[HPG_PA_V_LEAD];
        HPG_V_T v_odd[HPG_PA_V_LEAD];
        #pragma unroll
        for (int pb = 0; pb < HPG_PA_V_LEAD; ++pb)
            FUNC_CALL(hpg_v_rows)(HPG_PAGES (const global half *)V, ldv, k0 + pb * DPAS_K, causal_k, sv_col, lane,
                                  &v_even[pb], &v_odd[pb] HPG_VT);
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        FUNC_CALL(hpg_sv)(A_tile, S_slm + P_SLM_OFF(0, sg_i_sv * sv_score_blocks), HPG_PAGES (const global half *)V, ldv,
                          k0, causal_k, sv_col, lane, v_even, v_odd HPG_VT);
#endif
    }

#if V_TRANSPOSED
    // The tile of DPAS(P, V) is lane = value, component = query: to lane = query through this subgroup's KQ transpose
    // scratch (its KQ is done), as in hpg_kq().
    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r) {
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd) {
            __local uint *tr = transpose_slm + sg_ij * SUBGROUP_SIZE * SUBGROUP_SIZE;
            intel_sub_group_block_write8(tr, as_uint8(A_tile[r][cd]));
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            A_tile[r][cd] = as_float8(vload8(0, tr + lane * 8));
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
#endif
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
#if HPG_PC_XOR
            // The last tile's 1152 and the zp leave before 1 / l (a row without keys, l 0, stays 0).
    #if defined(NEG_SG8) && NEG_SG8 == 49
            const float8 vz = (float8)0.0f;  // negative control: the V zp nowhere (asymmetric per-channel cases must fail)
    #else
            const float8 vz = convert_float8(pc_vzp[cd]);
    #endif
            const half8 o = convert_half8((A_tile[r][cd] - pc_v_off[r] - vz * l) * inv_l * pc_vsc[cd]);
#elif HPG_PC && VAL_ZERO_POINTS
            // A row without keys (l 0) stays 0.
    #if defined(NEG_SG8) && NEG_SG8 == 49
            const float8 vz = (float8)0.0f;  // negative control: the V zp nowhere (asymmetric per-channel cases must fail)
    #else
            const float8 vz = inv_l > 0.0f ? convert_float8(pc_vzp[cd]) : (float8)0.0f;
    #endif
            const half8 o = convert_half8((A_tile[r][cd] * inv_l - vz) * pc_vsc[cd]);  // component = value column
#elif HPG_PC
            const half8 o = convert_half8(A_tile[r][cd] * inv_l * pc_vsc[cd]);  // component = value column
#else
            const half8 o = convert_half8(A_tile[r][cd] * inv_l);
#endif
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
#undef HPG_COMP
#undef HPG_PC
#undef HPG_PC_XOR
#undef HPG_VQ_T
#undef HPG_VQ_LOAD
#undef HPG_COMP_KQ_UNROLL
#undef P_SLM_OFF
#undef HPG_PA_WINDOW
#undef HPG_V_T
#undef HPG_V_LOAD
#undef HPG_VT_ARG
#undef HPG_VT
#undef HPG_VT_PAIRS
#undef HPG_PAGES_ARG
#undef HPG_PAGES

#define SDPA_OCL_HPG_PA_PREFILL_INL 1
