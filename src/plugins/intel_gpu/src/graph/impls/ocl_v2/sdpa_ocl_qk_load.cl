// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q staging and K tile loaders for sdpa_ocl.cl. Included once, by sdpa_ocl.cl only.

// One Q_slm chunk: the 16 queries from query_base by the DPAS_K channels of depth tile db, packed as
// the KQ B operand (8 dwords per lane, lane == query). q_pack is the caller's loop variable: the u4
// variant writes it element-wise, and a helper-local vector would drop the loop-carried value SROA
// keeps for it, which moved register allocation on the spill-bound u4 head-512 kernel.
#if IS_PA_K_U4
// u4: Q adopts the K page's permuted depth labelling (see PA_K_U4_CHANNEL), paid here once per
// workgroup instead of per k0 tile. A chunk spans its 32-channel window, read as two halves w0/w1,
// and q_pack dword j takes half `par` of dwords 2j and 2j+1, i.e. channels (win+4j+par,
// win+4j+2+par).
SDPA_OCL_INLINE void FUNC(q_chunk_u4)(__private uint8 *q_pack, const __global QRY_DATA_T *Q, const int QD_w,
                                      const int QD_h, const int QD_p, const uint ldq, const int q, const int d,
                                      const int query_base, const int db, const size_t lane) {
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
        (*q_pack)[j] = u4_par ? ((a >> 16) | (b & 0xFFFF0000u)) : ((a & 0x0000FFFFu) | (b << 16));
    }
}
#else
SDPA_OCL_INLINE void FUNC(q_chunk)(__private uint8 *q_pack, const __global QRY_DATA_T *Q, const int QD_w,
                                   const int QD_h, const int QD_p, const uint ldq, const int q, const int d,
                                   const int query_base, const int db, const size_t lane) {
    const int head_base = db * DPAS_K;
#if USE_2D_BLOCK_IO_Q
    if (query_base + SUBGROUP_SIZE <= q && head_base + DPAS_K <= d) {
        intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
            (global void *)Q, QD_w, QD_h, QD_p,
            (int2)(head_base / 2, query_base), (private uint *)q_pack);
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
        *q_pack = as_uint8(as_short16(qv));
    }
}
#endif

// ---- Per-k0-tile K hoists: everything the K tile reads need that depends only on the key, loaded
// once per k0 tile instead of per (depth tile, key).

#ifdef KV_COMPRESSED
// Plain SDPA: per-token K scale and zp of the keys key_base + ii * SUBGROUP_SIZE + lane, one 16-wide
// load each. Kept in half for the bias-trick dequant: zp absorbs the widen bias (+1152.0h), so a byte
// dequants as (as_half(0x6480 ^ byte) - (zp + 1152)) * scale. Keys at/past k get scale 0.
SDPA_OCL_INLINE void FUNC(k_comp_per_key)(OPTIONAL_SHAPE_INFO_ARG __private half *k_scale_lane,
                                          __private half *k_zpb_lane,
                                          const __global KEY_ATTR_SCALES_DATA_T *K_scales,
                                          const __global KEY_ATTR_ZP_DATA_T *K_zp, const uint k_comp_base,
                                          const int key_base, const int k, const size_t lane) {
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
}
#endif

#if IS_PA_MIXED
// block_indices[] entry of each DPAS row-block (an 8-aligned row-block never straddles a 16-key
// page). A row-block at/past k -- key_base can run past k on the last k0 tile -- takes page 0, so the
// lookup stays inside this subsequence's blocks.
SDPA_OCL_INLINE void FUNC(pa_k_pages)(__private uint *k_page, const __global INPUT3_TYPE *block_indices,
                                      const uint base_block_index, const int key_base, const int k) {
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        const int mb_key0 = key_base + mb * DPAS_ROWS;
        k_page[mb] = (mb_key0 < k) ? block_indices[base_block_index + mb_key0 / PAGED_ATTENTION_BLOCK_SIZE]
                                   : 0u;
    }
}

#if IS_PA_K_BY_CHANNEL
// BY_CHANNEL comp is indexed by CHANNEL, and the KQ A operand is K with lane == head dim, so a
// channel's (scale, zp) is a plain per-lane scalar: no sub_group_broadcast in the dequant, unlike
// BY_TOKEN's per-key comp. The pairs are interleaved, so one uint block read at dword db *
// SUBGROUP_SIZE gives lane L channel db * DPAS_K + L. Two guards, both mandatory:
//  - a tile reaching past K_HEAD_SIZE (a partial last tile, or u4's even-rounded DKS_ACTIVE) must
//    not read past the comp region, which is exactly K_HEAD_SIZE dwords;
//  - a key group at/past k had its page clamped to 0, whose comp bytes are arbitrary: sc = zp = 0
//    keeps the dequant finite, and a NaN would survive the -INFINITY mask (NaN + -INFINITY is NaN).
SDPA_OCL_INLINE void FUNC(pa_k_comp_by_channel)(__private half (*k_pa_sc_ch)[DKS_ACTIVE],
                                                __private half (*k_pa_zp_ch)[DKS_ACTIVE],
                                                const __global KEY_DATA_T *K, const __private uint *k_page,
                                                const size_t b0_kv, const int key_base, const int k,
                                                const size_t lane, const int lane_i) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        const global uint *k_comp_ch = (const global uint *)(
            K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv) +
            PA_K_COMP_OFF);
        const bool sc_valid = (key_base + kg * SUBGROUP_SIZE) < k;
    #if IS_PA_K_U4
        // u4: tiles 2g and 2g+1 want the comp of channels (win + 2L) and (win + 2L + 1), which are
        // adjacent, so one uint2 per lane covers the pair -- one coalesced 128-byte span. Same two
        // guards as i8.
        #pragma unroll
        for (int g = 0; g < DKS_ACTIVE / 2; ++g) {
            const int u4_win = g * (2 * DPAS_K);
            uint2 pair2 = (uint2)(0u, 0u);
            if (u4_win + 2 * DPAS_K <= K_HEAD_SIZE) {
                if (sc_valid)
                    pair2 = vload2(lane, k_comp_ch + u4_win);
            } else {
                const int c0 = u4_win + 2 * lane_i;
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
            } else if (db * DPAS_K + lane_i < K_HEAD_SIZE) {
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
// BY_TOKEN comp: per-key scale/zp, indexed by token only, loaded once per page with one 16-wide load
// each (lane L = token L; a 16-key group is exactly one page). The dequant takes each key's value
// with a sub_group_broadcast at a constant lane, which folds into the consumer. Keys at/past k get
// sc = zp = 0: their comp bytes were never written and could be NaN, and the block reads have no
// per-key guard to discard them.
SDPA_OCL_INLINE void FUNC(pa_k_comp_by_token)(__private half *k_pa_sc_lane, __private half *k_pa_zp_lane,
                                              const __global KEY_DATA_T *K, const __private uint *k_page,
                                              const size_t b0_kv, const int key_base, const int k,
                                              const size_t lane, const int lane_i) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        const global half *k_comp = (const global half *)(
            K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv) +
            PA_K_COMP_OFF);
        const bool sc_valid = (key_base + kg * SUBGROUP_SIZE + lane_i) < k;
        k_pa_sc_lane[kg] = sc_valid ? k_comp[lane] : (half)0.0h;
        k_pa_zp_lane[kg] = sc_valid ? k_comp[PAGED_ATTENTION_BLOCK_SIZE + lane] : (half)0.0h;
    }
}
#endif

#if USE_1D_BLOCK_IO_K_PA_U4
// Whole u4 page per key group in PA_PAGE_READS uc16 reads, hoisted out of the db loop: the page bytes
// do not depend on db (a byte is a channel pair). The host only enables it where block2d cannot reach
// (u4 rows of 16 or 32 bytes), which keeps the live page small. A group at/past k reads page 0, which
// is always allocated.
SDPA_OCL_INLINE void FUNC(pa_k_page_read_1d)(__private uchar16 (*k_pg)[PA_PAGE_READS(PA_K_ROW_ELEMS)],
                                             const __global KEY_DATA_T *K, const __private uint *k_page,
                                             const size_t b0_kv) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        const global uchar *k_pg_base = (const global uchar *)(
            K + PA_K_PAGE_OFF(k_page[kg * (SUBGROUP_SIZE / DPAS_ROWS)], b0_kv));
        #pragma unroll
        for (int r = 0; r < PA_PAGE_READS(PA_K_ROW_ELEMS); ++r)
            k_pg[kg][r] = intel_sub_group_block_read_uc16(k_pg_base + r * PA_PAGE_RD_BYTES);
    }
}
#endif
#endif

// ---- K tile of one depth tile db: k_raw[mb] is the KQ A operand of DPAS row-block mb (lane == head
// dim, element == key). Scratch arrays (kt, kw) belong to the caller: the inliner brackets a helper's
// own arrays with lifetime markers the inline code never had, which moved scratch allocation on a
// spill-bound kernel.

#if (USE_2D_BLOCK_IO_KV && !IS_PA_MIXED) || (PA_CUR_KV_F16 && !IS_PA_K_U4)
// f16 [key, head] surface (the K input, or MIXED's Kc) whose row key - y_sub holds key. The _16r
// builtin returns 16 key rows (2 row-blocks), so issue one read per 16-key group: with
// kq_sg_tile_keys == 32 a single read would leave k_raw[2..3] uninitialised.
SDPA_OCL_INLINE void FUNC(k_tile_b2d16)(__private ushort8 *k_raw, const __global void *surf, const int w, const int h,
                                        const int p, const int x0, const int db, const int key_base,
                                        const int y_sub) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        intel_sub_group_2d_block_read_16b_16r16x1c(
            (global void *)surf, w, h, p,
            (int2)(x0 + db * DPAS_K, key_base + kg * SUBGROUP_SIZE - y_sub),
            (private ushort *)&k_raw[kg * (SUBGROUP_SIZE / DPAS_ROWS)]);
    }
}
#endif

#if USE_2D_BLOCK_IO_K_I8
// Plain-SDPA int8 K via the 8-bit VNNI-transform read: row-major [key, head] read at (x = db *
// DPAS_K, y = key_base) gives lane = head with 4 consecutive keys per uint, no shuffle. One read spans
// 32 keys, of which this subgroup uses kq_sg_tile_keys (kq_sg_tile_keys / 4 uints). Bias-trick
// dequant, all in half: extract each key byte with shift+mask (as_char4 would cost a :b
// deinterleave), widen as as_half(0x6480 ^ byte) == byte + 1152, then subtract the folded (zp + 1152)
// and multiply by the scale.
SDPA_OCL_INLINE void FUNC(k_tile_i8_b2d)(__private ushort8 *k_raw, __private uint *kt, const __global KEY_DATA_T *K,
                                         const int KD_w, const int KD_h, const int KD_p,
                                         const __private half *k_scale_lane, const __private half *k_zpb_lane,
                                         const int key_base, const int db) {
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void *)K, KD_w, KD_h, KD_p,
        (int2)(db * DPAS_K, key_base), (private uint *)&kt[0]);
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb)
        k_raw[mb] = (ushort8)0;
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
#endif

#if IS_PA_MIXED && USE_2D_BLOCK_IO_K_PA
// Token-major f16 K page: a [PAGED_ATTENTION_BLOCK_SIZE keys, K_HEAD_SIZE] row-major tile, the [key,
// head] geometry of k_tile_b2d16 with the page as the surface and K_HEAD_SIZE as the pitch. One read
// covers one 16-key page, so loop per key group and take its page from k_page[] (indexed per
// row-block). The height is clamped to the keys the page holds: slots at/past k were never written,
// and a NaN from one would survive the masked-out score. A group entirely at/past k (height <= 0 is
// not a legal read) is zero-filled; reachable because key_base can run past k on the last k0 tile.
SDPA_OCL_INLINE void FUNC(pa_k_tile_b2d16)(__private ushort8 *k_raw, const __global KEY_DATA_T *K,
                                           const __private uint *k_page, const size_t b0_kv, const int key_base,
                                           const int k, const int d, const int db) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
        const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
        const int kp_rows = PA_PAGE_ROWS(k, kg_key0);
        if (kp_rows > 0) {
            // PA_K_PAGE_STRIDE by value (the config #if proves the ADJUSTED_* collapse here), but
            // spelled out: IGC strength-reduces (x * 16) * K_HEAD_SIZE and x * (16 * K_HEAD_SIZE)
            // differently at non-power-of-two heads (48/96), and this is the measured form.
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
}
#endif

#if IS_PA_MIXED && USE_2D_BLOCK_IO_K_PA_I8
// Token-major i8/u4 K page, either quant mode (BY_CHANNEL's data region has BY_TOKEN's geometry; only
// the comp and the dequant index differ): the data is a [PAGED_ATTENTION_BLOCK_SIZE, PA_K_ROW_ELEMS]
// byte tile, so the 8-bit VNNI-transform read lands lane = head with 4 keys per uint. The builtin is
// 32-row only on Xe2 while a page has 16 tokens, so the height is clamped and uints 0..3 are used;
// two key groups cannot share a read (their pages are not adjacent). The dequant is an explicit (q -
// zp) * scale in half, identical to the scalar gather, so SDPA_OCL_K_PA_I8_2D=0 is a clean bisection
// toggle (and the bias trick would round the writer's non-integer zp).
SDPA_OCL_INLINE void FUNC(pa_k_tile_q_b2d)(__private ushort8 *k_raw, __private uint *kt, const __global KEY_DATA_T *K,
                                           const __private uint *k_page,
#if IS_PA_K_BY_CHANNEL
                                           __private half (*k_pa_sc_ch)[DKS_ACTIVE],
                                           __private half (*k_pa_zp_ch)[DKS_ACTIVE],
#else
                                           const __private half *k_pa_sc_lane, const __private half *k_pa_zp_lane,
#endif
                                           const size_t b0_kv, const int key_base, const int k, const int d,
                                           const int db) {
    #pragma unroll
    for (int kg = 0; kg < kq_sg_tile_keys / SUBGROUP_SIZE; ++kg) {
        const int kg_key0 = key_base + kg * SUBGROUP_SIZE;
        const int kg_mb = kg * (SUBGROUP_SIZE / DPAS_ROWS);
        #pragma unroll
        for (int mb = 0; mb < SUBGROUP_SIZE / DPAS_ROWS; ++mb)
            k_raw[kg_mb + mb] = (ushort8)0;
        const int kp_rows = PA_PAGE_ROWS(k, kg_key0);
        if (kp_rows > 0) {
            #if IS_PA_K_U4
            // u4: the row is PA_K_ROW_ELEMS bytes and a byte column is a channel pair, so one read at
            // byte column PA_K_U4_WIN(db)/2 covers the tile pair (db, db^1). The partner tile re-issues
            // the same read (an L1 hit); sharing it would hoist k_raw out of the db loop (128 more
            // live ushorts at head 128). x is in bytes and a multiple of SUBGROUP_SIZE, which meets the
            // 8-bit "multiple of four" rule.
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
            // Per-channel scale/zp are per LANE, so they leave the key loop entirely: one pair for the
            // whole (page, head-dim tile) instead of BY_TOKEN's broadcast per key.
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
                    // The nibble select is lane-UNIFORM (the parity is the tile's, not the lane's), so
                    // it folds into the shift amount rather than a per-lane sel. Unsigned by
                    // construction: the int4 quantizer clamps to [0, 15] with zp = -min*scale, so
                    // there is no CHAR_MIN and no sign extension.
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
}
#endif

#if IS_PA_MIXED && USE_1D_BLOCK_IO_K_PA_U4
// u4 K tile from the page pa_k_page_read_1d hoisted: the same dequant and k_raw writes as the scalar
// gather, only the load differs (a register subscript into the page), so SDPA_OCL_K_PA_1D=0 bisects
// the read alone. The column group is db >> 1, a constant, so PA_PAGE_R/I fold. The gather's per-key
// `head < d && key < k` guard is dropped, as in the block2d reads: keys at/past k only need to be
// FINITE (the mask adds -INFINITY; a nibble is bounded and sc/zp were zeroed), and `head < d` folds
// into the scale ((n - zp) * 0 == 0).
SDPA_OCL_INLINE void FUNC(pa_k_tile_u4_1d)(__private ushort8 *k_raw,
                                           __private uchar16 (*k_pg)[PA_PAGE_READS(PA_K_ROW_ELEMS)],
                                           __private half (*k_pa_sc_ch)[DKS_ACTIVE],
                                           __private half (*k_pa_zp_ch)[DKS_ACTIVE], const int d, const int db,
                                           const int lane_i) {
    const int head = PA_K_U4_CHANNEL(db, lane_i);
    const int k_pg_col = (PA_K_U4_WIN(db) / 2) / SUBGROUP_SIZE;
    const bool head_ok = (head < d);
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        const int kg = mb / (SUBGROUP_SIZE / DPAS_ROWS);
        // Per-channel comp is this lane's own and constant across the row-block's keys, exactly as in
        // the scalar gather.
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
}
#endif

#if IS_PA_K_U4 && PA_CUR_KV_F16
// u4 Kc: tile db needs lane L = channel PA_K_U4_WIN(db) + 2L + PA_K_U4_PAR(db) (the permuted depth
// axis Q_slm is staged in), a stride-2 gather no 16b block read can do. Channels (win + 2L, win + 2L +
// 1) are adjacent halves, i.e. one dword at dword column win/2 + L, so a 32b read over Kc as a DWORD
// surface lands the pair in lane L and the parity is a half-select. Each window is read twice (once
// per parity), still far cheaper than the page dequant. Width/pitch stay in bytes, x is in dwords.
SDPA_OCL_INLINE void FUNC(kc_tile_u4_dword)(__private ushort8 *k_raw, __private uint *kw, const __global half *Kc_b2d,
                                            const int KcD_w_b2d, const int KcD_h, const int KcD_p, const int KcD_x0_dw,
                                            const int key_base, const int past_len, const int db) {
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
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
}
#endif

// ---- Per-key scalar gathers: the fallbacks wherever no block read applies. One message per (key,
// head); a key at/past k (or a head at/past d) leaves its element 0.

#if !IS_PA_MIXED && !USE_2D_BLOCK_IO_K_I8 && !USE_2D_BLOCK_IO_KV
// K input: f16/bf16 as-is, or plain-SDPA i8 with the per-token asymmetric dequant from the hoisted
// scale/zp.
SDPA_OCL_INLINE void FUNC(k_tile_gather)(__private ushort8 *k_raw, const __global KEY_DATA_T *K, const uint ldk,
#ifdef KV_COMPRESSED
                                         const __private half *k_scale_lane, const __private half *k_zpb_lane,
#endif
                                         const int key_base, const int k, const int d, const int db,
                                         const size_t lane) {
    const int head = db * DPAS_K + lane;
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        k_raw[mb] = (ushort8)0;
        #pragma unroll
        for (int key_offset = 0; key_offset < 8; ++key_offset) {
            const int key = key_base + mb * 8 + key_offset;
            #ifdef KV_COMPRESSED
                // sub_group_broadcast is a collective, so it must stay outside the per-lane (head < d)
                // guard; krel is a constant, so it folds.
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
}
#endif

#if IS_PA_MIXED && !USE_2D_BLOCK_IO_K_PA && !USE_2D_BLOCK_IO_K_PA_I8 && !USE_1D_BLOCK_IO_K_PA_U4 && IS_PA_KV_COMPRESSED
// i8/u4 K page: the d-major page (its 16-byte row is under the block2d minimum) and token-major pages
// whose pitch misses the host's block2d rule. PA_K_TOKEN_STRIDE / PA_K_HIDDEN_STRIDE select the
// addressing. The dequant is (q - zp) * scale in half, matching the reference; the plain-SDPA bias
// trick only pays off over a wide transform read.
SDPA_OCL_INLINE void FUNC(pa_k_tile_q_gather)(__private ushort8 *k_raw, const __global KEY_DATA_T *K,
                                              const __private uint *k_page,
#if IS_PA_K_BY_CHANNEL
                                              __private half (*k_pa_sc_ch)[DKS_ACTIVE],
                                              __private half (*k_pa_zp_ch)[DKS_ACTIVE],
#else
                                              const __private half *k_pa_sc_lane, const __private half *k_pa_zp_lane,
#endif
                                              const size_t b0_kv, const int key_base, const int k, const int d,
                                              const int db, const size_t lane, const int lane_i) {
    #if IS_PA_K_U4
    // Permuted depth: head is still the channel (so the `head < d` guard is unchanged), but two
    // channels share a byte, so the address is head >> 1 -- lane-contiguous bytes.
    const int head = PA_K_U4_CHANNEL(db, lane_i);
    const int head_addr = head >> 1;
    #else
    const int head = db * DPAS_K + lane;
    const int head_addr = head;
    #endif
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        k_raw[mb] = (ushort8)0;
        // Page base for this row-block, hoisted above; only the intra-page key offset and the head
        // vary here.
        const size_t mb_page_base =
            PA_K_PAGE_OFF(k_page[mb], b0_kv) +
            (size_t)head_addr * PA_K_HIDDEN_STRIDE;
        #if IS_PA_K_BY_CHANNEL
        // head == db * DPAS_K + lane here too, so the per-channel pair is this lane's own and is
        // constant across the row-block's keys: no broadcast, and it lifts out of the key loop. A
        // row-block maps to key group mb / (SUBGROUP_SIZE / DPAS_ROWS).
        const half k_sc = k_pa_sc_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
        const half k_zp = k_pa_zp_ch[mb / (SUBGROUP_SIZE / DPAS_ROWS)][db];
        #endif
        #pragma unroll
        for (int key_offset = 0; key_offset < DPAS_ROWS; ++key_offset) {
            const int krel = mb * DPAS_ROWS + key_offset;   // key's subgroup-local index
            const int key = key_base + krel;
            #if !IS_PA_K_BY_CHANNEL
            // sub_group_broadcast is a subgroup COLLECTIVE, so it must run on every lane -- keep it
            // outside the per-lane-divergent (head < d) guard below. krel is a compile-time constant
            // in this fully unrolled loop, so the broadcast folds into the consuming add/mul source
            // region rather than emitting a shuffle.
            const half k_sc = sub_group_broadcast(k_pa_sc_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
            const half k_zp = sub_group_broadcast(k_pa_zp_lane[krel / SUBGROUP_SIZE], krel % SUBGROUP_SIZE);
            #endif
            if (head < d && key < k) {
                // key_base is a multiple of PAGED_ATTENTION_BLOCK_SIZE, so the key's token within its
                // page is just its subgroup-local index.
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
}
#endif

#if IS_PA_MIXED && !USE_2D_BLOCK_IO_K_PA && !USE_2D_BLOCK_IO_K_PA_I8 && !USE_1D_BLOCK_IO_K_PA_U4 && !IS_PA_KV_COMPRESSED
// f16 K page (d-major, or token-major with a pitch the block2d rule rejects).
SDPA_OCL_INLINE void FUNC(pa_k_tile_gather)(__private ushort8 *k_raw, const __global KEY_DATA_T *K,
                                            const __private uint *k_page, const size_t b0_kv, const int key_base,
                                            const int k, const int d, const int db, const size_t lane) {
    const int head = db * DPAS_K + lane;
    #pragma unroll
    for (int mb = 0; mb < kq_key_blocks; ++mb) {
        k_raw[mb] = (ushort8)0;
        // Page base for this row-block, hoisted above; only the intra-page key offset and the head
        // vary here.
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
}
#endif

#if IS_PA_K_U4 && PA_CUR_KV_F16
// u4 Kc when the dword read cannot represent an odd half offset from the aligned surface origin: stay
// on the exact Kc source and gather the permuted channels directly. Falling back to the cache here
// would reintroduce quantization error and would use page addressing with a potentially
// non-page-aligned k0.
SDPA_OCL_INLINE void FUNC(kc_tile_u4_gather)(__private ushort8 *k_raw, const __global QRY_DATA_T *Kc, const uint ldk,
                                             const int key_base, const int past_len, const int k0, const int k_chunk,
                                             const int d, const int db, const int lane_i) {
    const int current_head = PA_K_U4_CHANNEL(db, lane_i);
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
#endif

#define SDPA_OCL_QK_LOAD_INL 1
