// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// V tile loaders for sdpa_ocl.cl. Included once, by sdpa_ocl.cl only.

#if IS_PA_MIXED
// Element offset of the V cache page holding `key`. The comp arrays follow the data rows, so the page
// stride is PAGED_ATTENTION_BLOCK_SIZE * ADJUSTED_V_HEAD_SIZE while the data row pitch stays
// PA_V_ROW_ELEMS. A key at/past k reads page 0, which is always allocated.
SDPA_OCL_INLINE size_t FUNC(pa_v_page_base)(const __global INPUT3_TYPE *block_indices, const uint base_block_index,
                                            const int key, const int k, const size_t b0_kv) {
    return PA_V_PAGE_OFF((key < k) ? block_indices[base_block_index + key / PAGED_ATTENTION_BLOCK_SIZE] : 0u, b0_kv);
}
#endif

#if IS_PA_K_U4 && PA_CUR_KV_F16
// Prefetch this chunk's Vc tiles ahead of the softmax, without retaining a private payload. One
// query partition covers all value columns, so the caller runs it on one of them only and each tile
// is prefetched once.
SDPA_OCL_INLINE void FUNC(vc_prefetch)(const __global half *Vc_b2d, const int VcD_w_b2d, const int VcD_h,
                                       const int VcD_p, const int VcD_x0, const size_t sg_j0_sv, const int dv,
                                       const int k0, const int k_chunk, const int past_len) {
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

// ---- V tile of one cp block (SUBGROUP_SIZE keys): vb[cd] is the S*V B operand of value block cd,
// f16 VNNI (dword key_pair packs keys 2*key_pair and 2*key_pair + 1, lane == value). Scratch arrays
// belong to the caller, as for the K tiles.

#if USE_2D_BLOCK_IO_KV || IS_PA_MIXED
// f16 [key, value] surface (the V input, MIXED's Vc, or one f16 V page) via the 16b VNNI-transform
// read: value block cd at column x0 + sg_j0_sv + cd * SUBGROUP_SIZE, the cp block at row k0 + cp *
// SUBGROUP_SIZE - y_sub. The coordinates are computed here, in the loop, from their leaves: a partial
// sum passed in from the caller changed instruction order (and scheduling) on MIXED Vc reads.
SDPA_OCL_INLINE void FUNC(v_tile_b2d16)(__private int8 *vb, const __global void *surf, const int w, const int h,
                                        const int p, const int x0, const size_t sg_j0_sv, const int k0, const int cp,
                                        const int y_sub) {
    #pragma unroll
    for (int cd = 0; cd < sv_value_blocks; ++cd) {
        intel_sub_group_2d_block_read_transform_16b_16r16x1c(
            (global void *)surf, w, h, p,
            (int2)(x0 + sg_j0_sv + cd * SUBGROUP_SIZE, k0 + cp * SUBGROUP_SIZE - y_sub), (private uint *)&vb[cd]);
    }
}
#endif

#if USE_2D_BLOCK_IO_V_I8 || (IS_PA_MIXED && USE_2D_BLOCK_IO_V_PA_I8)
// Per-key zp of a 16-key block, broadcast into the 4-key groups one 8-bit transform uint holds. The
// zp depends on the key only, so this is hoisted out of the value-block loop.
SDPA_OCL_INLINE void FUNC(zp_bcast16)(__private half4 *zp4, const half zp) {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int k0r = u * 4;
        zp4[u] = (half4)(sub_group_broadcast(zp, k0r + 0),
                         sub_group_broadcast(zp, k0r + 1),
                         sub_group_broadcast(zp, k0r + 2),
                         sub_group_broadcast(zp, k0r + 3));
    }
}
#endif

#if USE_2D_BLOCK_IO_V_I8
// Plain-SDPA int8 V: one _8b_32r16x1c read covers 16 value columns and 32 key rows, i.e. two cp
// blocks, so the caller issues it on even cp only and both use it (uints 0..3 for cp, 4..7 for
// cp + 1). The x2c / x4c variants fetch the subgroup's 32 / 64 value columns in one message, into
// the same block-major layout the x1c loop writes into &vt[cd * 8], so the dequant indexing is
// shared. coord.x must be a multiple of 4 for 8-bit data, which sg_j0_sv is. Columns past dv read
// as 0 and the store drops them.
SDPA_OCL_INLINE void FUNC(v_i8_read)(__private uint *vt, const __global VAL_DATA_T *V, const int VD_w, const int VD_h,
                                     const int VD_p, const size_t sg_j0_sv, const int k0, const int cp) {
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

// Per-token V scale and zp of this cp block. The scale depends only on the key, and pA is already
// lane = key, so it folds into pA with a per-lane multiply instead of being broadcast across V's
// head-dim lanes; zp is a subtraction, so it stays on the V side and is returned. Kept in half:
// V_scales/V_zp are already half, and half arithmetic is bit-identical to the float path over the
// int8 range (verified). The returned zp has the bias-trick widen bias folded in (zp + 1152.0h; bf16
// keeps the raw zp); keys at/past k get scale 0 (and zp 0), which zeroes their product.
SDPA_OCL_INLINE half FUNC(v_i8_comp_fold)(OPTIONAL_SHAPE_INFO_ARG __private short8 *pA,
                                          const __global VAL_ATTR_SCALES_DATA_T *V_scales,
                                          const __global VAL_ATTR_ZP_DATA_T *V_zp, const uint v_comp_base,
                                          const int k0, const int cp, const int k, const size_t lane) {
    const int vs_key = k0 + cp * SUBGROUP_SIZE + lane;
    const uint vs_co = v_comp_base + VAL_COMP_OFF(0, 0, vs_key, 0);
    const half vs_c = (vs_key < k) ? V_scales[vs_co] : (half)0.0f;
#if INPUT0_IS_BF16
    const half vzb_c = (vs_key < k) ? convert_half(V_zp[vs_co]) : (half)0.0f;
#else
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
    return vzb_c;
}

// Dequant of the cp block's half (vt_half) of the paired read into the f16 VNNI operand, with no
// subgroup shuffle: shift+mask byte extract, widen as as_half(0x6480 ^ byte) == byte + 1152, subtract
// the folded zp + 1152 (the scale is already in pA).
SDPA_OCL_INLINE void FUNC(v_i8_dequant)(__private int8 *vb, __private half4 *zpb4, const __private uint *vt,
                                        const int vt_half, const half vzb_c) {
    FUNC_CALL(zp_bcast16)(zpb4, vzb_c);
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
            // f16 VNNI operand: vb[cd][key_pair] packs keys (2*key_pair, 2*key_pair+1), which are
            // exactly deq4.lo / .hi for key_pairs (u*2, u*2+1).
            vb[cd][u * 2 + 0] = as_int(deq4.lo);
            vb[cd][u * 2 + 1] = as_int(deq4.hi);
#endif
        }
    }
}
#endif

#if IS_PA_MIXED && IS_PA_KV_COMPRESSED
// Compressed V page comp of this cp block: per-key scale/zp from the page's comp region
// (PA_V_COMP_OFF: [token], [block_size + token]); a cp block is one page, so lane == token. The scale
// folds into pA as in the plain-SDPA path (f16 only: every PA compressed path assumes f16) and the
// zp is returned for the dequant. Keys at/past k get scale and zp 0.
SDPA_OCL_INLINE half FUNC(pa_v_comp_fold)(__private short8 *pA, const __global VAL_DATA_T *V,
                                          const __global INPUT3_TYPE *block_indices, const uint base_block_index,
                                          const int k0, const int cp, const int k, const size_t b0_kv,
                                          const size_t lane) {
    const int vs_key_pa = k0 + cp * SUBGROUP_SIZE + lane;
    const size_t vs_page_pa =
        FUNC_CALL(pa_v_page_base)(block_indices, base_block_index, vs_key_pa, k, b0_kv);
    const global half *v_comp_pa =
        (const global half *)(V + vs_page_pa + PA_V_COMP_OFF);
    const int vs_tok_pa = vs_key_pa % PAGED_ATTENTION_BLOCK_SIZE;
    const half vs_c_pa = (vs_key_pa < k) ? v_comp_pa[vs_tok_pa] : (half)0.0f;
    const half v_zp_c = (vs_key_pa < k) ? v_comp_pa[PAGED_ATTENTION_BLOCK_SIZE + vs_tok_pa]
                                        : (half)0.0f;

    #pragma unroll
    for (int r = 0; r < sv_score_blocks; ++r)
        pA[r] = as_short8(as_half8(pA[r]) * vs_c_pa);
    return v_zp_c;
}
#endif

#if IS_PA_MIXED && IS_PA_KV_COMPRESSED && USE_2D_BLOCK_IO_V_PA_I8
// i8/u4 V page: the data region is a [PAGED_ATTENTION_BLOCK_SIZE tokens, PA_V_ROW_ELEMS] byte tile
// whose pitch passes the host's block2d rule. The 8b transform is 32-row only on Xe2 while a page
// has 16 tokens, so the height is clamped and uints 0..3 are used; two cp blocks cannot share a read
// (pages not adjacent). The scale is already in pA, so only the zp subtraction happens here.
SDPA_OCL_INLINE void FUNC(pa_v_tile_q_b2d)(__private int8 *vb, __private uint *vt_pa, __private half4 *vzp4,
                                           const __global VAL_DATA_T *V, const size_t v_page_base, const int cp_key0,
                                           const int k, const int dv, const size_t sg_j0_sv, const half v_zp_c) {
    const int vp_rows = PA_PAGE_ROWS(k, cp_key0);
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
                // u4 folds the upper half of the head dim back onto its low twin; the nibble select
                // below picks which one this tile wants. Both the base and PA_V_ROW_ELEMS are
                // multiples of SUBGROUP_SIZE, so a 16-lane tile never straddles the split. Identity
                // for i8.
                (int2)(PA_V_U4_COL(vcol), 0),
                (private uint *)&vt_pa[cd * 8]);
        }
    } else {
        #pragma unroll
        for (int u = 0; u < 8 * sv_value_blocks; ++u)
            vt_pa[u] = 0u;
    }
    FUNC_CALL(zp_bcast16)(vzp4, v_zp_c);
    #pragma unroll
    for (int cd = 0; cd < sv_value_blocks; ++cd) {
        #if IS_PA_K_U4
        // Which nibble this tile's head dims live in. Uniform across the subgroup (the split point is
        // a multiple of SUBGROUP_SIZE), so it folds into the shift amount rather than a per-lane
        // select.
        const int v_hi = PA_V_U4_HI(sg_j0_sv + cd * SUBGROUP_SIZE);
        #endif
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint w = vt_pa[cd * 8 + u];
            // Each uint packs 4 consecutive tokens as signed bytes, token u*4+b in byte b -- the same
            // packing the plain-SDPA i8 V path decodes.
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
            // f16 VNNI operand: vb[cd][key_pair] packs keys (2*kp, 2*kp+1), and deq4 already holds
            // keys u*4..u*4+3 in order, so .lo/.hi are exactly key_pairs (u*2, u*2+1).
            vb[cd][u * 2 + 0] = as_int(deq4.lo);
            vb[cd][u * 2 + 1] = as_int(deq4.hi);
        }
    }
}
#endif

#if IS_PA_MIXED && IS_PA_KV_COMPRESSED && USE_1D_BLOCK_IO_V_PA_U4
// u4 V page by whole-page 1D reads: the same dequant and vb writes as the scalar gather, only the
// load differs, so SDPA_OCL_V_PA_1D=0 bisects the read alone. The column group comes from sg_j0_sv
// (not a constant), so the base is biased by it and the index taken at c = 0 (see PA_PAGE_*); that
// makes the read per cd, which costs nothing because sv_value_blocks is 1 for the head sizes this
// path fires on. No per-key `key < k` guard (as in the block2d read): a key at/past k has a
// probability of exactly 0, so its V value only has to be finite -- a nibble is, and v_zp_c is 0
// there.
SDPA_OCL_INLINE void FUNC(pa_v_tile_u4_1d)(__private int8 *vb, __private uchar16 *v_pg, const __global VAL_DATA_T *V,
                                           const size_t v_page_base, const int dv, const size_t sg_j0_sv,
                                           const size_t lane, const half v_zp_c) {
    #pragma unroll
    for (int cd = 0; cd < sv_value_blocks; ++cd) {
        vb[cd] = (int8)0;
        const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
        const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
        // Nibble select as a uniform shift amount: v_hi is subgroup-uniform but not a compile-time
        // constant, and the `?:` form would cost a select on every element.
        const uint v_sh = PA_V_U4_HI(v_base) ? 4u : 0u;
        const global uchar *v_pg_base =
            (const global uchar *)(V + v_page_base) + PA_V_U4_COL(v_base);
        #pragma unroll
        for (int r = 0; r < PA_PAGE_READS(PA_V_ROW_ELEMS); ++r)
            v_pg[r] = intel_sub_group_block_read_uc16(v_pg_base + r * PA_PAGE_RD_BYTES);
        if (value < dv) {
            #pragma unroll
            for (int key_pair = 0; key_pair < DPAS_ROWS; ++key_pair) {
                // The token index IS the key's block-local index (the cp block is one page), spelled
                // as the loop constant because PA_PAGE_R/I need it at compile time.
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
#endif

#if IS_PA_MIXED && !IS_PA_KV_COMPRESSED && USE_2D_BLOCK_IO_V_PA
// f16 V page: a [PAGED_ATTENTION_BLOCK_SIZE tokens, V_HEAD_SIZE] row-major tile, so the 16b
// VNNI-transform read applies with the page as the surface and V_HEAD_SIZE as the pitch. The height
// is clamped to the tokens the page holds (unwritten slots could be NaN, which would survive the zero
// score); a block entirely at/past k (height <= 0 is not a legal read) is zero-filled -- reachable on
// the last k0 tile, and cp_key0 is constant only in cp, so this stays a real (uniform) branch.
SDPA_OCL_INLINE void FUNC(pa_v_tile_b2d16)(__private int8 *vb, const __global VAL_DATA_T *V, const size_t v_page_base,
                                           const int cp_key0, const int k, const int dv, const size_t sg_j0_sv) {
    const int vp_rows = PA_PAGE_ROWS(k, cp_key0);
    if (vp_rows > 0) {
        const global half *Vp = (const global half *)(V + v_page_base);
        const int VP_w = dv * (int)sizeof(half);
        const int VP_p = V_HEAD_SIZE * (int)sizeof(half);
        FUNC_CALL(v_tile_b2d16)(vb, Vp, VP_w, vp_rows, VP_p, 0, sg_j0_sv, 0, 0, 0);   // row 0 of the page
    } else {
        #pragma unroll
        for (int cd = 0; cd < sv_value_blocks; ++cd)
            vb[cd] = (int8)0;
    }
}
#endif

// ---- Per-value scalar gathers: the fallbacks wherever no block read applies. One message per
// (value, key pair); a key at/past k (or a value at/past dv) leaves its element 0.

#if IS_PA_MIXED && IS_PA_KV_COMPRESSED && !USE_2D_BLOCK_IO_V_PA_I8 && !USE_1D_BLOCK_IO_V_PA_U4
// Compressed V page (SDPA_OCL_V_PA_I8_2D=0, or a pitch that fails the block2d rule): the same dequant
// as the block reads, one message per value per key pair.
SDPA_OCL_INLINE void FUNC(pa_v_tile_q_gather)(__private int8 *vb, const __global VAL_DATA_T *V, const size_t v_page_base,
                                              const int cp_key0, const int k, const int dv, const size_t sg_j0_sv,
                                              const size_t lane, const int lane_i, const half v_zp_c) {
    #pragma unroll
    for (int cd = 0; cd < sv_value_blocks; ++cd) {
        vb[cd] = (int8)0;
        const int value = sg_j0_sv + cd * SUBGROUP_SIZE + lane;
        #if IS_PA_K_U4
        // Two head dims share a byte, so the address is the folded byte column plus the lane; the
        // nibble is the TILE's, hence uniform across the subgroup.
        const int v_base = sg_j0_sv + cd * SUBGROUP_SIZE;
        const int v_hi = PA_V_U4_HI(v_base);
        const int v_addr = PA_V_U4_COL(v_base) + lane_i;
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
}
#endif

#if IS_PA_MIXED && !IS_PA_KV_COMPRESSED && !USE_2D_BLOCK_IO_V_PA
// f16 V page with a pitch the block2d rule rejects.
SDPA_OCL_INLINE void FUNC(pa_v_tile_gather)(__private int8 *vb, const __global VAL_DATA_T *V, const size_t v_page_base,
                                            const int cp_key0, const int k, const int dv, const size_t sg_j0_sv,
                                            const size_t lane) {
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
}
#endif

#if !IS_PA_MIXED && !USE_2D_BLOCK_IO_V_I8 && !USE_2D_BLOCK_IO_KV
// V input: f16/bf16 as-is, or plain-SDPA i8 with the per-token (per-kv-head) asymmetric dequant. The
// scale/zp vary per key (token), so they are indexed by key0/key1 here, not by the value (head-dim)
// index.
SDPA_OCL_INLINE void FUNC(v_tile_gather)(OPTIONAL_SHAPE_INFO_ARG __private int8 *vb, const __global VAL_DATA_T *V,
                                         const uint ldv,
#ifdef KV_COMPRESSED
                                         const __global VAL_ATTR_SCALES_DATA_T *V_scales,
                                         const __global VAL_ATTR_ZP_DATA_T *V_zp, const size_t b1, const size_t b0_kv,
#endif
                                         const int k0, const int cp, const int k, const int dv, const size_t sg_j0_sv,
                                         const size_t lane) {
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
}
#endif

#define SDPA_OCL_V_LOAD_INL 1
