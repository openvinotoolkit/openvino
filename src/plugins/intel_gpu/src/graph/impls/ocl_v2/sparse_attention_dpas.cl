// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q3 prefill, systolic sparse attention over the union of several tokens' selections, shared by SparsePA (QSA_ADDRESS_MODE 0) and SparseSDPA (1).
//
// The lane axis of the XMX tile is filled with QSA_HEADS_PER_TILE (HT) heads x TT = 16 / HT tokens, head-major
// (lane = head * TT + token), so a tile only needs the UNION of TT per-token selections. The union is produced by a
// TT-way merge of the ascending per-token lists, done in parallel by the lanes < TT with subgroup reductions
// (lane t is the cursor of token t). A work-group is QSA_ATT_WORKERS subgroups splitting Dh / Dv; their QK partial
// tiles are summed through SLM in a fixed order. The XMX operand mapping is documented in qsa_common.cl.
//
// JIT: QSA_HEADS, QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_COMPRESS_RATIO, QSA_BLOCK_TOPK, QSA_SCALE,
//      QSA_ATT_WORKERS, QSA_HEADS_PER_TILE, QSA_ADDRESS_MODE, SUBGROUP_SIZE, (paged) QSA_PAGE_SIZE
// Dispatch: gws = [(QSA_HEADS / HT) * W * SG, tiles * HT, 1], lws = [W * SG, 1, 1]

#define QSA_XMX_WORKERS QSA_ATT_WORKERS
#define QSA_ATTENTION_HELPERS
#include "qsa_common.cl"

#if SUBGROUP_SIZE != 16
#    error "the systolic attention needs a 16-wide subgroup"
#endif

#define DPAS_HT QSA_HEADS_PER_TILE
#define DPAS_TT (16 / QSA_HEADS_PER_TILE)
#define DPAS_BLOCKS (16 / QSA_COMPRESS_RATIO)

#if (QSA_K_HEAD_SIZE % (QSA_XW * 16)) != 0 || (QSA_V_HEAD_SIZE % (QSA_XW * 16)) != 0
#    error "head sizes must split evenly into per-worker XMX tiles"
#endif

// Applies one KV step to the online-softmax state of every column. `keep` bit i says whether row i is visible to
// this lane's column; hidden rows get zero weight.
inline void FUNC(qsa_dpas_softmax_pv)(const __global half* value_cache,
                                      const uint voff,
                                      const bool lv,
                                      const uint keep,
                                      __private float* st,
                                      const uint worker,
                                      __private float* running_max,
                                      __private float* running_sum,
                                      __private float8* ro) {
    unroll_for (uint i = 0; i < 16; i++) {
        st[i] = ((keep >> i) & 1u) ? st[i] : QSA_NEG_INF;
    }
    float tile_max = st[0];
    unroll_for (uint i = 1; i < 16; i++) {
        tile_max = fmax(tile_max, st[i]);
    }
    const float new_max = fmax(*running_max, tile_max);
    const float corr = native_exp2(*running_max - new_max);
    float p[16];
    float tile_sum = 0.0f;
    unroll_for (uint i = 0; i < 16; i++) {
        const float e = native_exp2(st[i] - new_max);
        p[i] = st[i] <= QSA_NEG_INF ? 0.0f : e;
        tile_sum += p[i];
    }
    *running_sum = *running_sum * corr + tile_sum;
    *running_max = new_max;
    unroll_for (uint d = 0; d < QSA_XVT; d++) {
        ro[d] *= corr;
    }
    FUNC_CALL(qsa_xmx_pv)(value_cache, voff, lv, p, worker, ro);
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_dpas)(QSA_ATT_INPUT_ARGS,
                              __global half* output,  // paged: [tokens, Hq*Dv]; dense: [B, Hq, L, Dv]
#if QSA_ADDRESS_MODE == 0
                              const uint num_tiles,
                              const uint num_seqs,
                              const uint q_base,
                              const uint q_pitch
#else
                              const uint num_tiles,
                              const uint seq_len,
                              const uint history_len,
                              const uint k_pitch,
                              const uint v_pitch
#endif
) {
    const uint HQ = QSA_HEADS;
    const uint HKV = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint DV = QSA_V_HEAD_SIZE;
    const uint R = QSA_COMPRESS_RATIO;
    const uint K_TOPK = QSA_BLOCK_TOPK;
    const uint TT = DPAS_TT;

#if QSA_XW > 1
    __local float slm_st[2][QSA_XW][2][8][QSA_SG];
    uint st_buf = 0;
#endif

    const uint worker = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint head_group = get_group_id(0);
    const uint tile = get_group_id(1) / DPAS_HT;
    const uint sub = get_group_id(1) % DPAS_HT;
    if (head_group >= HQ / DPAS_HT || tile >= num_tiles) {
        return;
    }
    const uint head_base = head_group * DPAS_HT;
    // All heads of the group share one KV head - that is what makes the tile legal.
    const uint kv_head = head_base / (HQ / HKV);

    const uint col_tok = lane % TT;
    const uint col_head = lane / TT;

#if QSA_ADDRESS_MODE == 0
    uint seq = 0, tile_begin = 0;
    if (!FUNC_CALL(qsa_resolve_tile_paged)(subsequence_begins, num_seqs, tile, 16, &seq, &tile_begin)) {
        return;
    }
    const uint token0 = tile_begin + sub * TT;
    const uint seq_begin = (uint)subsequence_begins[seq];
    const uint seq_end = (uint)subsequence_begins[seq + 1];
    const uint past_len = (uint)past_lens[seq];
    const uint page_begin = (uint)block_indices_begins[seq];
    uint valid = seq_end > token0 ? seq_end - token0 : 0;
    if (valid > TT) {
        valid = TT;
    }
    if (valid == 0) {
        return;
    }
    const uint first_abs = past_len + token0 - seq_begin;
    const uint abs_pos = col_tok + first_abs;
    const bool lane_valid = col_tok < valid;
    const uint sel_row0 = token0;
    const uint tail_begin = (first_abs + 1) / R * R;
    const uint tail_end = first_abs + valid - 1;
    const uint q_col = q_base + (token0 + col_tok) * q_pitch + (head_base + col_head) * DH;
    const uint out_col = ((token0 + col_tok) * HQ + head_base + col_head) * DV;
    const bool tok_ok = lane < valid;
#else
    const uint tiles_per_row = (seq_len + 16 - 1) / 16;
    const uint batch = tile / tiles_per_row;
    const uint tile_begin = (tile % tiles_per_row) * 16;
    const uint token0 = tile_begin + sub * TT;
    uint valid = seq_len > token0 ? seq_len - token0 : 0;
    if (valid > TT) {
        valid = TT;
    }
    if (valid == 0) {
        return;
    }
    uint kl = (uint)key_length[batch];
    kl = kl < history_len ? kl : history_len;
    const uint shift = history_len - kl;
    const uint pad_q = seq_len - (kl < seq_len ? kl : seq_len);
    const uint abs_pos = col_tok + token0 + kl - seq_len;
    const bool lane_valid = (col_tok < valid) && (col_tok + token0 >= pad_q);
    const uint first_l = token0 > pad_q ? token0 : pad_q;
    const bool any_real = token0 + valid > pad_q;
    const uint sel_row0 = batch * seq_len + token0;
    const uint tail_begin = any_real ? (first_l + kl + 1 - seq_len) / R * R : 1;
    const uint tail_end = any_real ? token0 + valid - 1 + kl - seq_len : 0;
    const uint q_col = (((batch * HQ) + head_base + col_head) * seq_len + token0 + col_tok) * DH;
    const uint out_col = (((batch * HQ) + head_base + col_head) * seq_len + token0 + col_tok) * DV;
    const bool tok_ok = lane < valid && lane + token0 >= pad_q;
#endif
    const uint n_complete = (abs_pos + 1) / R;

    // Q as the B operand: the lane's own (head, token) row slice, pre-scaled by scale * log2(e) in half.
    int8 rq[QSA_XKT];
    unroll_for (uint k = 0; k < QSA_XKT; k++) {
        half16 qh = (half16)((half)0.0f);
        if (col_tok < valid) {
            qh = vload16(0, query + q_col + worker * QSA_XDH + k * 16);
        }
        qh = qh * (half)((float)QSA_SCALE * QSA_LOG2E);
        rq[k] = as_int8(qh);
    }

    float running_max = QSA_NEG_INF;
    float running_sum = 0.0f;
    float8 ro[QSA_XVT];
    unroll_for (uint d = 0; d < QSA_XVT; d++) {
        ro[d] = (float8)(0.0f);
    }

    // Lane t < TT owns the ascending selected list of token t.
    int cnt_l = 0;
    int cur_l = 0;
    int hv = 0x7fffffff;
    if (lane < TT && tok_ok) {
        cnt_l = selected_counts[sel_row0 + lane];
        if (cnt_l > 0) {
            hv = selected_blocks[(sel_row0 + lane) * K_TOPK];
        }
    }

    const uint lane_blk = lane / R;
    const uint lane_rr = lane % R;
    const uint row_mask = (1u << R) - 1u;

    for (;;) {
        // ---- pack the next DPAS_BLOCKS union blocks (minimum over the TT cursor heads) ----
        int blk[DPAS_BLOCKS];
        uint msk[DPAS_BLOCKS];
        bool have = true;
        unroll_for (uint j = 0; j < DPAS_BLOCKS; j++) {
            blk[j] = -1;
            msk[j] = 0;
            if (have) {
                const int next = sub_group_reduce_min(hv);
                if (next == 0x7fffffff) {
                    have = false;
                } else {
                    const bool sel = hv == next;
                    msk[j] = sub_group_reduce_add(sel ? (1u << lane) : 0u);
                    blk[j] = next;
                    if (sel) {
                        cur_l++;
                        hv = cur_l < cnt_l ? selected_blocks[(sel_row0 + lane) * K_TOPK + cur_l] : 0x7fffffff;
                    }
                }
            }
        }
        if (blk[0] < 0) {
            break;
        }
        const bool exhausted = blk[DPAS_BLOCKS - 1] < 0;

        // Lane = KV row of this step; a row is visible to a column iff that column's token selected its block.
        int bl = -1;
        uint keep = 0;
        unroll_for (uint j = 0; j < DPAS_BLOCKS; j++) {
            bl = lane_blk == j ? blk[j] : bl;
            keep |= (lane_valid && ((msk[j] >> col_tok) & 1u)) ? (row_mask << (j * R)) : 0u;
        }
        const bool lv = bl >= 0;
        const uint pos = (uint)bl * R + lane_rr;
        uint koff = 0, voff = 0;
        if (lv) {
            koff = QSA_KROW(pos);
            voff = QSA_VROW(pos);
        }
        const uint valid_kv = sub_group_reduce_add(lv ? 1u : 0u);

        float st[16];
#if QSA_XW > 1
        FUNC_CALL(qsa_xmx_qk)(key_cache, koff, valid_kv, rq, worker, &slm_st[st_buf][0][0][0][0], st);
        st_buf ^= 1;
#else
        FUNC_CALL(qsa_xmx_qk)(key_cache, koff, valid_kv, rq, worker, st);
#endif
        FUNC_CALL(qsa_dpas_softmax_pv)(value_cache, voff, lv, keep, st, worker, &running_max, &running_sum, ro);

        if (exhausted) {
            break;
        }
    }

    // ---- always-visible tail: the union of [n_complete(t) * r, abs_pos(t)] spans at most TT + r - 1 positions ----
    for (uint base = tail_begin; base <= tail_end; base += 16) {
        const uint pos = base + lane;
        const bool lv = pos <= tail_end;
        uint koff = 0, voff = 0;
        if (lv) {
            koff = QSA_KROW(pos);
            voff = QSA_VROW(pos);
        }
        const uint valid_kv = sub_group_reduce_add(lv ? 1u : 0u);

        // Position `base + i` is visible to a column iff n_complete * r <= base + i <= abs_pos.
        uint keep = 0;
        unroll_for (uint i = 0; i < 16; i++) {
            const uint pi = base + i;
            keep |= (lane_valid && n_complete * R <= pi && abs_pos >= pi) ? (1u << i) : 0u;
        }

        float st[16];
#if QSA_XW > 1
        FUNC_CALL(qsa_xmx_qk)(key_cache, koff, valid_kv, rq, worker, &slm_st[st_buf][0][0][0][0], st);
        st_buf ^= 1;
#else
        FUNC_CALL(qsa_xmx_qk)(key_cache, koff, valid_kv, rq, worker, st);
#endif
        FUNC_CALL(qsa_dpas_softmax_pv)(value_cache, voff, lv, keep, st, worker, &running_max, &running_sum, ro);
    }

    // Columns past the end of the sequence belong to no token.
    if (col_tok < valid) {
        const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        unroll_for (uint d = 0; d < QSA_XVT; d++) {
            vstore8(convert_half8_rte(ro[d] * inv), 0, output + out_col + worker * QSA_XDV + d * 8);
        }
    }
}
