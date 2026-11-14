// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// xe_hpg (DG2, SG8) PagedAttention GENERATE reader of a token-major K page: f16, i8 BY_TOKEN, i8 BY_CHANNEL and u4
// BY_CHANNEL (SDPAOclDecodeGenerator with SUBGROUP_SIZE 8). Included by sdpa_ocl_decode.cl in place of its own kernel,
// with the same contract (the SDPA_STAGE_1 intermediates, the partition and sliding-window arithmetic, the dequant
// identities at the top of sdpa_ocl_decode.cl) and the same work split: a workgroup takes Q_PER_WG q-heads of one kv
// group and one partition, KQ splits the partition's pages over the subgroups, S*V splits the head dims. What differs
// is the SG8 DPAS operand form (N = 8 lanes; A = one int per lane, i.e. two consecutive depth halves; B = int8, 16
// depth halves; C = float) and the absence of 2D block IO on xe_hpg, so every read is a per-lane one:
//   KQ   S[key] = sum_d Q[d] * K[key][d]: A = Q (lane l = head dims 2l, 2l + 1 of the tile), B = K (lane = key, the 16
//        head dims of the tile, one per-lane read of the token-major row). A page of 16 keys is two DPAS per tile, keys
//        lane and lane + 8, so the scores come out lane = key, two per lane per page.
//   SV   O[d] = sum_key P[key] * V[key][d]: A = P (lane l = keys 2l, 2l + 1 of the page, read back from slm_p), B = V
//        (lane = head dim, dword j = keys 2j, 2j + 1, read down the page's rows), C lane = head dim.
// The page of a chunk comes from one uniform load of its block-table entry (CHUNK_PAGE below).

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups                          : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short                     : enable

#define DPAS_K 16  // intel_sub_group_f16_f16_matrix_mad_k16 fixes the depth at 16

#define PAGE_HALVES (PAGED_ATTENTION_BLOCK_SIZE / SUBGROUP_SIZE)  // DPAS column sets per page
#define KEYS_PER_SG (SEQ_LEN_PARTITION_SIZE / SG_PER_WG)
#define KEY_GROUPS  (KEYS_PER_SG / PAGED_ATTENTION_BLOCK_SIZE)  // pages per subgroup in KQ
#define K_TILES     (K_HEAD_SIZE / DPAS_K)
#define V_TILES     (V_HEAD_SIZE / SUBGROUP_SIZE)
#define CHUNKS      (SEQ_LEN_PARTITION_SIZE / PAGED_ATTENTION_BLOCK_SIZE)  // pages per partition

// The page of a chunk of this partition: one load of its block-table entry at a workgroup-uniform address. Not the SG16
// form (a per-lane load of page `lane`, then sub_group_broadcast): a partition's 16 pages do not fit one lane each, and
// that form gave wrong pages on DG2 under some code shapes in the SG8 MIXED reader (pa_v_pages in sdpa_ocl, "xe_hpg
// bring-up" in docs/sdpa_ocl.md). A chunk past the last block reads page 0 (always allocated); its keys are masked.
#if defined(NEG_SG8) && NEG_SG8 == 32
// negative control: the page of chunk % 8, so pages 8-15 of a partition are wrong
#    define CHUNK_PAGE(chunk) FUNC_CALL(chunk_page)(block_indices, base_block_index, chunk_block0, total_blocks_num, (chunk) % SUBGROUP_SIZE)
#else
#    define CHUNK_PAGE(chunk) FUNC_CALL(chunk_page)(block_indices, base_block_index, chunk_block0, total_blocks_num, (chunk))
#endif

// Row pitch of a page's DATA region, in cache-dtype elements (sdpa_ocl_decode.cl): the head size for f16 and i8, the
// packed pitch for u4 (the host supplies both).
#ifndef K_ROW_ELEMS
#    define K_ROW_ELEMS K_HEAD_SIZE
#endif
#ifndef V_ROW_ELEMS
#    define V_ROW_ELEMS V_HEAD_SIZE
#endif
// Offset from a page base to its comp region, in cache-dtype elements (bytes when compressed). BY_TOKEN: per-token f16
// scale at [token], zp at [PAGED_ATTENTION_BLOCK_SIZE + token]; BY_CHANNEL: one (scale, zp) f16 pair, one dword, per
// channel.
#define K_COMP_OFF (K_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)
#define V_COMP_OFF (V_ROW_ELEMS * PAGED_ATTENTION_BLOCK_SIZE)

// u4 V "split" packing: head dim d sits in byte d % V_ROW_ELEMS, the low nibble below V_ROW_ELEMS and the high one
// above, so value tile cd (head dims cd * 8 .. cd * 8 + 7) reads byte column V_TILE_COL(cd) + lane.
#if IS_KV_U4
#    define V_READS       (V_ROW_ELEMS / SUBGROUP_SIZE)
#    define V_TILE_COL(t) (((t) >= V_READS ? (t) - V_READS : (t)) * SUBGROUP_SIZE)
#endif

// int8 / u4 -> f16 widen of the KQ B operand, the identities of sdpa_ocl_decode.cl (one tile's source dwords at a time):
//     as_half(0x6480 ^ b) == b + 1152 for every signed byte b,  as_half(0x6400 | n) == 1024 + n for every nibble n.
#if IS_KV_U4
#    define K_WIDEN_BIAS 1024.0f
#else
#    define K_WIDEN_BIAS 1152.0f
#endif
#define K_WIDEN_DWORDS(dst, src)                                                                                       \
    unroll_for(uint j = 0; j < DPAS_K / 4; ++j) {                                                                      \
        const uint w_ = (src);                                                                                         \
        (dst)[j * 2 + 0] = as_int(((w_ & 0x000000FFu) | ((w_ & 0x0000FF00u) << 8)) ^ 0x64806480u);                   \
        (dst)[j * 2 + 1] = as_int((((w_ >> 16) & 0x000000FFu) | ((w_ >> 8) & 0x00FF0000u)) ^ 0x64806480u);           \
    }
#define K_WIDEN_U4_DWORDS(dst, src)                                                                                    \
    unroll_for(uint j = 0; j < DPAS_K / 8; ++j) {                                                                      \
        const uint w_ = (src);                                                                                         \
        unroll_for(uint b_ = 0; b_ < 4; ++b_) {                                                                        \
            const uint p_ = w_ >> (8 * b_);                                                                            \
            (dst)[j * 4 + b_] = as_int(((p_ & 0x0000000Fu) | ((p_ & 0x000000F0u) << 12)) | 0x64006400u);              \
        }                                                                                                              \
    }

#define HEAD_GROUPS (KV_HEADS_NUM * HEAD_ITERS)

// S*V work split, as in sdpa_ocl_decode.cl: SV_DIM_SGS subgroups split the value tiles, the rest the chunks.
#define SV_KEY_SGS        (SG_PER_WG / SV_DIM_SGS)
#define CHUNKS_PER_KEY_SG (CHUNKS / SV_KEY_SGS)
#define SV_CHUNK_UNROLL   4  // pages per unrolled step of the S*V loop (see there)
#if SV_KEY_SGS > 1
#    define SV_NEEDS_OUTPUT_REDUCTION 1
#endif
#if (KV_HEADS_GROUP_SIZE % Q_PER_WG) != 0
#    define HAS_HEAD_LEFTOVERS 1
#    define SINK_HEAD(m) (head_base + ((m) < heads_this_wg ? (m) : 0))
#else
#    define SINK_HEAD(m) (head_base + (m))
#endif

// M-wide operand / accumulator types; QV() indexes them whether M is 1 (a scalar) or not.
#define A_VEC_TYPE MAKE_VECTOR_TYPE(int, Q_PER_WG)
#define H_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, Q_PER_WG)
#define S_VEC_TYPE MAKE_VECTOR_TYPE(SOFTMAX_ACCUMULATOR_TYPE, Q_PER_WG)
#define H2_TYPE    MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)

#define _QV_1(vec, idx) vec
#define _QV_2(vec, idx) vec[idx]
#define _QV_4(vec, idx) vec[idx]
#define _QV_8(vec, idx) vec[idx]
#define QV(vec, idx) CAT(_QV_, Q_PER_WG)(vec, idx)

#if SUBGROUP_SIZE != 8
#    error "sdpa_ocl_hpg_decode.cl: the SG8 GENERATE reader needs SUBGROUP_SIZE 8"
#endif
#if PAGED_ATTENTION_BLOCK_SIZE != 2 * SUBGROUP_SIZE
// A page is two DPAS column sets (keys lane and lane + 8) in KQ and one DPAS depth (16 keys) in S*V.
#    error "sdpa_ocl_hpg_decode.cl: PAGED_ATTENTION_BLOCK_SIZE must be twice SUBGROUP_SIZE"
#endif
#if Q_PER_WG != 1 && Q_PER_WG != 2 && Q_PER_WG != 4 && Q_PER_WG != 8
#    error "sdpa_ocl_hpg_decode.cl: Q_PER_WG must be 1, 2, 4 or 8"
#endif
#if Q_PER_WG >= 2 * KV_HEADS_GROUP_SIZE
#    error "sdpa_ocl_hpg_decode.cl: Q_PER_WG must be below twice KV_HEADS_GROUP_SIZE"
#endif
#if HEAD_ITERS != ((KV_HEADS_GROUP_SIZE + Q_PER_WG - 1) / Q_PER_WG)
#    error "sdpa_ocl_hpg_decode.cl: HEAD_ITERS must be ceil_div(KV_HEADS_GROUP_SIZE, Q_PER_WG)"
#endif
#if (SG_PER_WG % SV_DIM_SGS) != 0
#    error "sdpa_ocl_hpg_decode.cl: SV_DIM_SGS must divide SG_PER_WG"
#endif
#if (CHUNKS % SV_KEY_SGS) != 0
#    error "sdpa_ocl_hpg_decode.cl: SV_KEY_SGS must divide the partition's chunk count"
#endif
#if SEQ_LEN_PARTITION_SIZE % (SG_PER_WG * PAGED_ATTENTION_BLOCK_SIZE) != 0
#    error "sdpa_ocl_hpg_decode.cl: the partition must split evenly into whole pages per subgroup"
#endif
#if K_HEAD_SIZE % DPAS_K != 0
#    error "sdpa_ocl_hpg_decode.cl: K_HEAD_SIZE must be a multiple of the DPAS depth"
#endif
#if V_HEAD_SIZE % SUBGROUP_SIZE != 0
#    error "sdpa_ocl_hpg_decode.cl: V_HEAD_SIZE must be a multiple of the DPAS N dimension"
#endif
#if SG_PER_WG > SUBGROUP_SIZE
// The cross-subgroup combine reduces one value per subgroup across the lanes of one subgroup.
#    error "sdpa_ocl_hpg_decode.cl: SG_PER_WG must not exceed SUBGROUP_SIZE"
#endif
#if IS_KEY_BY_CHANNEL && !IS_KV_COMPRESSED
#    error "sdpa_ocl_hpg_decode.cl: IS_KEY_BY_CHANNEL requires IS_KV_COMPRESSED"
#endif
#if IS_KEY_BY_CHANNEL && (ADJUSTED_K_HEAD_SIZE != K_HEAD_SIZE)
#    error "sdpa_ocl_hpg_decode.cl: BY_CHANNEL must leave ADJUSTED_K_HEAD_SIZE == K_HEAD_SIZE"
#endif
#if IS_KV_U4 && !(IS_KV_COMPRESSED && IS_KEY_BY_CHANNEL)
#    error "sdpa_ocl_hpg_decode.cl: a u4 cache is compressed with a BY_CHANNEL key"
#endif
#if IS_KV_U4 && ((K_HEAD_SIZE % 2) != 0 || (V_ROW_ELEMS % SUBGROUP_SIZE) != 0)
#    error "sdpa_ocl_hpg_decode.cl: u4 needs an even K_HEAD_SIZE and V_ROW_ELEMS a multiple of SUBGROUP_SIZE"
#endif
#if USE_2D_BLOCK_IO_K || USE_2D_BLOCK_IO_V || PREFETCH_DIST || Q_IN_SLM || K_COMP_PER_READ || V_COMP_IN_SLM || V_TWIN_PAIR
// Xe2 paths of sdpa_ocl_decode.cl; the host sets them all to 0 for SG8.
#    error "sdpa_ocl_hpg_decode.cl: block IO, prefetch and the low-register layout are not implemented for SG8"
#endif

// One normalised output element (head, value) of this partition: tmp_out when the sequence's window spans several
// partitions, output otherwise (sdpa_ocl_decode.cl).
__attribute__((always_inline)) inline void FUNC(store_out)(__global OUTPUT_TYPE* output, __global OUTPUT_TYPE* tmp_out,
                                                           const uint effective_seq_len, const uint total_partitions_num,
                                                           const uint seq_idx, const uint head,
                                                           const uint partition_idx, const uint value,
                                                           const SOFTMAX_ACCUMULATOR_TYPE o) {
    if (effective_seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1) {
        const size_t tmp_out_offset = (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE * total_partitions_num +
                                      (size_t)head * V_HEAD_SIZE * total_partitions_num +
                                      (size_t)partition_idx * V_HEAD_SIZE + value;
        tmp_out[tmp_out_offset] = TO_OUTPUT_TYPE(o);
    } else {
        const size_t output_offset =
            (size_t)seq_idx * HEADS_NUM * V_HEAD_SIZE + (size_t)head * V_HEAD_SIZE + value;
        output[output_offset] = TO_OUTPUT_TYPE(o);
    }
}

__attribute__((always_inline)) inline uint FUNC(chunk_page)(const __global INPUT4_TYPE* block_indices, const uint base_block_index,
                                                            const uint chunk_block0, const uint total_blocks_num, const uint chunk) {
    const uint block = chunk_block0 + chunk;
    return block < total_blocks_num ? (uint)block_indices[base_block_index + block] : 0u;
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, SG_PER_WG, 1)))
KERNEL(sdpa_ocl_decode)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* block_indices,
    const __global INPUT5_TYPE* block_indices_begins,
#if HAS_SCALE_INPUT
    const __global SCALE_INPUT_TYPE* scale,
#endif
#ifdef HAS_SINK_INPUT
    const __global SINK_DATA_T* sink_ptr,
#endif
    __global OUTPUT_TYPE* output,
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out) {
    const uint lane = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();

    const uint head_group_idx = get_group_id(1) % HEAD_GROUPS;
    const uint seq_idx = get_group_id(1) / HEAD_GROUPS;
    const uint partition_idx = get_group_id(2);
    const uint total_partitions_num = get_num_groups(2);

    const uint seq_len = past_lens[seq_idx] + 1;
    const uint total_blocks_num = CEIL_DIV(seq_len, PAGED_ATTENTION_BLOCK_SIZE);

    // Sliding-window block skip, exactly as sdpa_ocl_decode.cl / pa_sdpa_opt (the host counts partitions the same way).
#if SLIDING_WINDOW_SIZE != 0
    const uint swa_start_block =
        (seq_len > SLIDING_WINDOW_SIZE) ? ((seq_len - SLIDING_WINDOW_SIZE) / PAGED_ATTENTION_BLOCK_SIZE) : 0;
    const uint effective_blocks_num = total_blocks_num - swa_start_block;
#else
    const uint swa_start_block = 0;
    const uint effective_blocks_num = total_blocks_num;
#endif
    const uint swa_start_token = swa_start_block * PAGED_ATTENTION_BLOCK_SIZE;
    const uint effective_seq_len = seq_len - swa_start_token;

    // Workgroup-uniform, so the whole workgroup leaves together and the barriers below are safe.
    if (partition_idx * SEQ_LEN_PARTITION_SIZE >= effective_blocks_num * PAGED_ATTENTION_BLOCK_SIZE) {
        return;
    }

    const uint kv_head_idx = head_group_idx / HEAD_ITERS;
    const uint head_iter = head_group_idx % HEAD_ITERS;
    const uint head_base = kv_head_idx * KV_HEADS_GROUP_SIZE + head_iter * Q_PER_WG;
#ifdef HAS_HEAD_LEFTOVERS
    const uint heads_this_wg = min((uint)(KV_HEADS_GROUP_SIZE - head_iter * Q_PER_WG), (uint)Q_PER_WG);
#endif

    const uint base_block_index = block_indices_begins[seq_idx];

    // ---- Q, scaled: lane l holds head dims (2l, 2l + 1) of every tile, the A operand's dword.
    H2_TYPE q_reg[Q_PER_WG][K_TILES];
    {
        const uint q_row = INPUT0_OFFSET +
                           seq_idx * (K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM);
#ifdef SCALE_VAL
        const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(SCALE_VAL);
#else
        const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(*scale);
#endif
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#ifdef HAS_HEAD_LEFTOVERS
            const uint head = head_base + (m < heads_this_wg ? m : 0);
#else
            const uint head = head_base + m;
#endif
            const uint q_base = q_row + head * K_HEAD_SIZE;
            unroll_for(uint t = 0; t < K_TILES; ++t) {
                q_reg[m][t] = vload2(0, query + q_base + t * DPAS_K + 2 * lane) * scale_val;
            }
        }
    }

#if IS_KV_COMPRESSED && !IS_KEY_BY_CHANNEL
    // sum_d Q[d], the whole BY_TOKEN K zero-point term (zp[key] * sum_d Q[d]).
    SOFTMAX_ACCUMULATOR_TYPE q_sum[Q_PER_WG];
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE acc = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        unroll_for(uint t = 0; t < K_TILES; ++t) {
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(q_reg[m][t].s0) + TO_SOFTMAX_ACCUMULATOR_TYPE(q_reg[m][t].s1);
        }
        q_sum[m] = sub_group_reduce_add(acc);
    }
#endif

    // ---- This subgroup's KEY_GROUPS pages of the partition (absolute token indices).
    const uint sg_key0 = swa_start_token + partition_idx * SEQ_LEN_PARTITION_SIZE + sgid * KEYS_PER_SG;

    // Block index (within the sequence) of the partition's first chunk: the window's skipped prefix added back.
    const uint chunk_block0 = swa_start_block + partition_idx * CHUNKS;

    size_t k_page_off[KEY_GROUPS];
    bool group_in_range[KEY_GROUPS];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const uint chunk = sgid * KEY_GROUPS + g;
        group_in_range[g] = (chunk_block0 + chunk) < total_blocks_num;
        const size_t page = (size_t)CHUNK_PAGE(chunk);
        k_page_off[g] = (page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
    }

#if IS_KV_COMPRESSED && !IS_KEY_BY_CHANNEL
    // Per-key scale and zero point of this subgroup's pages: key h * 8 + lane of page g, the lane's own scores.
    INPUT0_TYPE k_sc[KEY_GROUPS][PAGE_HALVES];
    INPUT0_TYPE k_zp[KEY_GROUPS][PAGE_HALVES];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(key_cache + k_page_off[g] + K_COMP_OFF);
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            k_sc[g][h] = comp[h * SUBGROUP_SIZE + lane];
            k_zp[g][h] = comp[PAGED_ATTENTION_BLOCK_SIZE + h * SUBGROUP_SIZE + lane];
        }
    }
#elif IS_KEY_BY_CHANNEL
    // The zero-point correction sum_d (Q[d] * sc[d]) * (zp[d] + K_WIDEN_BIAS) per (page, head), accumulated per lane in
    // the KQ loop from the f16-rounded product the A operand gets, reduced after it.
    S_VEC_TYPE k_corr_acc[KEY_GROUPS];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        k_corr_acc[g] = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
    }
#endif

    // ---- KQ: per tile and page two DPAS (keys lane and lane + 8) on the same A operand; the g loop is inside the tile
    // loop so the pages' DPAS chains interleave.
    S_VEC_TYPE s[KEY_GROUPS][PAGE_HALVES];
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            s[g][h] = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
        }
    }

    unroll_for(uint t = 0; t < K_TILES; ++t) {
        A_VEC_TYPE a;
#if !IS_KEY_BY_CHANNEL
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            QV(a, m) = as_int(q_reg[m][t]);
        }
#endif
        unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
            int8 kb[PAGE_HALVES];
            unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
#if defined(NEG_SG8) && NEG_SG8 == 30
                const uint key = lane;  // negative control: key lane for both halves (keys 8-15 of a page are wrong)
#else
                const uint key = h * SUBGROUP_SIZE + lane;
#endif
                const __global INPUT1_TYPE* krow = key_cache + k_page_off[g] + key * K_ROW_ELEMS;
#if IS_KV_U4
                // DPAS_K nibbles = DPAS_K / 2 bytes = 2 dwords of the row, channels 2b, 2b + 1 in byte b.
                const uint2 kw = as_uint2(vload8(0, krow + t * (DPAS_K / 2)));
                K_WIDEN_U4_DWORDS(kb[h], kw[j]);
#elif IS_KV_COMPRESSED
                const uint4 kw = as_uint4(vload16(0, krow + t * DPAS_K));
                K_WIDEN_DWORDS(kb[h], kw[j]);
#else
                kb[h] = as_int8(as_short16(vload16(0, (const __global ushort*)(krow + t * DPAS_K))));
#endif
            }
#if IS_KEY_BY_CHANNEL
            // This page's (scale, zp) pairs of channels t * 16 + 2 lane and + 1: the scale rides into the A operand (the
            // page's own, so it is rebuilt per page), the zp into the correction.
            const uint2 kc = vload2(0, (const __global uint*)(key_cache + k_page_off[g] + K_COMP_OFF) + t * DPAS_K + 2 * lane);
#    if defined(NEG_SG8) && NEG_SG8 == 33
            const H2_TYPE p0 = as_half2(kc.s1);  // negative control: the two channels' pairs swapped
            const H2_TYPE p1 = as_half2(kc.s0);
#    else
            const H2_TYPE p0 = as_half2(kc.s0);
            const H2_TYPE p1 = as_half2(kc.s1);
#    endif
            const H2_TYPE sc = (H2_TYPE)(p0.s0, p1.s0);
            const SOFTMAX_ACCUMULATOR_TYPE zp0 = TO_SOFTMAX_ACCUMULATOR_TYPE(p0.s1) + K_WIDEN_BIAS;
            const SOFTMAX_ACCUMULATOR_TYPE zp1 = TO_SOFTMAX_ACCUMULATOR_TYPE(p1.s1) + K_WIDEN_BIAS;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                const H2_TYPE qs = q_reg[m][t] * sc;
                QV(k_corr_acc[g], m) = fma(TO_SOFTMAX_ACCUMULATOR_TYPE(qs.s0), zp0, QV(k_corr_acc[g], m));
                QV(k_corr_acc[g], m) = fma(TO_SOFTMAX_ACCUMULATOR_TYPE(qs.s1), zp1, QV(k_corr_acc[g], m));
                QV(a, m) = as_int(qs);
            }
#endif
            unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
                s[g][h] = intel_sub_group_f16_f16_matrix_mad_k16(a, kb[h], s[g][h]);
            }
        }
    }

#if IS_KEY_BY_CHANNEL
    // The scale is in the A operand; what is left is the zero point with the widen bias, key-independent in a page.
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            const SOFTMAX_ACCUMULATOR_TYPE corr = sub_group_reduce_add(QV(k_corr_acc[g], m));
            unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
                QV(s[g][h], m) -= corr;
            }
        }
    }
#elif IS_KV_COMPRESSED
    // BY_TOKEN: the dequant factors out of the dot product per key (lane), in float so zp is not rounded.
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            const SOFTMAX_ACCUMULATOR_TYPE sc = TO_SOFTMAX_ACCUMULATOR_TYPE(k_sc[g][h]);
            const SOFTMAX_ACCUMULATOR_TYPE zp = TO_SOFTMAX_ACCUMULATOR_TYPE(k_zp[g][h]) + K_WIDEN_BIAS;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#    if defined(NEG_SG8) && NEG_SG8 == 34
                QV(s[g][h], m) = sc * QV(s[g][h], m);  // negative control: the zero-point term dropped
#    else
                QV(s[g][h], m) = sc * (QV(s[g][h], m) - zp * q_sum[m]);
#    endif
            }
        }
    }
#endif

    // ---- Mask (SOFTMAX_ACCUMULATOR_VAL_MIN, see sdpa_ocl_decode.cl), the same for all Q_PER_WG rows.
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            const uint token_idx = sg_key0 + g * PAGED_ATTENTION_BLOCK_SIZE + h * SUBGROUP_SIZE + lane;
            bool masked = !group_in_range[g] || token_idx >= seq_len;
#if SLIDING_WINDOW_SIZE != 0
            masked = masked || (seq_len > SLIDING_WINDOW_SIZE && token_idx < (seq_len - SLIDING_WINDOW_SIZE));
#endif
            if (masked) {
                s[g][h] = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_MIN);
            }
        }
    }

    // ---- Softmax over the partition: subgroup max, exchanged through SLM; probabilities against the partition max.
    S_VEC_TYPE m_sg = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_MIN);
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(m_sg, m) = SOFTMAX_ACCUMULATOR_MAX_FUNC(QV(m_sg, m), QV(s[g][h], m));
            }
        }
    }
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        QV(m_sg, m) = sub_group_reduce_max(QV(m_sg, m));
    }

    // Probabilities by key with the heads as the vector element, as in sdpa_ocl_decode.cl.
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max[SG_PER_WG * Q_PER_WG];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_sum[SG_PER_WG * Q_PER_WG];
    __local H_VEC_TYPE slm_p[SEQ_LEN_PARTITION_SIZE];

    if (lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_max[sgid * Q_PER_WG + m] = QV(m_sg, m);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    S_VEC_TYPE m_wg;
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE my_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (lane < SG_PER_WG) {
            my_max = slm_max[lane * Q_PER_WG + m];
        }
        QV(m_wg, m) = sub_group_reduce_max(my_max);
#ifdef HAS_SINK_INPUT
        // Attention sink: widens the max and the denominator of partition 0 only (sdpa_ocl_decode.cl).
        if (partition_idx == 0) {
            QV(m_wg, m) = SOFTMAX_ACCUMULATOR_MAX_FUNC(QV(m_wg, m), TO_SOFTMAX_ACCUMULATOR_TYPE(sink_ptr[SINK_HEAD(m)]));
        }
#endif
    }

    S_VEC_TYPE l_sg = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
    unroll_for(uint g = 0; g < KEY_GROUPS; ++g) {
        unroll_for(uint h = 0; h < PAGE_HALVES; ++h) {
            const S_VEC_TYPE e = native_exp(s[g][h] - m_wg);
            l_sg += e;
            H_VEC_TYPE pv;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(pv, m) = TO_INPUT0_TYPE(QV(e, m));
            }
            slm_p[sgid * KEYS_PER_SG + g * PAGED_ATTENTION_BLOCK_SIZE + h * SUBGROUP_SIZE + lane] = pv;
        }
    }
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        QV(l_sg, m) = sub_group_reduce_add(QV(l_sg, m));
    }
    if (lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_sum[sgid * Q_PER_WG + m] = QV(l_sg, m);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    S_VEC_TYPE l_wg;
    S_VEC_TYPE inv_l;
    unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
        SOFTMAX_ACCUMULATOR_TYPE my_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        if (lane < SG_PER_WG) {
            my_sum = slm_sum[lane * Q_PER_WG + m];
        }
        SOFTMAX_ACCUMULATOR_TYPE lw = sub_group_reduce_add(my_sum);
#ifdef HAS_SINK_INPUT
        if (partition_idx == 0) {
            lw += native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(sink_ptr[SINK_HEAD(m)]) - QV(m_wg, m));
        }
#endif
        QV(l_wg, m) = lw;
        QV(inv_l, m) = SOFTMAX_ACCUMULATOR_VAL_ONE / lw;
    }

    // ---- S*V: this subgroup owns value tiles {dim_slot, dim_slot + SV_DIM_SGS, ...} over CHUNKS_PER_KEY_SG pages.
    const uint dim_slot = sgid % SV_DIM_SGS;
    const uint key_slot = sgid / SV_DIM_SGS;
#ifdef SV_NEEDS_OUTPUT_REDUCTION
    __local SOFTMAX_ACCUMULATOR_TYPE slm_out[SV_KEY_SGS * Q_PER_WG * V_HEAD_SIZE];
#endif

    for (uint cd = dim_slot; cd < V_TILES; cd += SV_DIM_SGS) {
        S_VEC_TYPE acc = (S_VEC_TYPE)(SOFTMAX_ACCUMULATOR_VAL_ZERO);
        const uint value = cd * SUBGROUP_SIZE + lane;

        // Unrolled by SV_CHUNK_UNROLL, not whole: with every page unrolled IGC is free to hoist all the pages' V reads,
        // and i8 BY_CHANNEL head 128 at M = 4 spilled 0 / 1184 / 12160 bytes on DG2 depending only on how the page
        // index was read; unrolled by 4 or 8, every variant of smoke_paged_attention_decode_tm compiles without spill.
        __attribute__((opencl_unroll_hint(SV_CHUNK_UNROLL))) for (uint i = 0; i < CHUNKS_PER_KEY_SG; ++i) {
            const uint chunk = key_slot * CHUNKS_PER_KEY_SG + i;
            // Clamped to page 0 past the last block; such a chunk's probabilities are zero (or stage 1 drops the
            // whole partition), so nothing it reads reaches the output.
            const size_t page = (size_t)CHUNK_PAGE(chunk);
            const size_t v_page_off = (page * KV_HEADS_NUM + kv_head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;

            // A operand: keys (2 lane, 2 lane + 1) of the page.
#if defined(NEG_SG8) && NEG_SG8 == 31
            H_VEC_TYPE p0 = slm_p[chunk * PAGED_ATTENTION_BLOCK_SIZE + 2 * lane + 1];  // negative control: pair swapped
            H_VEC_TYPE p1 = slm_p[chunk * PAGED_ATTENTION_BLOCK_SIZE + 2 * lane];
#else
            H_VEC_TYPE p0 = slm_p[chunk * PAGED_ATTENTION_BLOCK_SIZE + 2 * lane];
            H_VEC_TYPE p1 = slm_p[chunk * PAGED_ATTENTION_BLOCK_SIZE + 2 * lane + 1];
#endif
#if IS_KV_COMPRESSED
            // Per-key V scale into the probabilities, the zp along the depth; both 0 for a key at / past seq_len, whose
            // comp bytes were never written (sdpa_ocl_decode.cl).
            const uint v_key0 = swa_start_token + partition_idx * SEQ_LEN_PARTITION_SIZE + chunk * PAGED_ATTENTION_BLOCK_SIZE;
            const __global INPUT0_TYPE* v_comp = (const __global INPUT0_TYPE*)(value_cache + v_page_off + V_COMP_OFF);
            const INPUT0_TYPE v_sc0 = (v_key0 + 2 * lane < seq_len) ? v_comp[2 * lane] : (INPUT0_TYPE)0;
#    if defined(NEG_SG8) && NEG_SG8 == 35
            const INPUT0_TYPE v_sc1 = v_sc0;  // negative control: the even key's V scale for both keys
#    else
            const INPUT0_TYPE v_sc1 = (v_key0 + 2 * lane + 1 < seq_len) ? v_comp[2 * lane + 1] : (INPUT0_TYPE)0;
#    endif
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(p0, m) *= v_sc0;
                QV(p1, m) *= v_sc1;
            }
            MAKE_VECTOR_TYPE(INPUT0_TYPE, DPAS_K) vzp;
            unroll_for(uint kk = 0; kk < DPAS_K; ++kk) {
                vzp[kk] = (v_key0 + kk < seq_len) ? v_comp[PAGED_ATTENTION_BLOCK_SIZE + kk] : (INPUT0_TYPE)0;
            }
#endif
            A_VEC_TYPE a;
            unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
                QV(a, m) = as_int((H2_TYPE)(QV(p0, m), QV(p1, m)));
            }

            // B operand, VNNI: dword j packs keys (2j, 2j + 1) of this lane's head dim.
            int8 vb;
#if IS_KV_U4
            const uint v_byte = V_TILE_COL(cd) + lane;
#endif
            unroll_for(uint j = 0; j < DPAS_K / 2; ++j) {
#if IS_KV_U4
                const INPUT2_TYPE b0 = value_cache[v_page_off + (size_t)(2 * j) * V_ROW_ELEMS + v_byte];
                const INPUT2_TYPE b1 = value_cache[v_page_off + (size_t)(2 * j + 1) * V_ROW_ELEMS + v_byte];
                const INPUT0_TYPE v0 = (INPUT0_TYPE)((cd < V_READS) ? (b0 & 0x0F) : (b0 >> 4));
                const INPUT0_TYPE v1 = (INPUT0_TYPE)((cd < V_READS) ? (b1 & 0x0F) : (b1 >> 4));
                vb[j] = as_int((H2_TYPE)(v0 - vzp[2 * j], v1 - vzp[2 * j + 1]));
#else
                const INPUT2_TYPE v0 = value_cache[v_page_off + (size_t)(2 * j) * V_ROW_ELEMS + value];
                const INPUT2_TYPE v1 = value_cache[v_page_off + (size_t)(2 * j + 1) * V_ROW_ELEMS + value];
#    if IS_KV_COMPRESSED
                vb[j] = as_int((H2_TYPE)((INPUT0_TYPE)v0 - vzp[2 * j], (INPUT0_TYPE)v1 - vzp[2 * j + 1]));
#    else
                vb[j] = as_int((H2_TYPE)(v0, v1));
#    endif
#endif
            }
            acc = intel_sub_group_f16_f16_matrix_mad_k16(a, vb, acc);
        }

#ifdef SV_NEEDS_OUTPUT_REDUCTION
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
            slm_out[(key_slot * Q_PER_WG + m) * V_HEAD_SIZE + value] = QV(acc, m);
        }
#else
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#    ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#    endif
            const uint head = head_base + m;
            const SOFTMAX_ACCUMULATOR_TYPE o = QV(acc, m) * QV(inv_l, m);
            FUNC_CALL(store_out)(output, tmp_out, effective_seq_len, total_partitions_num, seq_idx, head, partition_idx, value, o);
        }
#endif
    }

#ifdef SV_NEEDS_OUTPUT_REDUCTION
    // Fewer value tiles than subgroups: the key slots' partials are summed here (every probability already used m_wg).
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint cd = sgid; cd < V_TILES; cd += SG_PER_WG) {
        const uint value = cd * SUBGROUP_SIZE + lane;
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#    ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#    endif
            SOFTMAX_ACCUMULATOR_TYPE o = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            unroll_for(uint ks = 0; ks < SV_KEY_SGS; ++ks) {
                o += slm_out[(ks * Q_PER_WG + m) * V_HEAD_SIZE + value];
            }
            o *= QV(inv_l, m);
            const uint head = head_base + m;
            FUNC_CALL(store_out)(output, tmp_out, effective_seq_len, total_partitions_num, seq_idx, head, partition_idx, value, o);
        }
    }
#endif

    if (effective_seq_len > SEQ_LEN_PARTITION_SIZE && total_partitions_num > 1 && sgid == 0 && lane == 0) {
        unroll_for(uint m = 0; m < Q_PER_WG; ++m) {
#ifdef HAS_HEAD_LEFTOVERS
            if (m >= heads_this_wg) {
                continue;
            }
#endif
            const size_t offset = (size_t)seq_idx * HEADS_NUM * total_partitions_num +
                                  (size_t)(head_base + m) * total_partitions_num + partition_idx;
            exp_sums[offset] = QV(l_wg, m);
            max_logits[offset] = QV(m_wg, m);
        }
    }
}

#define SDPA_OCL_HPG_DECODE_INL 1
