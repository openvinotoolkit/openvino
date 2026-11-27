// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// PagedAttention for the MIXED stage with a few query tokens per subsequence (speculative decoding).
// A work-group processes one subsequence x one KV head x one partition. Each subgroup owns
// TOKENS_PER_SG query tokens x KV_HEADS_GROUP_SIZE heads = 8 rows (the DPAS M tile), so QK^T and P@V
// both run on DPAS with online softmax in registers. The dequantized K/V tile is shared through SLM.
// Partial results per partition are merged by paged_attention_opt.cl (SDPA_STAGE_1, MULTI_TOKENS_PROCESSING).

#include "include/batch_headers/common.cl"

#if HAS_SINK_INPUT || PAGED_ATTENTION_SCORES_OUTPUT
    #error "pa_small_q_opt.cl: sink input and scores output are not supported"
#endif

#if K_HEAD_SIZE != V_HEAD_SIZE || K_HEAD_SIZE % 32 != 0
    #error "pa_small_q_opt.cl: K and V head sizes must be equal and a multiple of 32"
#endif

#if SUBGROUP_SIZE != 16 || PAGED_ATTENTION_BLOCK_SIZE != 16
    #error "pa_small_q_opt.cl: SUBGROUP_SIZE and PAGED_ATTENTION_BLOCK_SIZE must be 16"
#endif

#if TOKENS_PER_SG * KV_HEADS_GROUP_SIZE != 8
    #error "pa_small_q_opt.cl: each subgroup must own exactly 8 query rows"
#endif

#if SEQ_LEN_PARTITION_SIZE % PAGED_ATTENTION_BLOCK_SIZE != 0
    #error "pa_small_q_opt.cl: SEQ_LEN_PARTITION_SIZE must be a multiple of PAGED_ATTENTION_BLOCK_SIZE"
#endif

#define HEAD_SIZE K_HEAD_SIZE
#define BLOCK PAGED_ATTENTION_BLOCK_SIZE
#define CHUNKS (HEAD_SIZE / SUBGROUP_SIZE)
#define ROWS 8
#define QUERY_ROW_STRIDE (K_HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM)

#if IS_KV_COMPRESSED && IS_INT4_COMPRESSED
    #define K_BLOCK_ELEMS (PACKED_ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE)
    #define V_BLOCK_ELEMS (PACKED_ADJUSTED_V_HEAD_SIZE * BLOCK)
#else
    #define K_BLOCK_ELEMS (ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE)
    #define V_BLOCK_ELEMS (ADJUSTED_V_HEAD_SIZE * BLOCK)
#endif

// Writes the K tile transposed to [token][head_dim] so each lane (key token) reads a contiguous DPAS B operand
inline void FUNC(dequant_key)(const __global INPUT1_TYPE* key_block, __local half* dst, uint lid, uint local_size) {
#if !IS_KV_COMPRESSED
    for (uint d = lid; d < HEAD_SIZE; d += local_size) {
        unroll_for (uint t = 0; t < BLOCK; t++) {
            dst[t * HEAD_SIZE + d] = key_block[d * BLOCK + t];
        }
    }
#elif IS_INT4_COMPRESSED && defined(IS_KEY_BY_CHANNEL)
    const __global uchar* key_bytes = (const __global uchar*)key_block;
    for (uint d = lid; d < HEAD_SIZE; d += local_size) {
        const __global uchar* row = key_bytes + d * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(row + PACKED_K_BLOCK_SIZE);
        const float comp_scale = convert_float(comp[0]);
        const float comp_zp = convert_float(comp[1]);
        const ulong packed = as_ulong(vload8(0, row));
        unroll_for (uint i = 0; i < PACKED_K_BLOCK_SIZE; i++) {
            const uint codes = (uint)((packed >> (i * 8)) & 0xfful);
            dst[(2 * i) * HEAD_SIZE + d] = convert_half(((float)(codes & 15u) - comp_zp) * comp_scale);
            dst[(2 * i + 1) * HEAD_SIZE + d] = convert_half(((float)(codes >> 4) - comp_zp) * comp_scale);
        }
    }
#elif IS_INT4_COMPRESSED
    const __global uchar* key_bytes = (const __global uchar*)key_block;
    const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(key_bytes + PACKED_K_HEAD_SIZE * BLOCK);
    for (uint packed_dim = lid; packed_dim < HEAD_SIZE / 2; packed_dim += local_size) {
        unroll_for (uint t = 0; t < BLOCK; t++) {
            const uint codes = key_bytes[packed_dim * BLOCK + t];
            const float comp_scale = convert_float(comp[t]);
            const float comp_zp = convert_float(comp[BLOCK + t]);
            dst[t * HEAD_SIZE + 2 * packed_dim] = convert_half(((float)(codes & 15u) - comp_zp) * comp_scale);
            dst[t * HEAD_SIZE + 2 * packed_dim + 1] = convert_half(((float)(codes >> 4) - comp_zp) * comp_scale);
        }
    }
#elif defined(IS_KEY_BY_CHANNEL)
    const __global char* key_bytes = (const __global char*)key_block;
    for (uint d = lid; d < HEAD_SIZE; d += local_size) {
        const __global char* row = key_bytes + d * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(row + BLOCK);
        const float comp_scale = convert_float(comp[0]);
        const float comp_zp = convert_float(comp[1]);
        unroll_for (uint t = 0; t < BLOCK; t++) {
            dst[t * HEAD_SIZE + d] = convert_half(((float)row[t] - comp_zp) * comp_scale);
        }
    }
#else
    const __global char* key_bytes = (const __global char*)key_block;
    const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(key_bytes + HEAD_SIZE * BLOCK);
    for (uint d = lid; d < HEAD_SIZE; d += local_size) {
        unroll_for (uint t = 0; t < BLOCK; t++) {
            const float comp_scale = convert_float(comp[t]);
            const float comp_zp = convert_float(comp[BLOCK + t]);
            dst[t * HEAD_SIZE + d] = convert_half(((float)key_bytes[d * BLOCK + t] - comp_zp) * comp_scale);
        }
    }
#endif
}

// Writes the V tile transposed to [head_dim][token] so each lane (head dim) reads a contiguous DPAS B operand.
// Rows at or beyond valid_tokens are zeroed: they may hold uninitialized data (NaN/Inf) and P.V would compute 0 * NaN.
inline void FUNC(dequant_value)(const __global INPUT2_TYPE* value_block, __local half* dst, uint lid, uint local_size, uint valid_tokens) {
    for (uint slot = lid; slot < BLOCK * CHUNKS; slot += local_size) {
        const uint token = slot / CHUNKS;
        const uint dim_base = (slot % CHUNKS) * SUBGROUP_SIZE;
        if (token >= valid_tokens) {
            unroll_for (uint j = 0; j < SUBGROUP_SIZE; j++) {
                dst[(dim_base + j) * BLOCK + token] = 0;
            }
            continue;
        }
#if !IS_KV_COMPRESSED
        const __global INPUT2_TYPE* row = value_block + token * HEAD_SIZE;
        unroll_for (uint j = 0; j < SUBGROUP_SIZE; j++) {
            dst[(dim_base + j) * BLOCK + token] = row[dim_base + j];
        }
#elif IS_INT4_COMPRESSED
        const __global uchar* row = (const __global uchar*)value_block + token * PACKED_ADJUSTED_V_HEAD_SIZE;
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(row + PACKED_V_HEAD_SIZE);
        const float comp_scale = convert_float(comp[0]);
        const float comp_zp = convert_float(comp[1]);
        const ulong packed = as_ulong(vload8(0, row + dim_base / 2));
        unroll_for (uint i = 0; i < SUBGROUP_SIZE / 2; i++) {
            const uint codes = (uint)((packed >> (i * 8)) & 0xfful);
            const uint dim = dim_base + 2 * i;
            dst[dim * BLOCK + token] = convert_half(((float)(codes & 15u) - comp_zp) * comp_scale);
            dst[(dim + 1) * BLOCK + token] = convert_half(((float)(codes >> 4) - comp_zp) * comp_scale);
        }
#else
        const __global char* value_bytes = (const __global char*)value_block;
        const __global char* row = value_bytes + token * HEAD_SIZE;
        const __global INPUT0_TYPE* comp = (const __global INPUT0_TYPE*)(value_bytes + HEAD_SIZE * BLOCK);
        const float comp_scale = convert_float(comp[token]);
        const float comp_zp = convert_float(comp[BLOCK + token]);
        unroll_for (uint j = 0; j < SUBGROUP_SIZE; j++) {
            dst[(dim_base + j) * BLOCK + token] = convert_half(((float)row[dim_base + j] - comp_zp) * comp_scale);
        }
#endif
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(pa_small_q_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* block_indices,
    const __global INPUT5_TYPE* block_indices_begins,
    const __global INPUT6_TYPE* subsequence_begins,
#if HAS_SCALE_INPUT
    const __global SCALE_INPUT_TYPE* scale,
#endif
#if HAS_ALIBI
    const __global ALIBI_INPUT_TYPE* alibi_slopes,
#endif
#if HAS_QQ_BIAS
    const __global QQ_BIAS_DATA_T* qq_bias,
    const __global QQ_BIAS_BEGINS_DATA_T* qq_bias_begins,
#endif
    __global OUTPUT_TYPE* output,
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out,
    __global const int* gws_subseq_mapping
) {
    // query: [total_tokens, HEADS_NUM * K_HEAD_SIZE]
    // output: [total_tokens, HEADS_NUM * V_HEAD_SIZE]
    // exp_sums / max_logits: [total_tokens, HEADS_NUM, total_partitions_num]
    // tmp_out: [total_tokens, HEADS_NUM, total_partitions_num, V_HEAD_SIZE]
    const uint lid = get_local_id(2);
    const uint local_size = get_local_size(2);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();
    const uint subsequence_idx = get_group_id(0);
    const uint kv_head_idx = get_group_id(1);
    const uint partition_idx = get_group_id(2);
    const uint total_partitions_num = get_num_groups(2);

    const uint query_begin = subsequence_begins[subsequence_idx];
    const uint query_count = subsequence_begins[subsequence_idx + 1] - query_begin;
    if (query_count == 0)
        return;

    const uint past_len = past_lens[subsequence_idx];
    const uint context_len = past_len + query_count;
    const uint block_table_begin = block_indices_begins[subsequence_idx];
    const uint head_begin = kv_head_idx * KV_HEADS_GROUP_SIZE;
    const uint partition_start = partition_idx * SEQ_LEN_PARTITION_SIZE;

#ifdef SCALE_VAL
    const float scale_val = SCALE_VAL;
#else
    const float scale_val = convert_float(scale[0]);
#endif

#if HAS_QQ_BIAS
    const uint qq_bias_num = qq_bias_begins[subsequence_idx + 1] - qq_bias_begins[subsequence_idx];
    const uint qq_bias_spec_num = (uint)native_sqrt((float)qq_bias_num);
    const uint cumulated_qq_bias_num = qq_bias_begins[subsequence_idx];
#endif

#ifdef SINGLE_KV_SLM
    // K and V of a tile share one buffer: halves SLM per work-group at the cost of two extra barriers per tile
    __local half kv_slm[BLOCK * HEAD_SIZE] __attribute__((aligned(64)));
#else
    // Double-buffered so that dequantization of the next tile does not wait for readers of the current one
    __local half k_slm[2 * BLOCK * HEAD_SIZE] __attribute__((aligned(64)));
    __local half v_slm[2 * HEAD_SIZE * BLOCK] __attribute__((aligned(64)));
#endif

    // DPAS A operand: lane holds head dim (c * SUBGROUP_SIZE + sglid) of each row.
    // Rows beyond query_count clamp to the first token and are masked out below.
    short8 q_cache[CHUNKS];
    unroll_for (uint c = 0; c < CHUNKS; c++) {
        half8 q;
        unroll_for (uint m = 0; m < ROWS; m++) {
            const uint token_off = sgid * TOKENS_PER_SG + m / KV_HEADS_GROUP_SIZE;
            const uint token = query_begin + (token_off < query_count ? token_off : 0);
            const uint head = head_begin + m % KV_HEADS_GROUP_SIZE;
            q[m] = query[INPUT0_OFFSET + token * QUERY_ROW_STRIDE + head * HEAD_SIZE + c * SUBGROUP_SIZE + sglid];
        }
        q_cache[c] = as_short8(q);
    }

    float8 acc[CHUNKS];
    unroll_for (uint c = 0; c < CHUNKS; c++) {
        acc[c] = (float8)(0.0f);
    }
    float8 row_max = (float8)(-INFINITY);
    float8 row_sum = (float8)(0.0f);

    // The first query row defines the union of keys needed by this subsequence's sliding windows.
    uint tile_start = partition_start;
#if SLIDING_WINDOW_SIZE != 0
    const uint first_query_seq_len = past_len + 1;
    if (first_query_seq_len > SLIDING_WINDOW_SIZE) {
        const uint window_start = first_query_seq_len - SLIDING_WINDOW_SIZE;
        tile_start = max(tile_start, (window_start / BLOCK) * BLOCK);
    }
#endif

    uint tile_count = 0;
    const uint partition_end = partition_start + SEQ_LEN_PARTITION_SIZE;
    if (tile_start < context_len && tile_start < partition_end) {
        const uint partition_remaining = partition_end - tile_start;
        tile_count = min((uint)(partition_remaining / BLOCK), (uint)CEIL_DIV(context_len - tile_start, BLOCK));
    }

    // Prefetch the next tile's block-table entry so its dependent global load overlaps this tile's
    // compute instead of stalling each iteration's dequant (matters most for long context / many tiles).
    uint physical_block = tile_count ? (uint)block_indices[block_table_begin + tile_start / BLOCK] : 0;

    for (uint tile = 0; tile < tile_count; tile++) {
        const uint key_start = tile_start + tile * BLOCK;
        const uint valid_tokens = min((uint)BLOCK, context_len - key_start);
        const __global INPUT1_TYPE* key_block = key_cache + physical_block * KV_HEADS_NUM * K_BLOCK_ELEMS + kv_head_idx * K_BLOCK_ELEMS;
        const __global INPUT2_TYPE* value_block = value_cache + physical_block * KV_HEADS_NUM * V_BLOCK_ELEMS + kv_head_idx * V_BLOCK_ELEMS;
        if (tile + 1 < tile_count)
            physical_block = (uint)block_indices[block_table_begin + (key_start + BLOCK) / BLOCK];
#ifdef SINGLE_KV_SLM
        __local half* k_cur = kv_slm;
        __local half* v_cur = kv_slm;
        // Previous tile's P.V reads must finish before K overwrites the buffer
        if (tile > 0)
            barrier(CLK_LOCAL_MEM_FENCE);
        FUNC_CALL(dequant_key)(key_block, k_cur, lid, local_size);
#else
        __local half* k_cur = k_slm + (tile & 1) * (BLOCK * HEAD_SIZE);
        __local half* v_cur = v_slm + (tile & 1) * (HEAD_SIZE * BLOCK);
        FUNC_CALL(dequant_key)(key_block, k_cur, lid, local_size);
        FUNC_CALL(dequant_value)(value_block, v_cur, lid, local_size, valid_tokens);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // S[8 rows, 16 key tokens] = Q . K^T. Interleave the SLM load with the MAD to overlap load latency.
        float8 s = (float8)(0.0f);
        for (uint c = 0; c < CHUNKS; c += 2) {
            const int8 k0 = vload8(sglid * CHUNKS + c, (const __local int*)k_cur);
            s = intel_sub_group_f16_f16_matrix_mad_k16(q_cache[c], k0, s);
            const int8 k1 = vload8(sglid * CHUNKS + c + 1, (const __local int*)k_cur);
            s = intel_sub_group_f16_f16_matrix_mad_k16(q_cache[c + 1], k1, s);
        }
        s *= (float8)(scale_val);

        // Lane owns key token key_pos; the causal length differs per row
        const uint key_pos = key_start + sglid;
        float scores[ROWS];
        vstore8(s, 0, scores);
        unroll_for (uint m = 0; m < ROWS; m++) {
            const uint token_off = sgid * TOKENS_PER_SG + m / KV_HEADS_GROUP_SIZE;
            const uint seq_len = past_len + token_off + 1;
            bool valid = token_off < query_count && key_pos < seq_len;
#if SLIDING_WINDOW_SIZE != 0
            valid = valid && !(seq_len > SLIDING_WINDOW_SIZE && key_pos < seq_len - SLIDING_WINDOW_SIZE);
#endif
#if HAS_QQ_BIAS
            // Speculative tree mask between draft tokens of the same subsequence
            if (valid && qq_bias_num > 0 && key_pos >= past_len) {
                const uint spec_offset = key_pos - past_len;
                if (spec_offset < qq_bias_spec_num && qq_bias[cumulated_qq_bias_num + token_off * qq_bias_spec_num + spec_offset] == 0)
                    valid = false;
            }
#endif
#if HAS_ALIBI
            scores[m] += convert_float(alibi_slopes[head_begin + m % KV_HEADS_GROUP_SIZE]) * ((int)key_pos + 1 - (int)seq_len);
#endif
            if (!valid)
                scores[m] = -INFINITY;
        }

        // Online softmax; reductions are uniform across the subgroup
        float tile_max[ROWS];
        unroll_for (uint m = 0; m < ROWS; m++) {
            tile_max[m] = sub_group_reduce_max(scores[m]);
        }
        const float8 new_max = fmax(row_max, vload8(0, tile_max));
        const bool rescale_needed = any(new_max != row_max);
        const float8 old_scale = select(native_exp(row_max - new_max), (float8)(0.0f), isinf(row_max));
        const float8 p = select(native_exp(vload8(0, scores) - new_max), (float8)(0.0f), isinf(new_max));

        float p_arr[ROWS];
        vstore8(p, 0, p_arr);
        float tile_sum[ROWS];
        unroll_for (uint m = 0; m < ROWS; m++) {
            tile_sum[m] = sub_group_reduce_add(p_arr[m]);
        }
        row_sum = row_sum * old_scale + vload8(0, tile_sum);
        row_max = new_max;

        if (rescale_needed) {
            unroll_for (uint c = 0; c < CHUNKS; c++) {
                acc[c] *= old_scale;
            }
        }

#ifdef SINGLE_KV_SLM
        barrier(CLK_LOCAL_MEM_FENCE);
        FUNC_CALL(dequant_value)(value_block, v_cur, lid, local_size, valid_tokens);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        // O[8 rows, head dims] += P . V; A operand lane == key token
        const short8 p_half = as_short8(convert_half8(p));
        for (uint c = 0; c < CHUNKS; c += 2) {
            const int8 v0 = vload8(c * SUBGROUP_SIZE + sglid, (const __local int*)v_cur);
            acc[c] = intel_sub_group_f16_f16_matrix_mad_k16(p_half, v0, acc[c]);
            const int8 v1 = vload8((c + 1) * SUBGROUP_SIZE + sglid, (const __local int*)v_cur);
            acc[c + 1] = intel_sub_group_f16_f16_matrix_mad_k16(p_half, v1, acc[c + 1]);
        }
        // Double buffer needs no trailing barrier: the next write to this buffer happens after the next tile's barrier,
        // which every subgroup reaches only after finishing this tile.
    }

    float max_arr[ROWS];
    float sum_arr[ROWS];
    vstore8(row_max, 0, max_arr);
    vstore8(row_sum, 0, sum_arr);

    unroll_for (uint m = 0; m < ROWS; m++) {
        const uint token_off = sgid * TOKENS_PER_SG + m / KV_HEADS_GROUP_SIZE;
        if (token_off >= query_count)
            continue;

        const uint seq_len = past_len + token_off + 1;
        const uint token = query_begin + token_off;
        const uint head = head_begin + m % KV_HEADS_GROUP_SIZE;
        const float denom = sum_arr[m];
        const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

        // The finalization stage skips tokens fitting into a single partition, so those are written to output directly
        if (seq_len > SEQ_LEN_PARTITION_SIZE) {
            if (sglid == 0) {
                const uint exp_sums_offset = token * HEADS_NUM * total_partitions_num + head * total_partitions_num + partition_idx;
                exp_sums[exp_sums_offset] = denom;
                // Finite for fully masked partitions so finalization's exp(max - global_max) never sees -inf
                max_logits[exp_sums_offset] = isinf(max_arr[m]) ? SOFTMAX_ACCUMULATOR_VAL_MIN : max_arr[m];
            }
            const uint tmp_out_offset = token * (HEADS_NUM * V_HEAD_SIZE * total_partitions_num) +
                                        head * (V_HEAD_SIZE * total_partitions_num) +
                                        partition_idx * V_HEAD_SIZE;
            unroll_for (uint c = 0; c < CHUNKS; c++) {
                tmp_out[tmp_out_offset + c * SUBGROUP_SIZE + sglid] = TO_OUTPUT_TYPE(acc[c][m] * inv_denom);
            }
        } else if (partition_idx == 0) {
            const uint output_offset = token * (HEADS_NUM * V_HEAD_SIZE) + head * V_HEAD_SIZE;
            unroll_for (uint c = 0; c < CHUNKS; c++) {
                output[output_offset + c * SUBGROUP_SIZE + sglid] = TO_OUTPUT_TYPE(acc[c][m] * inv_denom);
            }
        }
    }
}

#undef HEAD_SIZE
#undef BLOCK
#undef CHUNKS
#undef ROWS
#undef QUERY_ROW_STRIDE
#undef K_BLOCK_ELEMS
#undef V_BLOCK_ELEMS
