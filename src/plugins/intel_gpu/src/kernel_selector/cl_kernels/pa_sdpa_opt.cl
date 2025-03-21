// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define SUBGROUPS_PER_WG ((HEAD_SIZE / SUBGROUP_SIZE) * SG_SCALE_FACTOR)
#define PAGED_ATTENTION_BLOCKS_PER_PARTITION (SEQ_LEN_PARTITION_SIZE / PAGED_ATTENTION_BLOCK_SIZE)

#if HEAD_SIZE > 128 || HEADS_PER_WI > 1
    #define STORE_QUERY_TO_SLM 1
#endif

#ifdef SDPA_STAGE_0

#if SEQ_LEN_PARTITION_SIZE % PAGED_ATTENTION_BLOCK_SIZE != 0
    #error pa_sdpa_opt.cl
#endif

#if SUBGROUP_SIZE != PAGED_ATTENTION_BLOCK_SIZE
    #error pa_sdpa_opt.cl
#endif

#if HEADS_PER_WI > 1
    #define GET_MULTIPLE_HEAD_IDX_OR_FIXED_VAL(multiple_heads_idx, fixed_val) multiple_heads_idx
#else
    #define GET_MULTIPLE_HEAD_IDX_OR_FIXED_VAL(multiple_heads_idx, fixed_val) fixed_val
#endif

#define QUERIES_PER_WI HEADS_PER_WI
#define SOFTMAX_ACCUMULATOR_VEC_TYPE MAKE_VECTOR_TYPE(SOFTMAX_ACCUMULATOR_TYPE, QUERIES_PER_WI)
#define TO_SOFTMAX_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, SOFTMAX_ACCUMULATOR_VEC_TYPE)(x)

#define _VEC_1_ELEMENT_ACCESSOR(vec, idx) vec
#define _VEC_2_ELEMENT_ACCESSOR(vec, idx) vec[idx]
#define _VEC_3_ELEMENT_ACCESSOR(vec, idx) vec[idx]
#define _VEC_4_ELEMENT_ACCESSOR(vec, idx) vec[idx]
#define _VEC_8_ELEMENT_ACCESSOR(vec, idx) vec[idx]
#define _VEC_16_ELEMENT_ACCESSOR(vec, idx) vec[idx]
#define _VEC_ELEMENT_ACCESSOR(VECTOR_SIZE, vec, idx) CAT(_VEC_,CAT(VECTOR_SIZE,_ELEMENT_ACCESSOR))(vec, idx)

#define GET_VECTOR_ELEMENT(vec, idx) _VEC_ELEMENT_ACCESSOR(QUERIES_PER_WI, vec, idx)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE * SG_SCALE_FACTOR)))
KERNEL(pa_sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* block_indices,
    const __global INPUT5_TYPE* block_indices_begins,
#if MULTI_TOKENS_PROCESSING
    const __global INPUT6_TYPE* subsequence_begins,
#endif
#if HAS_SCALE_INPUT
    const __global SCALE_INPUT_TYPE* scale,
#endif
#if HAS_ALIBI
    const __global ALIBI_INPUT_TYPE* alibi_slopes,
#endif
    __global OUTPUT_TYPE* output,
#if PAGED_ATTENTION_SCORES_OUTPUT
    __global SOFTMAX_ACCUMULATOR_TYPE* softmax_results,
    const __global int* subsequence_offsets,
#endif
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
#if MULTI_TOKENS_PROCESSING
    , __global const int* gws_subseq_mapping
#endif
) {
    // Input shapes:
    // query: [sequences_num, HEADS_NUM * HEAD_SIZE]
    // key_cache: [num_blocks, HEADS_NUM, HEAD_SIZE, PAGED_ATTENTION_BLOCK_SIZE]
    // value_cache: [num_blocks, HEADS_NUM, PAGED_ATTENTION_BLOCK_SIZE, HEAD_SIZE]
    // past_lens: [sequences_num]
    // subsequence_begins: [sequences_num + 1]
    // block_indices: [used_blocks_num]
    // block_indices_begins: [sequences_num + 1]
    // rotated_block_indices: [num_rotated_blocks ]
    // rotation_deltas [num_rotated_blocks, 1 || PAGED_ATTENTION_BLOCK_SIZE ]
    // rotation_trig_lut [MAX_CONTEXT_LEN, HEAD_SIZE]
    //
    // Output shapes:
    // output: [sequences_num, HEADS_NUM * HEAD_SIZE]
    // exp_sums: [sequences_num, HEADS_NUM, total_partitions_num]
    // max_logits: [sequences_num, HEADS_NUM, total_partitions_num]
    // tmp_out: [sequences_num, HEADS_NUM, total_partitions_num, HEAD_SIZE]

    const uint seq_idx = get_global_id(0);
#if HEADS_PER_WI > 1
    const uint heads_group_idx = get_global_id(1);
    const uint head_num_idx =
        (heads_group_idx * HEADS_PER_WI) -
        ((heads_group_idx / ITERATIONS_PER_KV_HEADS_GROUP) * (HEADS_PER_WI - HEADS_LEFTOVERS_NUM) * (HEADS_LEFTOVERS_NUM > 0));
    const uint iter_heads_num =
        min(KV_HEADS_GROUP_SIZE - ((heads_group_idx % ITERATIONS_PER_KV_HEADS_GROUP) * HEADS_PER_WI), (uint)HEADS_PER_WI);
#else
    const uint head_num_idx = get_global_id(1);
#endif
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint total_partitions_num = get_num_groups(2);
    const uint lid = get_local_id(2);

#if SG_SCALE_FACTOR == 2
    const uint head_size_idx = lid % HEAD_SIZE;
#elif SG_SCALE_FACTOR == 1
    const uint head_size_idx = lid;
#else
    #error "pa_sdpa_opt.cl: Unsupported scale factor"
#endif

    const uint batch_idx = seq_idx;

#if MULTI_TOKENS_PROCESSING
    const int subsequence_idx = gws_subseq_mapping[seq_idx];
    const int subsequence_begin = subsequence_begins[subsequence_idx];
    const int subsequence_end = subsequence_begins[subsequence_idx + 1];
    const uint seq_len = past_lens[subsequence_idx] + 1 + (seq_idx - subsequence_begin);
#else
    const uint subsequence_idx = seq_idx;
    const uint seq_len = past_lens[seq_idx] + 1;
#endif

    const uint partition_idx = get_group_id(2);

    if (partition_idx * SEQ_LEN_PARTITION_SIZE >= seq_len) {
        return;
    }

    const uint total_blocks_num = CEIL_DIV(seq_len, PAGED_ATTENTION_BLOCK_SIZE);

#ifdef STORE_QUERY_TO_SLM
    // SLM buffer for query inputs
    __local INPUT0_TYPE slm_query[HEAD_SIZE * QUERIES_PER_WI];
#endif

    // SLM for intermediate QK results
    __local SOFTMAX_ACCUMULATOR_TYPE slm_qk_vals[SEQ_LEN_PARTITION_SIZE * QUERIES_PER_WI];

    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WGs
    __local SOFTMAX_ACCUMULATOR_TYPE slm_qk_max_vals[SUBGROUPS_PER_WG * QUERIES_PER_WI];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_vals[SUBGROUPS_PER_WG * QUERIES_PER_WI];

    SOFTMAX_ACCUMULATOR_VEC_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;

    {
#if STORE_QUERY_TO_SLM
        for (uint i = sgid * SUBGROUP_SIZE; i < HEADS_PER_WI * HEAD_SIZE; i += SUBGROUP_SIZE) {
            const uint query_idx_local = i % HEAD_SIZE + sglid;
            const uint head_idx = i / HEAD_SIZE;

#if HEADS_LEFTOVERS_NUM > 0
            if (head_idx >= iter_heads_num)
                break;
#endif

            const uint query_idx = INPUT0_OFFSET +
                                   seq_idx * (HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM) +
                                   (head_num_idx + head_idx) * HEAD_SIZE +
                                   query_idx_local;

            INPUT0_TYPE q_val = BLOCK_READN(INPUT0_TYPE, 1, query, query_idx);

            // Apply scale value directly to the query input to improve accuracy in case of a high range of input data
#ifdef SCALE_VAL
            q_val = TO_INPUT0_TYPE(SCALE_VAL) * q_val;
#else
            q_val = *scale * q_val;
#endif

            slm_query[head_idx * HEAD_SIZE + query_idx_local] = q_val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#else // !STORE_QUERY_TO_SLM
        INPUT0_TYPE q_val[HEAD_SIZE / SUBGROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUBGROUP_SIZE; i++) {
            const uint query_idx = INPUT0_OFFSET +
                                   seq_idx * (HEAD_SIZE * HEADS_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM) +
                                   head_num_idx * HEAD_SIZE +
                                   i * SUBGROUP_SIZE;
            q_val[i] = BLOCK_READN(INPUT0_TYPE, 1, query, query_idx);

            // Apply scale value directly to the query input to improve accuracy in case of a high range of input data
#ifdef SCALE_VAL
            q_val[i] = TO_INPUT0_TYPE(SCALE_VAL) * q_val[i];
#else
            q_val[i] = *scale * q_val[i];
#endif
        }
#endif // STORE_QUERY_TO_SLM

        const uint blocks_num_per_partition = min(total_blocks_num - partition_idx * PAGED_ATTENTION_BLOCKS_PER_PARTITION, (uint)PAGED_ATTENTION_BLOCKS_PER_PARTITION);

        uint blocks_num = blocks_num_per_partition / SUBGROUPS_PER_WG;
        if (sgid < blocks_num_per_partition % SUBGROUPS_PER_WG)
            blocks_num++;

        const uint start_block_idx = block_indices_begins[subsequence_idx] + partition_idx * PAGED_ATTENTION_BLOCKS_PER_PARTITION + sgid;
        for (uint block_num = 0; block_num < blocks_num; block_num++) {
            const uint head_idx = head_num_idx / KV_HEADS_GROUP_SIZE;
            const uint block_offset = block_indices[start_block_idx + block_num * SUBGROUPS_PER_WG] * ADJUSTED_HEAD_SIZE * KV_HEADS_NUM * SUBGROUP_SIZE + head_idx * ADJUSTED_HEAD_SIZE * SUBGROUP_SIZE;

            SOFTMAX_ACCUMULATOR_VEC_TYPE qk_acc = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            #define KEY_VEC_SIZE SUBGROUP_SIZE
            #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_VEC_SIZE)
#if IS_KV_COMPRESSED
            const uint key_comp_offset = block_offset + HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE* key_comp_ptr = key_cache + key_comp_offset;
            INPUT0_TYPE comp_scale = key_comp_ptr[0 + sglid];
            INPUT0_TYPE comp_zp = key_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid];
#endif

            unroll_for (uint qk_idx = 0; qk_idx < HEAD_SIZE / KEY_VEC_SIZE; qk_idx++) {
                #define KEY_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_VEC_SIZE)
                #define TO_KEY_BLOCK_UNCOMPRESSED_TYPE(val) CAT(convert_, KEY_BLOCK_UNCOMPRESSED)(val)

#if IS_KV_COMPRESSED
                KEY_BLOCK_UNCOMPRESSED k_vals;
                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                    k_vals[i] = BLOCK_READN(INPUT1_TYPE, 1, key_cache, block_offset + qk_idx * SUBGROUP_SIZE * KEY_VEC_SIZE + i * SUBGROUP_SIZE);
                    k_vals[i] = (k_vals[i] - comp_zp) * comp_scale;
                }
#else
                KEY_BLOCK k_vals = 0;
                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                    k_vals[i] = BLOCK_READN(INPUT1_TYPE, 1, key_cache, block_offset + qk_idx * SUBGROUP_SIZE * KEY_VEC_SIZE + i * SUBGROUP_SIZE);
                }
#endif

#if XE2_QK_MULTIPLICATION
#if STORE_QUERY_TO_SLM
                MAKE_VECTOR_TYPE(INPUT0_TYPE, QUERIES_PER_WI) q_val;
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    GET_VECTOR_ELEMENT(q_val, q_idx) = slm_query[q_idx * HEAD_SIZE + qk_idx * KEY_VEC_SIZE + sglid];
                }
#endif

                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
#if STORE_QUERY_TO_SLM
                    qk_acc = mad(TO_SOFTMAX_ACCUMULATOR_VEC_TYPE(sub_group_broadcast(q_val, i)), TO_SOFTMAX_ACCUMULATOR_TYPE(k_vals[i]), qk_acc);
#else
                    qk_acc = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(q_val[qk_idx], i)), TO_SOFTMAX_ACCUMULATOR_TYPE(k_vals[i]), qk_acc);
#endif
                }
#else // !XE2_QK_MULTIPLICATION
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
#if STORE_QUERY_TO_SLM
                    SOFTMAX_ACCUMULATOR_TYPE q_val = slm_query[q_idx * HEAD_SIZE + qk_idx * KEY_VEC_SIZE + sglid];
#endif
                    unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
#if STORE_QUERY_TO_SLM
                        GET_VECTOR_ELEMENT(qk_acc, q_idx) = mad(sub_group_broadcast(q_val, i), TO_SOFTMAX_ACCUMULATOR_TYPE(k_vals[i]), GET_VECTOR_ELEMENT(qk_acc, q_idx));
#else
                        qk_acc = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(q_val[qk_idx], i)), TO_SOFTMAX_ACCUMULATOR_TYPE(k_vals[i]), qk_acc);
#endif
                    }
                }
#endif // XE2_QK_MULTIPLICATION
            }

            const uint token_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + block_num * SUBGROUPS_PER_WG * SUBGROUP_SIZE + sgid * SUBGROUP_SIZE + sglid;

#ifdef HAS_ALIBI
            const int alibi_val = (1 - seq_len) + token_idx;
            unroll_for (uint q_idx = 0; q_idx < HEADS_PER_WI; q_idx++) {
                GET_VECTOR_ELEMENT(qk_acc, q_idx) += alibi_slopes[head_num_idx + q_idx] * alibi_val;
            }
#endif

#if SLIDING_WINDOW_SIZE != 0
            if (token_idx >= seq_len || (seq_len > SLIDING_WINDOW_SIZE && token_idx < (seq_len - SLIDING_WINDOW_SIZE)))
#else
            if (token_idx >= seq_len)
#endif
                qk_acc = SOFTMAX_ACCUMULATOR_VAL_MIN;

            qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_VEC_TYPE(qk_acc));

            unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                slm_qk_vals[q_idx * SEQ_LEN_PARTITION_SIZE + block_num * SUBGROUPS_PER_WG * SUBGROUP_SIZE + sgid * SUBGROUP_SIZE + sglid] = GET_VECTOR_ELEMENT(qk_acc, q_idx);
            }
        }

        unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
            GET_VECTOR_ELEMENT(qk_max, q_idx) = sub_group_reduce_max(GET_VECTOR_ELEMENT(qk_max, q_idx));
        }
    }

    {
        // SoftMax calculation
        if (sglid < QUERIES_PER_WI) {
            const uint head_idx = GET_MULTIPLE_HEAD_IDX_OR_FIXED_VAL(sglid, 0);
            slm_qk_max_vals[head_idx * SUBGROUPS_PER_WG + sgid] = GET_VECTOR_ELEMENT(qk_max, head_idx);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG) {
            unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                GET_VECTOR_ELEMENT(qk_max, q_idx) = slm_qk_max_vals[q_idx * SUBGROUPS_PER_WG + sglid];
            }
        }

        // Final max value after reduction across of all SG and WI
        unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
            GET_VECTOR_ELEMENT(qk_max, q_idx) = sub_group_reduce_max(GET_VECTOR_ELEMENT(qk_max, q_idx));
        }

        SOFTMAX_ACCUMULATOR_VEC_TYPE exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        const uint qk_iters_num = CEIL_DIV(SEQ_LEN_PARTITION_SIZE, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
        for (uint qk_idx = 0; qk_idx < qk_iters_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            // TODO: const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx
            const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;

#if SEQ_LEN_PARTITION_SIZE % SUBGROUPS_PER_WG * SUBGROUP_SIZE == 0
            if (global_data_idx < seq_len) {
#else
            if (global_data_idx < seq_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
#endif
                unroll_for(uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    const uint slm_idx = q_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx;
                    SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(slm_qk_vals[slm_idx]) - GET_VECTOR_ELEMENT(qk_max, q_idx));
                    slm_qk_vals[slm_idx] = qk_new;
                    GET_VECTOR_ELEMENT(exp_sum, q_idx) += qk_new;
                }
            }
        }

        unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
            GET_VECTOR_ELEMENT(exp_sum, q_idx) = sub_group_reduce_add(GET_VECTOR_ELEMENT(exp_sum, q_idx));
        }

        if (sglid < QUERIES_PER_WI) {
            const uint head_idx = GET_MULTIPLE_HEAD_IDX_OR_FIXED_VAL(sglid, 0);
            slm_exp_sum_vals[head_idx * SUBGROUPS_PER_WG + sgid] = GET_VECTOR_ELEMENT(exp_sum, head_idx);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG) {
            unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
               GET_VECTOR_ELEMENT(exp_sum, q_idx) = slm_exp_sum_vals[q_idx * SUBGROUPS_PER_WG + sglid];
            }
        }

        // Final sum of all exp_sum values
        unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
            GET_VECTOR_ELEMENT(exp_sum, q_idx) = sub_group_reduce_add(GET_VECTOR_ELEMENT(exp_sum, q_idx));
        }

        for (uint qk_idx = 0; qk_idx < qk_iters_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;

#if SEQ_LEN_PARTITION_SIZE % SUBGROUPS_PER_WG * SUBGROUP_SIZE == 0
            if (global_data_idx < seq_len) {
#else
            if (global_data_idx < seq_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
#endif
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    const uint slm_idx = q_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx;
                    SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(slm_qk_vals[slm_idx]) / GET_VECTOR_ELEMENT(exp_sum, q_idx);
                    slm_qk_vals[slm_idx] = qk_new;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        {
            // Save temporary exm_sums and max_logits values for each partition_num
#if HEADS_PER_WI > 1 && HEADS_LEFTOVERS_NUM > 0
            if (seq_len > SEQ_LEN_PARTITION_SIZE && sgid < HEADS_PER_WI && sgid < iter_heads_num) {
#else
            if (seq_len > SEQ_LEN_PARTITION_SIZE && sgid < HEADS_PER_WI) {
#endif
                const uint head_idx = GET_MULTIPLE_HEAD_IDX_OR_FIXED_VAL(sgid, 0);
                const uint exp_sums_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                             (head_num_idx + head_idx) * total_partitions_num +
                                             partition_idx;
                exp_sums[exp_sums_offset] = GET_VECTOR_ELEMENT(exp_sum, head_idx);

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = GET_VECTOR_ELEMENT(qk_max, head_idx);
            }

#if PAGED_ATTENTION_SCORES_OUTPUT
#if MULTI_TOKENS_PROCESSING
            const uint subsequence_idx = gws_subseq_mapping[seq_idx];
            const uint subsequence_start_pos = subsequence_begins[subsequence_idx];
            const uint subsequence_end_pos = subsequence_begins[subsequence_idx + 1];
            const bool save_softmax_results = seq_idx == subsequence_end_pos - 1;
#else
            const uint subsequence_idx = seq_idx;
            const bool save_softmax_results = true;
#endif // MULTI_TOKENS_PROCESSING
            // PagedAttention is supposed to save only last "row" of the QK matrix multiplication,
            // so save SEQ_LEN_PARTITION_SIZE elements for each partition
            if (save_softmax_results) {
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
#if HEADS_LEFTOVERS_NUM > 0
                    if (q_idx >= iter_heads_num)
                        break;
#endif
                    const uint output_offset = subsequence_idx * HEADS_NUM * total_partitions_num * SEQ_LEN_PARTITION_SIZE +
                                               (head_num_idx + q_idx) * total_partitions_num * SEQ_LEN_PARTITION_SIZE +
                                               partition_idx * SEQ_LEN_PARTITION_SIZE;
                    for (uint i = sgid * SUBGROUP_SIZE + sglid; i < SEQ_LEN_PARTITION_SIZE; i += SUBGROUPS_PER_WG * SUBGROUP_SIZE) {
                        softmax_results[output_offset + i] = slm_qk_vals[i];
                    }
                }
            }
#endif // PAGED_ATTENTION_SCORES_OUTPUT
        }
    }

    {
        // QK*V calculation
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, QUERIES_PER_WI) acc = OUTPUT_VAL_ZERO;

        const uint partition_seq_len = min(seq_len - partition_idx * SEQ_LEN_PARTITION_SIZE, (uint)SEQ_LEN_PARTITION_SIZE);

#if SG_SCALE_FACTOR > 1
        const uint block_start_idx = (sgid / (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) * (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR / SUBGROUP_SIZE);
        const uint block_end_idx = min(block_start_idx + (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR / SUBGROUP_SIZE), partition_seq_len / SUBGROUP_SIZE);
#else
        const uint block_start_idx = 0;
        const uint block_end_idx = partition_seq_len / SUBGROUP_SIZE;
#endif

        uint blocks_num_per_partition = min(total_blocks_num - partition_idx * PAGED_ATTENTION_BLOCKS_PER_PARTITION, (uint)PAGED_ATTENTION_BLOCKS_PER_PARTITION);

        uint leftovers = blocks_num_per_partition * PAGED_ATTENTION_BLOCK_SIZE - partition_seq_len;
        if (leftovers != 0) {
            leftovers = PAGED_ATTENTION_BLOCK_SIZE - leftovers;
            blocks_num_per_partition = blocks_num_per_partition - 1;
        }

        const uint start_block_idx = block_indices_begins[subsequence_idx] + partition_idx * PAGED_ATTENTION_BLOCKS_PER_PARTITION;

        for (uint block_num = block_start_idx; block_num < block_end_idx; block_num++) {
            const uint head_idx = head_num_idx / KV_HEADS_GROUP_SIZE;
            const uint block_offset = block_indices[start_block_idx + block_num] * KV_HEADS_NUM * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            const uint value_offset = block_offset + head_size_idx;

#if IS_KV_COMPRESSED
            const uint value_comp_offset = block_offset + HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE* value_comp_ptr = value_cache + value_comp_offset;
            INPUT0_TYPE comp_scale = value_comp_ptr[0 + sglid];
            INPUT0_TYPE comp_zp = value_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid];
#endif

            #define VALUE_VEC_SIZE SUBGROUP_SIZE
            #define VALUE_UNCOMPRESSED INPUT0_TYPE
            #define VALUE_BLOCK MAKE_VECTOR_TYPE(INPUT2_TYPE, VALUE_VEC_SIZE)
            #define VALUE_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(INPUT0_TYPE, VALUE_VEC_SIZE)
            #define TO_VALUE_UNCOMPRESSED_TYPE(val) CAT(convert_, VALUE_UNCOMPRESSED)(val)

            VALUE_BLOCK v_vals_packed;
            unroll_for (uint i = 0; i < VALUE_VEC_SIZE; i++) {
                v_vals_packed[i] = BLOCK_READN(INPUT2_TYPE, 1, value_cache, value_offset + i * HEAD_SIZE);
            }

#if IS_KV_COMPRESSED
            VALUE_BLOCK_UNCOMPRESSED value_vals;
            unroll_for (uint i = 0; i < VALUE_VEC_SIZE; i++) {
                value_vals[i] = (TO_VALUE_UNCOMPRESSED_TYPE(v_vals_packed[i]) - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
            }
#else
            VALUE_BLOCK value_vals = v_vals_packed;
#endif

            unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                OUTPUT_TYPE qk_val = slm_qk_vals[q_idx * SEQ_LEN_PARTITION_SIZE + block_num * PAGED_ATTENTION_BLOCK_SIZE + sglid];

                unroll_for (uint i = 0; i < VALUE_VEC_SIZE; i++) {
                    GET_VECTOR_ELEMENT(acc, q_idx) = mad(sub_group_broadcast(qk_val, i), value_vals[i], GET_VECTOR_ELEMENT(acc, q_idx));
                }
            }
        }


#if SG_SCALE_FACTOR > 1
        if (sgid >= SUBGROUPS_PER_WG / SG_SCALE_FACTOR) {
#endif
        if (leftovers != 0) {
            const uint head_idx = head_num_idx / KV_HEADS_GROUP_SIZE;
            const uint last_block_idx = start_block_idx + blocks_num_per_partition;
            const uint block_offset = block_indices[last_block_idx] * KV_HEADS_NUM * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            const uint value_offset = block_offset + head_size_idx;

#if IS_KV_COMPRESSED
            const uint value_comp_offset = block_offset + HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE* value_comp_ptr = value_cache + value_comp_offset;
            INPUT0_TYPE comp_scale = value_comp_ptr[0 + sglid];
            INPUT0_TYPE comp_zp = value_comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + sglid];
#endif

            MAKE_VECTOR_TYPE(OUTPUT_TYPE, QUERIES_PER_WI) qk_val;
            unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                GET_VECTOR_ELEMENT(qk_val, q_idx) = slm_qk_vals[q_idx * SEQ_LEN_PARTITION_SIZE + blocks_num_per_partition * PAGED_ATTENTION_BLOCK_SIZE + sglid];
            }
            for (uint i = 0; i < leftovers; i++) {
                INPUT2_TYPE value_packed = BLOCK_READN(INPUT2_TYPE, 1, value_cache, value_offset + i * HEAD_SIZE);
#if IS_KV_COMPRESSED
                #define VALUE_UNCOMPRESSED INPUT0_TYPE
                #define TO_VALUE_UNCOMPRESSED_TYPE(val) CAT(convert_, VALUE_UNCOMPRESSED)(val)
                VALUE_UNCOMPRESSED value_val = (TO_VALUE_UNCOMPRESSED_TYPE(value_packed) - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
#else
                VALUE_UNCOMPRESSED value_val = value_packed;
#endif

                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    GET_VECTOR_ELEMENT(acc, q_idx) = mad(sub_group_broadcast(GET_VECTOR_ELEMENT(qk_val, q_idx), i), value_val, GET_VECTOR_ELEMENT(acc, q_idx));
                }
            }
        }


#if SG_SCALE_FACTOR > 1
        }
#endif

#if SG_SCALE_FACTOR > 1
        // Reuse slm_qk_vals SLM to sum-up results between two sets of subgroups
        __local OUTPUT_TYPE* tmp_reduction_slm_mem = (__local OUTPUT_TYPE*)&slm_qk_vals[0];
        if ((partition_seq_len > (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR)) || (leftovers != 0)) {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (sgid >= SUBGROUPS_PER_WG / SG_SCALE_FACTOR) {
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    tmp_reduction_slm_mem[q_idx * HEAD_SIZE + head_size_idx] = GET_VECTOR_ELEMENT(acc, q_idx);
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (sgid < SUBGROUPS_PER_WG / SG_SCALE_FACTOR) {
                unroll_for (uint q_idx = 0; q_idx < QUERIES_PER_WI; q_idx++) {
                    GET_VECTOR_ELEMENT(acc, q_idx) += tmp_reduction_slm_mem[q_idx * HEAD_SIZE + head_size_idx];
                }
            }
        }
#endif

#if SG_SCALE_FACTOR > 1
        if (sgid < SUBGROUPS_PER_WG / SG_SCALE_FACTOR) {
#endif

        if (seq_len > SEQ_LEN_PARTITION_SIZE) {
            unroll_for (uint q_idx = 0; q_idx < HEADS_PER_WI; q_idx++) {
#if HEADS_LEFTOVERS_NUM > 0
                if (q_idx >= iter_heads_num)
                    break;
#endif
                const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * total_partitions_num) +
                                            (head_num_idx + q_idx) * (HEAD_SIZE * total_partitions_num) +
                                            partition_idx * HEAD_SIZE +
                                            sgid * SUBGROUP_SIZE +
                                            sglid;

                tmp_out[tmp_out_offset] = GET_VECTOR_ELEMENT(acc, q_idx);
            }
        } else {
            unroll_for (uint q_idx = 0; q_idx < HEADS_PER_WI; q_idx++) {
#if HEADS_LEFTOVERS_NUM > 0
                if (q_idx >= iter_heads_num)
                    break;
#endif
                const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                           (head_num_idx + q_idx) * HEAD_SIZE +
                                           sgid * SUBGROUP_SIZE +
                                           sglid;

                output[output_offset] = GET_VECTOR_ELEMENT(acc, q_idx);
            }
        }

#if SG_SCALE_FACTOR > 1
        }
#endif
    }
}

#endif

#ifdef SDPA_STAGE_1

#if SOFTMAX_ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#elif SOFTMAX_ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#else
#error Unexpected SOFTMAX_ACCUMULATOR data type size
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(pa_sdpa_finalization_stage)(
    const __global INPUT3_TYPE* past_lens,
#if MULTI_TOKENS_PROCESSING
    const __global INPUT6_TYPE* subsequence_begins,
#endif
    __global OUTPUT_TYPE* output,
#if PAGED_ATTENTION_SCORES_OUTPUT
    __global SOFTMAX_ACCUMULATOR_TYPE* softmax_results,
    const __global int* subsequence_offsets,
#endif
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
#if MULTI_TOKENS_PROCESSING
    const __global int* gws_subseq_mapping,
#endif
    const uint total_partitions_num) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_size_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();

#if MULTI_TOKENS_PROCESSING
    const int subsequence_idx = gws_subseq_mapping[seq_idx];
    const int subsequence_begin = subsequence_begins[subsequence_idx];
    const uint seq_len = past_lens[subsequence_idx] + 1 + (seq_idx - subsequence_begin);
#else
    const uint seq_len = past_lens[seq_idx] + 1;
#endif

    const uint num_of_partitions = CEIL_DIV(seq_len, SEQ_LEN_PARTITION_SIZE);

    if (seq_len <= SEQ_LEN_PARTITION_SIZE) {
        /* Short path, no need any actions for currently processing sequence */
        return;
    } else if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24) = 98304 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                         head_num_idx * total_partitions_num + partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_partitions) {
                exp_sum[i] = exp_sums[exp_sums_offset];
                max_logit[i] = max_logits[max_logit_offset];
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logit[i]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Update exp_sum with respect to the global maximum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            if (partition_idx < num_of_partitions) {
                exp_sum[i] = exp_sum[i] * native_exp(max_logit[i] - global_max);
                local_exp_sum += exp_sum[i];
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        for (uint partition_num = 0; partition_num < num_of_partitions; partition_num++) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * total_partitions_num * HEAD_SIZE) +
                                        head_num_idx * (total_partitions_num * HEAD_SIZE) +
                                        partition_num * HEAD_SIZE +
                                        head_size_idx;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_num / SUBGROUP_SIZE], partition_num % SUBGROUP_SIZE)) / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    } else {
        /* Global memory kernel version, can handle any number of tokens */
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint max_logit_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                          head_num_idx * total_partitions_num + partition_idx;

            if (partition_idx < num_of_partitions) {
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Calculate global sum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                         head_num_idx * total_partitions_num + partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_partitions) {
                local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
        for (uint partition_num = 0; partition_num < num_of_partitions; partition_num++) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * total_partitions_num * HEAD_SIZE) +
                                        head_num_idx * (total_partitions_num * HEAD_SIZE) +
                                        partition_num * HEAD_SIZE +
                                        head_size_idx;

            const uint exp_sums_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                         head_num_idx * total_partitions_num + partition_num;
            const uint max_logit_offset = exp_sums_offset;

            SOFTMAX_ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    }
}

#endif

#ifdef SDPA_STAGE_2
#define MAX_PARTITIONS_NUM 128

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(pa_sdpa_scores_calculation)(
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT6_TYPE* subsequence_begins,
    __global OUTPUT1_TYPE* scores_output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* softmax_output,
    const __global int* subsequence_offsets,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint is_mixed_mode) {
    const uint subsequence_idx = get_global_id(2);
    const uint partition_global_idx = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint partition_idx = get_group_id(0);
    const uint partition_size = get_local_size(0);
    const uint max_seq_len = get_global_size(0);
    const uint partitions_num = get_num_groups(0);
    const uint sgid = get_sub_group_id();
    const uint sgid_num = get_num_sub_groups();
    const uint sglid = get_sub_group_local_id();

    const int subsequence_begin = subsequence_begins[subsequence_idx];
    const int subsequence_end = subsequence_begins[subsequence_idx + 1];
    const uint seq_len = (subsequence_end - subsequence_begin) + past_lens[subsequence_idx];

    const uint num_of_partitions = CEIL_DIV(seq_len, partition_size);

    if (partition_idx >= num_of_partitions)
        return;

    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sums[HEADS_NUM];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_global_exp_sum[HEADS_NUM];

    SOFTMAX_ACCUMULATOR_TYPE total_score = SOFTMAX_ACCUMULATOR_VAL_ZERO;
    if (seq_len <= partition_size) {
        // If seq_len is less than the partition size, just reduce the results over the heads
        for (uint head_idx = 0; head_idx < HEADS_NUM; head_idx++) {
            const uint input_offset = subsequence_idx * HEADS_NUM * max_seq_len + head_idx * max_seq_len + partition_global_idx;
            SOFTMAX_ACCUMULATOR_TYPE softmax_value = softmax_output[input_offset];
            total_score += softmax_value;
        }
    } else if (seq_len <= partition_size * MAX_PARTITIONS_NUM) {
        // Optimized version for longer prompts (up to partition_size * MAX_PARTITIONS_NUM, ~64K tokens)

        // Depending on the previous kernel exp_sums and max_logits might have different structure:
        // For ordinary 1st and 2nd token kernels, there is only a single entry per subsequence.
        // However, for mixed mode execution, exp_sums and max_logits include information for all
        // tokens of each subsequence, but only the last one is needed for score calculation.
        const uint subsequence_pos = is_mixed_mode ? subsequence_end - 1 : subsequence_idx;

        for (uint head_idx = sgid; head_idx < HEADS_NUM; head_idx += sgid_num) {
            SOFTMAX_ACCUMULATOR_TYPE max_logit[MAX_PARTITIONS_NUM / SUBGROUP_SIZE];
            SOFTMAX_ACCUMULATOR_TYPE exp_sum[MAX_PARTITIONS_NUM / SUBGROUP_SIZE];

            const uint exp_sums_offset = subsequence_pos * HEADS_NUM * partitions_num + head_idx * partitions_num;
            for (int i = 0; i < partitions_num / SUBGROUP_SIZE; i++) {
                max_logit[i] = max_logits[exp_sums_offset + i * SUBGROUP_SIZE + sglid];
                exp_sum[i] = exp_sums[exp_sums_offset + i * SUBGROUP_SIZE + sglid];
            }

            const uint partitions_leftovers = partitions_num % SUBGROUP_SIZE;
            if (partitions_leftovers != 0) {
                const uint idx = partitions_num / SUBGROUP_SIZE;
                max_logit[idx] = sglid >= partitions_leftovers ? SOFTMAX_ACCUMULATOR_VAL_MIN : max_logits[exp_sums_offset + idx * SUBGROUP_SIZE + sglid];
                exp_sum[idx] = sglid >= partitions_leftovers ? SOFTMAX_ACCUMULATOR_VAL_ZERO : exp_sums[exp_sums_offset + idx * SUBGROUP_SIZE + sglid];
            }

            SOFTMAX_ACCUMULATOR_TYPE global_max_logit = max_logit[0];
            for (uint i = 1; i < CEIL_DIV(partitions_num, SUBGROUP_SIZE); i++) {
                global_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(global_max_logit, max_logit[i]);
            }

            global_max_logit = sub_group_reduce_max(global_max_logit);

            SOFTMAX_ACCUMULATOR_TYPE global_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            for (uint i = 0; i < CEIL_DIV(partitions_num, SUBGROUP_SIZE); i++) {
                SOFTMAX_ACCUMULATOR_TYPE adjusted_exp_sum = exp_sum[i] * native_exp(max_logit[i] - global_max_logit);
                // slm_exp_sums[head_idx][i * SUBGROUP_SIZE + sglid] = adjusted_exp_sum;
                if (i * SUBGROUP_SIZE + sglid == partition_idx)
                    slm_exp_sums[head_idx] = adjusted_exp_sum;
                global_exp_sum += adjusted_exp_sum;
            }

            global_exp_sum = sub_group_reduce_add(global_exp_sum);

            slm_global_exp_sum[head_idx] = global_exp_sum;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint head_idx = 0; head_idx < HEADS_NUM; head_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE adjusted_exp_sum = slm_exp_sums[head_idx];
            SOFTMAX_ACCUMULATOR_TYPE global_exp_sum = slm_global_exp_sum[head_idx];

            const uint input_offset = subsequence_idx * HEADS_NUM * max_seq_len + head_idx * max_seq_len + partition_global_idx;
            SOFTMAX_ACCUMULATOR_TYPE softmax_value = softmax_output[input_offset];

            softmax_value = softmax_value * adjusted_exp_sum / global_exp_sum;
            total_score += softmax_value;
        }
    } else {
        // Non optimized fallback version
        const uint subsequence_pos = is_mixed_mode ? subsequence_end - 1 : subsequence_idx;
        for (uint head_idx = 0; head_idx < HEADS_NUM; head_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE global_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;
            const uint max_logits_base_offset = subsequence_pos * HEADS_NUM * partitions_num + head_idx * partitions_num;
            for (uint i = 0; i < CEIL_DIV(partitions_num, SUBGROUP_SIZE); i++) {
                const uint partition_offset = i * SUBGROUP_SIZE + sglid;
                SOFTMAX_ACCUMULATOR_TYPE max_logit = partition_offset >= partitions_num ? SOFTMAX_ACCUMULATOR_VAL_MIN : max_logits[max_logits_base_offset + partition_offset];
                global_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(global_max_logit, max_logit);
            }

            global_max_logit = sub_group_reduce_max(global_max_logit);

            SOFTMAX_ACCUMULATOR_TYPE global_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            SOFTMAX_ACCUMULATOR_TYPE partition_adjusted_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            const uint exp_sums_base_offset = subsequence_pos * HEADS_NUM * partitions_num + head_idx * partitions_num;
            for (uint i = 0; i < CEIL_DIV(partitions_num, SUBGROUP_SIZE); i++) {
                const uint partition_offset = i * SUBGROUP_SIZE + sglid;
                SOFTMAX_ACCUMULATOR_TYPE exp_sum = partition_offset >= partitions_num ? SOFTMAX_ACCUMULATOR_VAL_ZERO : exp_sums[exp_sums_base_offset + partition_offset];
                SOFTMAX_ACCUMULATOR_TYPE max_logit = partition_offset >= partitions_num ? SOFTMAX_ACCUMULATOR_VAL_MIN : max_logits[max_logits_base_offset + partition_offset];
                SOFTMAX_ACCUMULATOR_TYPE adjusted_exp_sum = exp_sum * native_exp(max_logit - global_max_logit);
                global_exp_sum += adjusted_exp_sum;

                // Save and broadcast the adjusted exp_sum for the currently being processed partition
                if (i == partition_idx / SUBGROUP_SIZE)
                    partition_adjusted_exp_sum = sub_group_broadcast(adjusted_exp_sum, partition_idx % SUBGROUP_SIZE);
            }

            global_exp_sum = sub_group_reduce_add(global_exp_sum);

            const uint input_offset = subsequence_idx * HEADS_NUM * max_seq_len + head_idx * max_seq_len + partition_global_idx;
            SOFTMAX_ACCUMULATOR_TYPE softmax_value = softmax_output[input_offset];

            softmax_value = softmax_value * partition_adjusted_exp_sum / global_exp_sum;
            total_score += softmax_value;
        }
    }

    const uint output_offset = subsequence_offsets[subsequence_idx];
    if (partition_global_idx < seq_len) {
        scores_output[output_offset + partition_global_idx] = total_score;
    }
}

#undef MAX_PARTITIONS_NUM
#endif
