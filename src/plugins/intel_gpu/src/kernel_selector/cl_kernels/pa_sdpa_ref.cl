// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define SUB_GROUP_SIZE 16
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUB_GROUP_SIZE)

// The size of portion of HEAD_SIZE each WI process
#define Q_LOAD_ITERS (HEAD_SIZE / SUB_GROUP_SIZE)

// How much QK outputs each subgroup calculates per block
#define QK_VALS_PER_SG_PER_ITER (BLOCK_SIZE / SUBGROUPS_PER_WG)

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * KV_HEADS_NUM * BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

#ifdef SDPA_STAGE_0

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE)))
KERNEL(pa_sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* max_context_len,
    const __global INPUT4_TYPE* context_lens,
    const __global INPUT5_TYPE* block_tables,
    const __global INPUT6_TYPE* scale,
#ifdef USE_SEQ_LEN_SPLIT
    __global OUTPUT_TYPE* output,
    __global ACCUMULATOR_TYPE* exp_sums,
    __global ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
#else
    __global OUTPUT_TYPE* output
#endif
) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint num_of_portions = get_num_groups(2);

    const uint batch_idx = seq_idx / INPUT0_FEATURE_NUM;
    const uint token_idx = seq_idx % INPUT0_FEATURE_NUM;

    const uint context_len = context_lens[batch_idx];

    const uint blocks_pitch = INPUT5_FEATURE_NUM;

#ifdef USE_SEQ_LEN_SPLIT
    const uint portion_id = get_group_id(2);
    const uint block_start_idx = portion_id * SEQ_LEN_PORTION_SIZE / BLOCK_SIZE;

    if (portion_id * SEQ_LEN_PORTION_SIZE >= context_len) {
        return;
    }
#else
    const uint block_start_idx = 0;
#endif

    const uint total_blocks_num = CEIL_DIV(context_len, BLOCK_SIZE);

    __local OUTPUT_TYPE qk_vals_local[SHARED_MEM_SIZE];
    ACCUMULATOR_TYPE qk_max = ACCUMULATOR_VAL_MIN;

    {
        INPUT0_TYPE q_val[HEAD_SIZE / SUB_GROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUB_GROUP_SIZE; i++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                   head_num_idx * HEAD_SIZE +
                                   i * SUB_GROUP_SIZE;
            q_val[i] = QUERY_BLOCK_READ(query, query_idx);
        }

#ifdef USE_SEQ_LEN_SPLIT
        const uint blocks_num = ((portion_id + 1) * SEQ_LEN_PORTION_SIZE > context_len) ? (total_blocks_num - (portion_id * SEQ_LEN_PORTION_SIZE / BLOCK_SIZE))
                                                                                        : (SEQ_LEN_PORTION_SIZE / BLOCK_SIZE);
#else
        const uint blocks_num = total_blocks_num;
#endif
        uint token_idx_debug = 0;
        for (uint block_num = 0; block_num < blocks_num; block_num++) {
            const uint block_idx = batch_idx * blocks_pitch + block_start_idx + block_num;
            const uint block_offset = block_tables[block_idx] * KV_CACHE_BLOCK_STRIDE;

            OUTPUT_TYPE qk[QK_VALS_PER_SG_PER_ITER] = {0};

            for (uint q_idx = 0; q_idx < Q_LOAD_ITERS; q_idx++) {
                for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                    uint current_token = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                    if (current_token >= context_len)
                        continue;

                    const uint key_idx = block_offset +
                                        (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) +
                                        (X_BLOCK_SIZE * QK_VALS_PER_SG_PER_ITER) * sgid +
                                        (SUB_GROUP_SIZE / X_BLOCK_SIZE * BLOCK_SIZE * X_BLOCK_SIZE) * q_idx +
                                        (sglid / X_BLOCK_SIZE) * X_BLOCK_SIZE * BLOCK_SIZE +
                                        (sglid % X_BLOCK_SIZE) + qk_idx * X_BLOCK_SIZE;

#if X_BLOCK_SIZE == 16
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
                    INPUT1_TYPE k_val = KEY_BLOCK_READ(key_cache, key_idx);
#else
                    INPUT1_TYPE k_val = key_cache[key_idx];
#endif

                    qk[qk_idx] = mad(q_val[q_idx], k_val, qk[qk_idx]);
                }
            }

            // Summurize qk calculation across all WIs and apply scale
            for (uint qk_idx = 0; qk_idx < QK_VALS_PER_SG_PER_ITER; qk_idx++) {
                const uint current_token = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + qk_idx;
                if (current_token < context_len) {
                    qk[qk_idx] = sub_group_reduce_add(qk[qk_idx]);

                    // Apply scale
                    qk[qk_idx] = scale[0] * qk[qk_idx];

                    // Apply attention mask for context processing stage
                    const bool is_prefill_stage = INPUT0_FEATURE_NUM > 1;
                    if (is_prefill_stage && current_token > token_idx) {
                        qk[qk_idx] = qk[qk_idx] + OUTPUT_VAL_MIN;
                    }

                    qk_max = ACCUMULATOR_MAX_FUNC(qk_max, TO_ACCUMULATOR_TYPE(qk[qk_idx]));
                }
            }

            // Save QK results to local memory
            if (sglid < QK_VALS_PER_SG_PER_ITER) {
                const uint current_token_global_idx = (block_start_idx + block_num) * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#ifdef USE_SEQ_LEN_SPLIT
                const uint current_token_local = block_num * BLOCK_SIZE + sgid * QK_VALS_PER_SG_PER_ITER + sglid;
#else
                const uint current_token_local = current_token_global_idx;
#endif
                qk_vals_local[current_token_local] = current_token_global_idx >= context_len ? 0 : qk[sglid];
            }
        }
    }

    // Apply SoftMax operation
    __local ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];
    {
        if (sglid == 0)
            qk_max_vals[sgid] = qk_max;

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final max value after reduction across of all SG and WI
        qk_max = sub_group_reduce_max(qk_max);

        ACCUMULATOR_TYPE exp_sum = ACCUMULATOR_VAL_ZERO;
#ifdef USE_SEQ_LEN_SPLIT
        const uint qk_num = (num_of_portions == 1) ? CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE)
                                                   : CEIL_DIV(SEQ_LEN_PORTION_SIZE, SUBGROUPS_PER_WG * SUB_GROUP_SIZE);
#else
        const uint qk_num = CEIL_DIV(context_len, SUBGROUPS_PER_WG * SUB_GROUP_SIZE);
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
#ifdef USE_SEQ_LEN_SPLIT
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PORTION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                ACCUMULATOR_TYPE val = native_exp(TO_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) - qk_max);
                exp_sum += val;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(val);
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all exp_sum values
        exp_sum = sub_group_reduce_add(exp_sum);

        const ACCUMULATOR_TYPE inv_sum = ACCUMULATOR_VAL_ONE / exp_sum;

        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
            const uint global_data_idx = block_start_idx * BLOCK_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUB_GROUP_SIZE) + sgid * SUB_GROUP_SIZE + sglid;
#ifdef USE_SEQ_LEN_SPLIT
            if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PORTION_SIZE) {
#else
            if (global_data_idx < context_len) {
#endif
                ACCUMULATOR_TYPE val = TO_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) * inv_sum;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(val);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#ifdef USE_SEQ_LEN_SPLIT
        {
            // Save temporary exm_sums and max_logits values for each portion
            if (num_of_portions > 1 && sgid == 0) {
                const uint num_of_portions = get_num_groups(2);
                const uint exp_sums_offset = seq_idx * HEADS_NUM * num_of_portions +
                                             head_num_idx * num_of_portions +
                                             portion_id;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }
#endif
    }

    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;

#ifdef USE_SEQ_LEN_SPLIT
        const uint qk_num = ((portion_id + 1) * SEQ_LEN_PORTION_SIZE > context_len) ? (context_len - (portion_id * SEQ_LEN_PORTION_SIZE))
                                                                                    : (SEQ_LEN_PORTION_SIZE);
#else
        const uint qk_num = context_len;
#endif
        for (uint qk_idx = 0; qk_idx < qk_num; qk_idx += SUB_GROUP_SIZE) {
            const uint qk_offset_local = qk_idx + sglid;
            const uint qk_offset_global = block_start_idx * BLOCK_SIZE + qk_offset_local;

            OUTPUT_TYPE qk = qk_offset_global < context_len ? qk_vals_local[qk_offset_local] : OUTPUT_VAL_ZERO;

            const uint block_idx = block_tables[batch_idx * blocks_pitch + block_start_idx + (qk_idx / BLOCK_SIZE)];

            const uint value_cache_offset = block_idx * KV_CACHE_BLOCK_STRIDE +
                                            (head_num_idx / NUM_QUERIES_PER_KV_HEAD) * (HEAD_SIZE * BLOCK_SIZE) +
                                            sgid * (SUB_GROUP_SIZE * BLOCK_SIZE) +
                                            sglid * BLOCK_SIZE +
                                            ((qk_idx / SUB_GROUP_SIZE) % (BLOCK_SIZE / SUB_GROUP_SIZE)) * SUB_GROUP_SIZE;

            #define VALUE_VEC_TYPE MAKE_VECTOR_TYPE(INPUT2_TYPE, SUB_GROUP_SIZE)
            #define AS_VALUE_VEC(val) CAT(as_, VALUE_VEC_TYPE)(val)
#if INPUT2_TYPE_SIZE == 4
            #define VALUE_VLOAD(offset, ptr) CAT(vload, SUB_GROUP_SIZE)(offset, ptr)
#else
            #define VALUE_VLOAD(offset, ptr) CAT(vload, SUB_GROUP_SIZE)(offset, (__global ushort*)(ptr))
#endif
            VALUE_VEC_TYPE v_val = AS_VALUE_VEC(VALUE_VLOAD(0, value_cache + value_cache_offset));

            if (block_start_idx * BLOCK_SIZE + qk_idx + SUB_GROUP_SIZE <= context_len) {
                unroll_for (uint v_idx = 0; v_idx < SUB_GROUP_SIZE; v_idx++) {
                    OUTPUT_TYPE qk_val = sub_group_broadcast(qk, v_idx);
                    acc = mad(qk_val, v_val[v_idx], acc);
                }
            } else {
                for (uint v_idx = 0; v_idx < SUB_GROUP_SIZE; v_idx++) {
                    OUTPUT_TYPE qk_val = sub_group_broadcast(qk, v_idx);
                    if (block_start_idx * BLOCK_SIZE + qk_idx + v_idx < context_len) {
                        acc = mad(qk_val, v_val[v_idx], acc);
                    }
                }
            }
        }

#ifdef USE_SEQ_LEN_SPLIT
        if (num_of_portions > 1) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * num_of_portions) +
                                        head_num_idx * (HEAD_SIZE * num_of_portions) +
                                        portion_id * HEAD_SIZE +
                                        sgid * SUB_GROUP_SIZE +
                                        sglid;

            // tmp_output data layout [num_seqs, num_heads, num_portions, head_size]
            tmp_out[tmp_out_offset] = acc;
        }
        else
#endif
        {
            const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                       head_num_idx * HEAD_SIZE +
                                       sgid * SUB_GROUP_SIZE +
                                       sglid;

            output[output_offset] = acc;
        }

    }
}

#endif

#ifdef SDPA_STAGE_1

#if ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#elif ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#else
#error Unexpected ACCUMULATOR data type size
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pa_sdpa_finalization_stage)(
    const __global INPUT4_TYPE* context_lens,
    __global OUTPUT_TYPE* output,
    const __global ACCUMULATOR_TYPE* exp_sums,
    const __global ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint total_num_of_portions) {
    const uint batch_idx = get_global_id(0);
    const uint token_idx = get_global_id(1);
    const uint head_dim = get_global_id(2);
    const uint head_num_idx = head_dim / HEAD_SIZE;
    const uint head_size_idx = head_dim % HEAD_SIZE;
    const uint sglid = get_sub_group_local_id();

    const uint seq_offset = batch_idx * get_global_size(1) + token_idx;

    const uint num_of_portions = CEIL_DIV(context_lens[batch_idx], SEQ_LEN_PORTION_SIZE);

    if (total_num_of_portions == 1) {
        /* Short path, just copies input to output */
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;
        output[out_offset] = tmp_out[out_offset];
    } else if (num_of_portions <= SUB_GROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PORTION_SIZE(256) * SUB_GROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24) = 98304 tokens */
        ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {ACCUMULATOR_VAL_ZERO};
        ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {ACCUMULATOR_VAL_MIN};
        ACCUMULATOR_TYPE local_exp_sum = ACCUMULATOR_VAL_ZERO;
        ACCUMULATOR_TYPE local_max_logit = ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_portions, SUB_GROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint portion_idx = i * SUB_GROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + portion_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (portion_idx < num_of_portions) {
                exp_sum[i] = exp_sums[exp_sums_offset];
                max_logit[i] = max_logits[max_logit_offset];
                local_max_logit = ACCUMULATOR_MAX_FUNC(local_max_logit, max_logit[i]);
            }
        }

        ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Update exp_sum with respect to the global maximum
        for (uint i = 0; i < iters_num; i++) {
            const uint portion_idx = i * SUB_GROUP_SIZE + sglid;
            if (portion_idx < num_of_portions) {
                exp_sum[i] = exp_sum[i] * native_exp(max_logit[i] - global_max);
                local_exp_sum += exp_sum[i];
            }
        }

        ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        ACCUMULATOR_TYPE acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_offset * (HEADS_NUM * total_num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (total_num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_ACCUMULATOR_TYPE(out_val) * TO_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[portion / SUB_GROUP_SIZE], portion)) / TO_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    } else {
        /* Global memory kernel version, can handle any number of tokens */
        ACCUMULATOR_TYPE local_exp_sum = ACCUMULATOR_VAL_ZERO;
        ACCUMULATOR_TYPE local_max_logit = ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_portions, SUB_GROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint portion_idx = i * SUB_GROUP_SIZE + sglid;
            const uint max_logit_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + portion_idx;

            if (portion_idx < num_of_portions) {
                local_max_logit = ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
            }
        }

        ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Calculate global sum
        for (uint i = 0; i < iters_num; i++) {
            const uint portion_idx = i * SUB_GROUP_SIZE + sglid;
            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + portion_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (portion_idx < num_of_portions) {
                local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
            }
        }

        ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        ACCUMULATOR_TYPE acc = 0.0f;
        for (uint portion = 0; portion < num_of_portions; portion++) {
            const uint tmp_out_offset = seq_offset * (HEADS_NUM * total_num_of_portions * HEAD_SIZE) +
                                        head_num_idx * (total_num_of_portions * HEAD_SIZE) +
                                        portion * HEAD_SIZE +
                                        head_size_idx;

            const uint exp_sums_offset = seq_offset * HEADS_NUM * total_num_of_portions +
                                         head_num_idx * total_num_of_portions + portion;
            const uint max_logit_offset = exp_sums_offset;

            ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_ACCUMULATOR_TYPE(global_sum);
        }
        const uint out_offset = seq_offset * (HEADS_NUM * HEAD_SIZE) +
                                head_num_idx * HEAD_SIZE +
                                head_size_idx;

        output[out_offset] = TO_OUTPUT_TYPE(acc);
    }
}

#endif
