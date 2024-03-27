// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define SUBGROUP_SIZE 16
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUBGROUP_SIZE)

// The size of partition_num of HEAD_SIZE each WI process
#define Q_LOAD_ITERS (HEAD_SIZE / SUBGROUP_SIZE)

// How much QK outputs each subgroup calculates per block
#define QK_VALS_PER_SG_PER_ITER CEIL_DIV(VLLM_BLOCK_SIZE, SUBGROUPS_PER_WG)

#define KV_CACHE_BLOCK_STRIDE (HEAD_SIZE * KV_HEADS_NUM * VLLM_BLOCK_SIZE)

#define QUERY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

#define VLLM_BLOCKS_PER_PARTITION SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE


#ifdef SDPA_STAGE_0

// key shape   [num_blocks, NUM_HEADS, head_size, VLLM_BLOCK_SIZE]
// value shape [num_blocks, NUM_HEADS, VLLM_BLOCK_SIZE, head_size]

#if SEQ_LEN_PARTITION_SIZE % VLLM_BLOCK_SIZE != 0
    #error pa_sdpa_opt.cl
#endif

#if SUBGROUP_SIZE != VLLM_BLOCK_SIZE
    #error pa_sdpa_opt.cl
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, HEAD_SIZE)))
KERNEL(pa_sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query,
    const __global INPUT1_TYPE* key_cache,
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT3_TYPE* past_lens,
    const __global INPUT4_TYPE* subsequence_begins,
    const __global INPUT5_TYPE* block_indices,
    const __global INPUT6_TYPE* block_indices_begins,
    __global OUTPUT_TYPE* output
    , __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums
    , __global SOFTMAX_ACCUMULATOR_TYPE* max_logits
    , __global OUTPUT_TYPE* tmp_out
) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_size_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint total_partitions_num = get_num_groups(2);

    const uint batch_idx = seq_idx;

    // const uint context_len = context_lens[batch_idx];
    const uint seq_len = past_lens[seq_idx] + 1;

    const uint partition_idx = get_group_id(2);
    const uint block_start_idx = partition_idx * SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE;

    if (partition_idx * SEQ_LEN_PARTITION_SIZE >= seq_len) {
        return;
    }

    const uint total_blocks_num = CEIL_DIV(seq_len, VLLM_BLOCK_SIZE);

    __local OUTPUT_TYPE qk_vals_local[SHARED_MEM_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG];

    SOFTMAX_ACCUMULATOR_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;

    {

        INPUT0_TYPE q_val[HEAD_SIZE / SUBGROUP_SIZE];
        unroll_for (uint i = 0; i < HEAD_SIZE / SUBGROUP_SIZE; i++) {
            const uint query_idx = seq_idx * HEAD_SIZE * HEADS_NUM +
                                   head_num_idx * HEAD_SIZE +
                                   i * SUBGROUP_SIZE;
            q_val[i] = QUERY_BLOCK_READ(query, query_idx);
        }

        // uint blocks_num = ((partition_idx + 1) * SEQ_LEN_PARTITION_SIZE > context_len) ? (total_blocks_num - (partition_idx * SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE))
                                                                                    //    : (SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE);

        // blocks_num = blocks_num / SEQ_LEN_PARTITION_SIZE / VLLM_BLOCK_SIZE;

        // const uint blocks_per_sg = SEQ_LEN_PARTITION_SIZE / SUBGROUP_SIZE / SUBGROUPS_PER_WG;
        const uint blocks_num_per_partition = min(total_blocks_num - partition_idx * VLLM_BLOCKS_PER_PARTITION, (uint)VLLM_BLOCKS_PER_PARTITION);



        uint blocks_num = blocks_num_per_partition / SUBGROUPS_PER_WG;
        if (sgid < blocks_num_per_partition % SUBGROUPS_PER_WG)
            blocks_num++;

        const uint start_block_idx = block_indices_begins[seq_idx] + partition_idx * VLLM_BLOCKS_PER_PARTITION + sgid;
        for (uint block_num = 0; block_num < blocks_num; block_num++) {
            const uint block_offset = block_indices[start_block_idx + block_num * SUBGROUPS_PER_WG] * HEAD_SIZE * HEADS_NUM * VLLM_BLOCK_SIZE + head_num_idx * HEAD_SIZE * VLLM_BLOCK_SIZE;

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && sglid == 0 && (sgid == 0 || sgid == 1 || sgid == 2)) {
            //     printf("sgid=%d, seq_len=%d, block_start_idx=%d, total_blocks_num=%d, blocks_num_per_partition=%d, blocks_num=%d, start_block_idx=%d, past_lens[seq_idx]=%d. block_num=%d, block_indices[]=%d, partition_idx=%d\n",
            //         sgid, seq_len, block_start_idx, total_blocks_num, blocks_num_per_partition, blocks_num, start_block_idx, past_lens[seq_idx], block_num, block_indices[start_block_idx + block_num * SUBGROUPS_PER_WG], partition_idx);
            // }

            INPUT0_TYPE qk_acc = INPUT0_VAL_ZERO;

            #define KEY_VEC_SIZE 16
            #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset);
            #define KEY_VEC MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_VEC_SIZE)
            unroll_for (uint qk_idx = 0; qk_idx < HEAD_SIZE / KEY_VEC_SIZE; qk_idx++) {
                KEY_VEC k_vals = 0;
                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                    k_vals[i] = KEY_BLOCK_READ(key_cache, block_offset + qk_idx * VLLM_BLOCK_SIZE * KEY_VEC_SIZE + i * SUBGROUP_SIZE);
                }

                // if (seq_len == 16 && (sglid == 15 || sglid == 0) && head_num_idx == 0) {
                //     unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                //         printf("%d. q=%f k=%f\n", qk_idx * KEY_VEC_SIZE + i,  sub_group_broadcast(q_val[qk_idx], i), k_vals[i]);
                //     }
                // }

                unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                    qk_acc = mad(sub_group_broadcast(q_val[qk_idx], i), k_vals[i], qk_acc);
                }

                // KEY_VEC_SIZE 16
                // KEY_VEC k_vals = 0;
                // k_vals = BLOCK_READN(key_cache, block_offset + qk_idx * VLLM_BLOCK_SIZE * KEY_VEC_SIZE);

                // unroll_for (uint i = 0; i < KEY_VEC_SIZE; i++) {
                //     qk_acc = mad(sub_group_broadcast(q_val[qk_idx / 2], (qk_idx % 2 /* qk_idx & 1 */) * KEY_VEC_SIZE + i), k_vals, qk_acc);
                // }
            }

#ifdef SCALE_VAL
            qk_acc = TO_INPUT0_TYPE(SCALE_VAL) * qk_acc;
#endif

            if (partition_idx * SEQ_LEN_PARTITION_SIZE + block_num * SUBGROUPS_PER_WG * SUBGROUP_SIZE + sgid * SUBGROUP_SIZE + sglid >= seq_len)
                qk_acc = INPUT0_VAL_MIN;

            qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc));

            qk_vals_local[block_num * SUBGROUPS_PER_WG * SUBGROUP_SIZE + sgid * SUBGROUP_SIZE + sglid] = qk_acc;
        }


        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
        //     printf("sgid=%d. qk_max before=%f\n", sgid, qk_max);
        // }


        qk_max = sub_group_reduce_max(qk_max);


        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
        //     printf("sgid=%d. qk_max after=%f\n", sgid, qk_max);
        // }
    }

    // Apply SoftMax operation
    {
        if (sglid == 0) {
            qk_max_vals[sgid] = qk_max;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
        if (sglid < SUBGROUPS_PER_WG)
            qk_max = qk_max_vals[sglid];

        // Final max value after reduction across of all SG and WI
        qk_max = sub_group_reduce_max(qk_max);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
        //     printf("sgid=%d. Total qk_max=%f\n", sgid, qk_max);
        // }

        SOFTMAX_ACCUMULATOR_TYPE exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        const uint qk_iters_num = CEIL_DIV(SEQ_LEN_PARTITION_SIZE, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
        for (uint qk_idx = 0; qk_idx < qk_iters_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            // TODO: const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx
            const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;

            // ??? Why it was:
            // if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
            if (global_data_idx < seq_len) {
                SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) - qk_max);
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0)) {
                //     printf("sgid=%d sglid=%d. local_data_idx=%d global_data_idx=%d. Updated val = %f\n", sgid, sglid, local_data_idx, global_data_idx, qk_vals_local[local_data_idx]);
                // }

                exp_sum += qk_new;
            }
        }

        exp_sum = sub_group_reduce_add(exp_sum);

        if (sglid == 0)
            qk_sum_vals[sgid] = exp_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;

        if (sglid < SUBGROUPS_PER_WG)
            exp_sum = qk_sum_vals[sglid];

        // Final sum of all exp_sum values
        exp_sum = sub_group_reduce_add(exp_sum);

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
        //     printf("sgid=%d. Total exp_sum=%f\n", sgid, exp_sum);
        // }

        for (uint qk_idx = 0; qk_idx < qk_iters_num; qk_idx++) {
            const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
            const uint global_data_idx = partition_idx * SEQ_LEN_PARTITION_SIZE + qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;

            // ??? Why it was:
            // if (global_data_idx < context_len && local_data_idx < SEQ_LEN_PARTITION_SIZE) {
            if (global_data_idx < seq_len) {
                SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_vals_local[local_data_idx]) / exp_sum;
                qk_vals_local[local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
                //     printf("sgid=%d sglid=%d. local_data_idx=%d global_data_idx=%d. Updated qk_vals_local = %f\n", sgid, sglid, local_data_idx, global_data_idx, qk_vals_local[local_data_idx]);
                // }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        {

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0)) {
            //     printf("sgid=%d total_partitions_num=%d\n", sgid, total_partitions_num);
            // }
            // Save temporary exm_sums and max_logits values for each partition_num
            if (seq_len > SEQ_LEN_PARTITION_SIZE && sgid == 0) {
                const uint exp_sums_offset = seq_idx * HEADS_NUM * total_partitions_num +
                                             head_num_idx * total_partitions_num +
                                             partition_idx;
                exp_sums[exp_sums_offset] = exp_sum;

                const uint max_logits_offset = exp_sums_offset;
                max_logits[max_logits_offset] = qk_max;
            }
        }
    }

    {
        OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;

        // const uint qk_num = ((partition_idx + 1) * SEQ_LEN_PARTITION_SIZE > context_len) ? (context_len - (partition_idx * SEQ_LEN_PARTITION_SIZE))
        //                                                                             : (SEQ_LEN_PARTITION_SIZE);

        const uint partition_seq_len = min(seq_len - partition_idx * SEQ_LEN_PARTITION_SIZE, (uint)SEQ_LEN_PARTITION_SIZE);
        uint blocks_num_per_partition = min(total_blocks_num - partition_idx * VLLM_BLOCKS_PER_PARTITION, (uint)VLLM_BLOCKS_PER_PARTITION);

        uint leftovers = blocks_num_per_partition * VLLM_BLOCK_SIZE - partition_seq_len;
        if (leftovers != 0) {
            leftovers = VLLM_BLOCK_SIZE - leftovers;
            blocks_num_per_partition = blocks_num_per_partition - 1;
        }

        const uint start_block_idx = block_indices_begins[seq_idx] + partition_idx * VLLM_BLOCKS_PER_PARTITION;

        // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
        //     printf("sgid=%d partition_seq_len=%d blocks_num_per_partition=%d leftovers=%d start_block_idx=%d\n", sgid, partition_seq_len, blocks_num_per_partition, leftovers, start_block_idx);
        // }

        for (uint block_num = 0; block_num < blocks_num_per_partition; block_num++) {
            const uint block_offset = block_indices[start_block_idx + block_num] * HEADS_NUM * HEAD_SIZE * VLLM_BLOCK_SIZE + head_num_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + sgid * SUBGROUP_SIZE;

            #define VALUE_VEC_SIZE 16
            #define VALUE_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT2_TYPE, 1, ptr, offset);
            #define VALUE_VEC MAKE_VECTOR_TYPE(INPUT2_TYPE, VALUE_VEC_SIZE)

            #if VALUE_VEC_SIZE != VLLM_BLOCK_SIZE
            #error pa_sdpa_opt.cl
            #endif

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
            //     printf("main QKV: block_num=%d sgid=%d block_offset=%d in_block_offset=%d\n", block_num, sgid, block_offset, head_num_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + sgid * SUBGROUP_SIZE);
            // }

            OUTPUT_TYPE qk_val = qk_vals_local[block_num * VLLM_BLOCK_SIZE + sglid];

            VALUE_VEC value_vals;
            unroll_for (uint i = 0; i < VALUE_VEC_SIZE; i++) {
                value_vals[i] = VALUE_BLOCK_READ(value_cache, block_offset + i * HEAD_SIZE);
            }

            unroll_for (uint i = 0; i < VALUE_VEC_SIZE; i++) {
                acc = mad(sub_group_broadcast(qk_val, i), value_vals[i], acc);
            }
        }

        if (leftovers != 0) {
            const uint last_block_idx = start_block_idx + blocks_num_per_partition;
            const uint block_offset = block_indices[last_block_idx] * HEAD_SIZE * HEADS_NUM * VLLM_BLOCK_SIZE + head_num_idx * HEAD_SIZE * VLLM_BLOCK_SIZE + sgid * SUBGROUP_SIZE;

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1) && sglid == 0) {
            //     printf("sgid=%d last_block_idx=%d block_indices[last_block_idx]=%d leftovers=%d blocks_num_per_partition=%d\n", sgid, last_block_idx, block_indices[last_block_idx], leftovers, blocks_num_per_partition);
            // }

            OUTPUT_TYPE qk_val = qk_vals_local[blocks_num_per_partition * VLLM_BLOCK_SIZE + sglid];
            for (uint i = 0; i < leftovers; i++) {
                INPUT2_TYPE value_val = BLOCK_READN(INPUT2_TYPE, 1, value_cache, block_offset + i * HEAD_SIZE);
                acc = mad(sub_group_broadcast(qk_val, i), value_val, acc);
            }
        }

        if (seq_len > SEQ_LEN_PARTITION_SIZE) {
            const uint tmp_out_offset = seq_idx * (HEADS_NUM * HEAD_SIZE * total_partitions_num) +
                                        head_num_idx * (HEAD_SIZE * total_partitions_num) +
                                        partition_idx * HEAD_SIZE +
                                        sgid * SUBGROUP_SIZE +
                                        sglid;

            // tmp_output data layout [num_seqs, num_heads, total_partitions_num, head_size]
            tmp_out[tmp_out_offset] = acc;
        } else {
            const uint output_offset = seq_idx * (HEADS_NUM * HEAD_SIZE) +
                                       head_num_idx * HEAD_SIZE +
                                       sgid * SUBGROUP_SIZE +
                                       sglid;

            // if (get_global_id(0) == 0 && get_global_id(1) == 0 && (sgid == 0 || sgid == 1)) {
            //     printf("sgid=%d output_offset=%d, acc=%f\n", sgid, output_offset, acc);
            // }

            output[output_offset] = acc;
        }

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
    __global OUTPUT_TYPE* output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint total_partitions_num) {
    const uint seq_idx = get_global_id(0);
    const uint head_num_idx = get_global_id(1);
    const uint head_size_idx = get_global_id(2);
    const uint sglid = get_sub_group_local_id();

    const uint seq_len = past_lens[seq_idx] + 1;

    const uint num_of_partitions = CEIL_DIV(seq_len, SEQ_LEN_PARTITION_SIZE);

    if (seq_len <= SEQ_LEN_PARTITION_SIZE) {
        /* Short path, no need any actions for currently processing sequnce */
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
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_num / SUBGROUP_SIZE], partition_num)) / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
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
