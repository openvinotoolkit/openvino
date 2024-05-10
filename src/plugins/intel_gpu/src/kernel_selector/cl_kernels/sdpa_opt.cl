// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]


#define VALUE_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT2_TYPE, 1, ptr, offset)
#define SOURCE_SEQ_LEN INPUT1_SIZE_Y
#define TARGET_SEQ_LEN INPUT0_SIZE_Y
#define SUBGROUPS_PER_WG (HEAD_SIZE / SUBGROUP_SIZE)

#ifdef SDPA_STAGE_0

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global INPUT4_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / INPUT0_FEATURE_NUM; /* BATCH dim */
    const uint b1_idx = batch_idx % INPUT0_FEATURE_NUM; /* HEADS_NUM dim */

#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
    const uint target_seq_idx = (uint)get_global_id(1) * TARGET_SEQ_LEN_BLOCK_SIZE;
#else
    const uint target_seq_idx = get_global_id(1);
#endif
    const uint lid = get_local_id(2);
    const uint head_size_idx = lid;

    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint partition_idx = get_group_id(2);
    const uint num_of_partitions = get_num_groups(2);
    const uint wi_num_per_partition = get_local_size(2);

    const uint start_partition_idx = partition_idx * SEQ_LEN_PARTITION_SIZE;
    const uint partition_seq_len =
        ((partition_idx + 1) < num_of_partitions) ? (SEQ_LEN_PARTITION_SIZE)
                                                  : (SOURCE_SEQ_LEN - partition_idx * SEQ_LEN_PARTITION_SIZE);

    // SLM for query inputs
    __local INPUT0_TYPE query_local[HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM for intermediate QK results
    __local OUTPUT_TYPE qk_local[SEQ_LEN_PARTITION_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WG
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];

    {
        // Gemm1 and SoftMax calculation

        SOFTMAX_ACCUMULATOR_TYPE qk_max[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
            qk_max[i] = SOFTMAX_ACCUMULATOR_VAL_MIN;
        }

        {
            // Gemm1 calculation
#if HAS_SCALE_INPUT
            const OUTPUT_TYPE scale_val = *scale;
#else
            const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(HEAD_SIZE));
#endif
            {
                // Query input loading to SLM
                #define QUERY_STEP SUBGROUP_SIZE * SUBGROUPS_PER_WG
                uint query_local_offset = sgid * SUBGROUP_SIZE + sglid;

#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
                const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#else
                const uint seq_idx_end = 1;
#endif
                uint query_offset = INPUT0_GET_INDEX(b0_idx, b1_idx, target_seq_idx, (sgid * SUBGROUP_SIZE));
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    #define QUERY_BLOCK_SIZE 1

                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);

                    query_local[query_local_offset] = val;
                    query_local_offset += QUERY_STEP;
                    query_offset += QUERY_STEP;
                }
                #undef QUERY_BLOCK_SIZE
                #undef QUERY_STEP

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // Main Gemm1 calculation loop
            // Each SG performs element-wise multiplications of Q[HEAD_SIZE]xK[HEAD_SIZE] values
            // HEAD_SIZE / SUBGROUPS_PER_WG times in the loop and saves the result to the qk_local SLM buffer
            for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE / SUBGROUP_SIZE)) {
                uint key_offset = INPUT1_GET_INDEX(b0_idx, b1_idx, start_partition_idx + seq_len, 0);

                INPUT0_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {INPUT0_VAL_ZERO};

                uint head_idx_index = 0;
                #define KEY_BLOCK_SIZE 8
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 4
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 2
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(query_vals_reg[i], key_vals[i], acc[seq_idx]);
                        }

                        query_offset += HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 1
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vals = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        acc[seq_idx] = mad(query_vals_reg, key_vals, acc[seq_idx]);
                        query_offset += HEAD_SIZE;
                    }
                }

                // Sum up all accumulators accross single SG and save result to SLM
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = sub_group_reduce_add(acc[seq_idx]);
                    qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx];
                }
            }

            {
                // Wait until all SG finishes their calculations and apply scale and attention mask to the results
                barrier(CLK_LOCAL_MEM_FENCE);

                INPUT0_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
                const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#else
                const uint seq_idx_end = 1;
#endif
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    // Iterate over all values QK values in SLM and apply scale and attention mask
                    for (uint seq_len = sgid * SUBGROUP_SIZE + sglid; seq_len < partition_seq_len; seq_len += (HEAD_SIZE)) {
                        // Read value from SLM and apply scale
                        qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];
                        qk_val[seq_idx] *= scale_val;

                        // Apply attention mask
#if IS_CAUSAL
                        if (start_partition_idx + seq_len > target_seq_idx + seq_idx)
                            qk_val[seq_idx] += INPUT0_VAL_MIN;
#elif !IS_CAUSAL && HAS_ATTN_MASK_INPUT
                        const uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0_idx, b1_idx, target_seq_idx + seq_idx, start_partition_idx + seq_len);
                        qk_val[seq_idx] += attn_mask[attn_mask_offset];
#endif

                        // Update qk_max value
                        qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx], TO_SOFTMAX_ACCUMULATOR_TYPE(qk_val[seq_idx]));

                        // Save modified qk value back to SLM
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = qk_val[seq_idx];
                    }
                }
            }
        } // Gemm1 calculation end

        {
            // SoftMax calculation
#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
            const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#else
            const uint seq_idx_end = 1;
#endif
            // Find the maximum value of qk in the subgroup
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            // Find the maximum value of qk across all subgroups in the workgroup
            if (sglid == 0) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = qk_max[seq_idx];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_MIN;

                if (sglid < SUBGROUPS_PER_WG)
                    qk_max[seq_idx] = qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Final maximum value of qk after reduction across all subgroups
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            SOFTMAX_ACCUMULATOR_TYPE exp_sum[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
            const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + head_size_idx;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) - qk_max[seq_idx]);
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                        exp_sum[seq_idx] += qk_new;
                    }
                }
            }

            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);

                if (sglid == 0)
                    qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = exp_sum[seq_idx];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                exp_sum[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_ZERO;

                if (sglid < SUBGROUPS_PER_WG)
                    exp_sum[seq_idx] = qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Find the final sum of all exp_sum[seq_idx] values in workgroup
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);
            }

            // const SOFTMAX_ACCUMULATOR_TYPE inv_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum[seq_idx];
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + sgid * SUBGROUP_SIZE + sglid;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) / exp_sum[seq_idx];
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            {
                // If the number of partitions is greater than 1, save exm_sums and max_logits to the temporary buffers
                // Use single WI in the WG, since all the WIs have the same value
                if (num_of_partitions > 1 && head_size_idx == 0) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        const uint exp_sums_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                                    b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                                    (seq_idx + target_seq_idx) * (num_of_partitions) +
                                                    partition_idx;
                        exp_sums[exp_sums_offset] = exp_sum[seq_idx];

                        const uint max_logits_offset = exp_sums_offset;
                        max_logits[max_logits_offset] = qk_max[seq_idx];
                    }
                }
            }
        } // SoftMax calculation end
    } // Gemm1 + SoftMax calculations end

    {
        // Gemm2 calculation
        OUTPUT_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {OUTPUT_VAL_ZERO};

        for (uint seq_len = 0; seq_len < partition_seq_len / SUBGROUP_SIZE; seq_len++) {
            uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len * SUBGROUP_SIZE + sglid];
            }

            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc[seq_idx]);
                }

                value_offset += HEAD_SIZE;
            }
        }

        const uint seq_len_leftover_start = (partition_seq_len / SUBGROUP_SIZE) * SUBGROUP_SIZE;
        for (uint seq_len = seq_len_leftover_start; seq_len < partition_seq_len; seq_len++) {
            const uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + seq_len, head_size_idx);

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];
            }

            INPUT2_TYPE value_val = VALUE_BLOCK_READ(value_input, value_offset);

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                acc[seq_idx] = mad(qk_val[seq_idx], value_val, acc[seq_idx]);
            }
        }

        // If the number of partitions is greater than 1, save results to the temporary buffer;
        // otherwise, save results directly to the main output.
        if (num_of_partitions > 1) {
#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
            const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#else
            const uint seq_idx_end = 1;
#endif
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                // Data layout of tmp_output buf: [batch, heads_num, q_len, partition_idx, head_size]
                const uint tmp_out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            (target_seq_idx + seq_idx) * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            head_size_idx;
                tmp_out[tmp_out_offset] = acc[seq_idx];
            }
        } else {
#if TARGET_SEQ_LEN_BLOCK_SIZE > 1
            const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#else
            const uint seq_idx_end = 1;
#endif
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, head_size_idx);

                    output[output_offset] = acc[seq_idx];
            }
        }
    } // Gemm2 calculation end
}

#endif

#ifdef SDPA_STAGE_1

// MTL iGPU faces high register pressure issue with a higher number of REG_VERSION_MAX_VALUES_PER_WI.
// To mitigate this, add an additional level of SDPA results processing
// with lower register pressure (REG_VERSION_MAX_VALUES_PER_WI_LOWER).

#if SOFTMAX_ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 8
#elif SOFTMAX_ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 16
#else
#error Unexpected SOFTMAX_ACCUMULATOR data type size
#endif

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt_finalization_stage)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint num_of_partitions) {
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / INPUT0_FEATURE_NUM;
    const uint b1_idx = batch_idx % INPUT0_FEATURE_NUM;
    const uint target_seq_idx = get_global_id(1);
    const uint sglid = get_sub_group_local_id();

    if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI_LOWER) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI_LOWER(8/16) = 32768/65536 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI_LOWER] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI_LOWER] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
                                         partition_idx;
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

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);
                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) *
                    TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_idx / SUBGROUP_SIZE], partition_idx % SUBGROUP_SIZE)) /
                    TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }
            const uint out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    } else if (num_of_partitions <= SUBGROUP_SIZE * REG_VERSION_MAX_VALUES_PER_WI) {
        /* Registers kernel version, can handle up to SEQ_LEN_PARTITION_SIZE(256) * SUBGROUP_SIZE(16) * REG_VERSION_MAX_VALUES_PER_WI(24/48) = 98304/196608 tokens */
        SOFTMAX_ACCUMULATOR_TYPE exp_sum[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
        SOFTMAX_ACCUMULATOR_TYPE max_logit[REG_VERSION_MAX_VALUES_PER_WI] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
                                         partition_idx;
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

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);
                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) *
                    TO_SOFTMAX_ACCUMULATOR_TYPE(sub_group_broadcast(exp_sum[partition_idx / SUBGROUP_SIZE], partition_idx % SUBGROUP_SIZE)) /
                    TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }
            const uint out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    } else {
        /* Global memory kernel version, can handle any number of tokens, but could be very slow. */
        SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
        SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;

        const uint iters_num = CEIL_DIV(num_of_partitions, SUBGROUP_SIZE);
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint max_logit_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                          b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                          target_seq_idx * (num_of_partitions) +
                                          partition_idx;


            if (partition_idx < num_of_partitions) {
                local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits[max_logit_offset]);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

        // Calculate global sum
        for (uint i = 0; i < iters_num; i++) {
            const uint partition_idx = i * SUBGROUP_SIZE + sglid;
            const uint exp_sums_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions) +
                                         partition_idx;
            const uint max_logit_offset = exp_sums_offset;

            if (partition_idx < num_of_partitions) {
                local_exp_sum += exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);
            }
        }

        SOFTMAX_ACCUMULATOR_TYPE global_sum = sub_group_reduce_add(local_exp_sum);

        for (uint head_size_idx = 0; head_size_idx < HEAD_SIZE / SUBGROUP_SIZE; head_size_idx++) {
            SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
            for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
                const uint tmp_out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * HEAD_SIZE) +
                                            target_seq_idx * (num_of_partitions * HEAD_SIZE) +
                                            partition_idx * (HEAD_SIZE) +
                                            (head_size_idx * SUBGROUP_SIZE + sglid);

                const uint exp_sums_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * num_of_partitions) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                            target_seq_idx * (num_of_partitions) +
                                            partition_idx;
                const uint max_logit_offset = exp_sums_offset;

                SOFTMAX_ACCUMULATOR_TYPE new_exp_sum = exp_sums[exp_sums_offset] * native_exp(max_logits[max_logit_offset] - global_max);

                OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
                acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * new_exp_sum / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
            }

            const uint out_offset = b0_idx * (INPUT0_FEATURE_NUM * TARGET_SEQ_LEN * HEAD_SIZE) +
                                    b1_idx * (TARGET_SEQ_LEN * HEAD_SIZE) +
                                    target_seq_idx * (HEAD_SIZE) +
                                    (head_size_idx * SUBGROUP_SIZE + sglid);

            output[out_offset] = TO_OUTPUT_TYPE(acc);
        }
    }
}

#endif
