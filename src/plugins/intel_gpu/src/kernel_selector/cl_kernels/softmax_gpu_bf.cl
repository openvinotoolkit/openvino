// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#if IS_DYNAMIC

#define CALC_POWER(n) ({uint pos = 0; uint i = n; do { i >>= 1; ++pos; } while (i); --pos;})

#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define BLOCK_TYPE INPUT0_TYPE

#define OPT_BLOCK_SIZE 8

#define BLOCK_READ_OPT(ptr, offset) CAT(DT_INPUT_BLOCK_READ, OPT_BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE_OPT(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, OPT_BLOCK_SIZE)(ptr, offset, val)
#define BLOCK_TYPE_OPT MAKE_VECTOR_TYPE(INPUT0_TYPE, OPT_BLOCK_SIZE)

#else

#if SUBGROUP_BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define BLOCK_TYPE INPUT0_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, SUBGROUP_BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, SUBGROUP_BLOCK_SIZE)(ptr, offset, val)
#define BLOCK_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, SUBGROUP_BLOCK_SIZE)
#endif

#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL (softmax_gpu_continuous_bfyx)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     // in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          // how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  // this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       // how many elements are in one data set
    const uint data_sets_count = DATA_SETS_COUNT;   // how many data sets are in the processing payload

    // To use subgroup block write functions, offset should be 16 bytes aligned.
    // ************************************************************************
    // | aligned_offset | 16 bytes aligned data offset | actual leftovers
    // ************************************************************************
    // leftover = aligned_offset + actual_leftovers
#if !IS_DYNAMIC
    const uint origin_items_num = ITEMS_NUM;               // how many elements are processed per one WI
    const uint origin_leftovers = LEFTOVERS;
#else
    // since workers_per_data_set is calculated by power of 2
    // items_num can be calculated by dividing data_set_size by power of 2
    const uint power = CALC_POWER(workers_per_data_set);
    const uint origin_items_num = data_set_size>>power;
    const uint origin_leftovers = data_set_size-(origin_items_num<<power);
#endif
    const uint data_set_offset = data_set_idx * data_set_size;
    const uint data_set_offset_byte_counts = data_set_offset * sizeof(INPUT0_TYPE);
    const uint aligned_offset = ((workers_per_data_set > SUB_GROUP_SIZE) && (data_set_offset_byte_counts & 0xF))
                ? ((((data_set_offset_byte_counts >> 4) + 1) << 4) / sizeof(INPUT0_TYPE) - data_set_offset) : 0;
    const uint items_num = (origin_leftovers < aligned_offset) ? (origin_items_num - 1) : origin_items_num;
    const uint leftovers = (origin_leftovers < aligned_offset) ? (origin_leftovers + workers_per_data_set) : origin_leftovers;

    const uint subgroup_offset = get_sub_group_id() * get_sub_group_size() * items_num;
    const uint aligned_data_offset = data_set_offset + subgroup_offset + aligned_offset;
    const uint actual_leftovers = leftovers - aligned_offset;
    const uint leftover_idx = data_set_offset + aligned_offset + workers_per_data_set * items_num + in_data_set_idx;

    INPUT0_TYPE my_chunk[STACK_SIZE];
    INPUT0_TYPE my_sum = UNIT_VAL_ZERO;

    __local INPUT0_TYPE lg_storage[SLM_SIZE];

    // Read inputs and Get maximum value from data set
    uint input_idx=0;
#if IS_DYNAMIC
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; input_idx < num_iters; input_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp = BLOCK_READ_OPT(input, aligned_data_offset + input_idx * get_sub_group_size());
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++)
            {
                my_chunk[input_idx+j] = vec_tmp[j];
            }
        }

        for (; input_idx < items_num; input_idx++)
        {
            BLOCK_TYPE vec_tmp = BLOCK_READ(input, aligned_data_offset + input_idx * get_sub_group_size());
            my_chunk[input_idx] = vec_tmp;
        }
    }
#else
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        for (; input_idx<items_num - (items_num % SUBGROUP_BLOCK_SIZE); input_idx+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp = BLOCK_READ(input, aligned_data_offset + input_idx * get_sub_group_size());
#if SUBGROUP_BLOCK_SIZE == 1
            my_chunk[input_idx] = vec_tmp;
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                INPUT0_TYPE tmp = vec_tmp[j];
                my_chunk[input_idx+j] = tmp;
            }
#endif
        }
    }
#endif

    for (; input_idx < items_num; input_idx++)
    {
        my_chunk[input_idx] = input[aligned_data_offset + get_sub_group_local_id() + input_idx * get_sub_group_size()];
    }

    if (in_data_set_idx < aligned_offset)
    {
        INPUT0_TYPE tmp = input[data_set_offset + in_data_set_idx];
        my_chunk[input_idx++] = tmp;
    }

    if (in_data_set_idx < actual_leftovers)
    {
        INPUT0_TYPE tmp = input[leftover_idx];
        my_chunk[input_idx++] = tmp;
    }

    INPUT0_TYPE my_maximum = -UNIT_VAL_MAX;
    {
        const uint num_iters = input_idx;
        for (uint j=0; j<num_iters; ++j)
        {
            my_maximum = max(my_maximum, my_chunk[j]);
        }
    }

    my_maximum = sub_group_reduce_max(my_maximum);

    if (get_sub_group_local_id() == 0)
        lg_storage[get_sub_group_id()] = my_maximum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint j=1; j<get_num_sub_groups(); ++j)
            my_maximum = max(my_maximum, lg_storage[j]);

        lg_storage[0] = my_maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //my_maximum from this point is in fact global maximum
    my_maximum = lg_storage[0];

    // Get exp(x-max) and sum of exp(x-max)
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint num_iters = input_idx;
    for (uint j=0; j<num_iters; ++j)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[j] - my_maximum);
        my_sum += tmp;
        my_chunk[j] = tmp;
    }

    my_sum = sub_group_reduce_add(my_sum);

    if (get_sub_group_local_id() == 0)
        lg_storage[get_sub_group_id()] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint j=1; j<get_num_sub_groups(); ++j)
            my_sum += lg_storage[j];

        lg_storage[0] = my_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    my_sum = lg_storage[0];

    // Write outputs
    uint output_idx = 0;
#if HAS_FUSED_OPS
#if IS_DYNAMIC
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; output_idx < num_iters; output_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp;
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++)
            {
                ACTIVATION_TYPE dequantized = my_chunk[output_idx + j] / my_sum;
                FUSED_OPS_MAIN;
                vec_tmp[j] = FUSED_OPS_RESULT_MAIN;
            }
            BLOCK_WRITE_OPT(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }

        for (; output_idx<items_num; output_idx++)
        {
            ACTIVATION_TYPE dequantized = my_chunk[output_idx] / my_sum;
            FUSED_OPS_MAIN;
            vec_tmp = FUSED_OPS_RESULT_MAIN;
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }
#else
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        for (; output_idx < items_num - (items_num % SUBGROUP_BLOCK_SIZE); output_idx+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp;
#if SUBGROUP_BLOCK_SIZE == 1
            ACTIVATION_TYPE dequantized = my_chunk[output_idx] / my_sum;
            FUSED_OPS_MAIN;
            vec_tmp = FUSED_OPS_RESULT_MAIN;
#else
            for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                ACTIVATION_TYPE dequantized = my_chunk[output_idx + j] / my_sum;
                FUSED_OPS_MAIN;
                vec_tmp[j] = FUSED_OPS_RESULT_MAIN;
            }
#endif
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }
#endif
    for (; output_idx < items_num; output_idx++)
    {
        ACTIVATION_TYPE dequantized = my_chunk[output_idx] / my_sum;
        FUSED_OPS_MAIN;
        output[aligned_data_offset + get_sub_group_local_id() + i * get_sub_group_size()] = FUSED_OPS_RESULT_MAIN;
    }

    if (in_data_set_idx < aligned_offset)
    {
        ACTIVATION_TYPE dequantized = my_chunk[output_idx++] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[data_set_offset + in_data_set_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }

    if (in_data_set_idx < actual_leftovers)
    {
        ACTIVATION_TYPE dequantized = my_chunk[output_idx++] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[leftover_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }
#else
#if IS_DYNAMIC
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; output_idx < num_iters; output_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp;
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++)
                vec_tmp[j] = ACTIVATION(my_chunk[output_idx + j] / my_sum, ACTIVATION_PARAMS);
            BLOCK_WRITE_OPT(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }

        for (; output_idx < items_num; output_idx++)
        {
            BLOCK_TYPE vec_tmp;
            vec_tmp = ACTIVATION(my_chunk[output_idx] / my_sum, ACTIVATION_PARAMS);
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }
#else
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        for (; output_idx<items_num - (items_num % SUBGROUP_BLOCK_SIZE); output_idx+=SUBGROUP_BLOCK_SIZE)
        {
            BLOCK_TYPE vec_tmp;
#if SUBGROUP_BLOCK_SIZE == 1
            vec_tmp = ACTIVATION(my_chunk[output_idx] / my_sum, ACTIVATION_PARAMS);
#else
            for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
                vec_tmp[j] = ACTIVATION(my_chunk[output_idx + j] / my_sum, ACTIVATION_PARAMS);
#endif
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }
#endif
    for (; output_idx < items_num; output_idx++)
    {
        output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()] = ACTIVATION(my_chunk[output_idx] / my_sum, ACTIVATION_PARAMS);
    }

    if (in_data_set_idx < aligned_offset)
        output[data_set_offset + in_data_set_idx] = ACTIVATION(my_chunk[output_idx++] / my_sum, ACTIVATION_PARAMS);

    if (in_data_set_idx < actual_leftovers)
        output[leftover_idx] = ACTIVATION(my_chunk[output_idx++] / my_sum, ACTIVATION_PARAMS);
#endif
}
#ifdef CALC_POWER
#undef CALC_POWER
#endif
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef BLOCK_TYPE

