// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#if IS_DYNAMIC

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

#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS, 1, 1)))
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
    const uint origin_items_num = data_set_size / workers_per_data_set;
    const uint origin_leftovers = data_set_size % workers_per_data_set;

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

#if IS_DYNAMIC
    // use output buffer as intermediate variable instead of my_chunk when (item_num+2) is bigger than STACK_SIZE
    // this happens when data_set_size > 16384 (engineInfo.maxWorkGroupSize=512) or data_set_size > 32768 (engineInfo.maxWorkGroupSize=1024)
    const bool use_output_buffer = (items_num + 2) > STACK_SIZE;
#else
    const bool use_output_buffer = false;
#endif

    INPUT0_TYPE my_chunk[STACK_SIZE];
    INPUT0_TYPE my_sum = UNIT_VAL_ZERO;
    INPUT0_TYPE my_maximum = -UNIT_VAL_MAX;

    // Read inputs and Get maximum value from data set
    uint input_idx=0;


#if IS_DYNAMIC
    // Case for my_chunk[] when (items_num + 2) <= STACK_SIZE
    if (!use_output_buffer) {
#endif
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
                my_maximum = max(my_maximum, vec_tmp[j]);
            }
        }

        for (; input_idx < items_num; input_idx++)
        {
            BLOCK_TYPE vec_tmp = BLOCK_READ(input, aligned_data_offset + input_idx * get_sub_group_size());
            my_chunk[input_idx] = vec_tmp;
            my_maximum = max(my_maximum, vec_tmp);
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
            my_maximum = max(my_maximum, vec_tmp);
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                INPUT0_TYPE tmp = vec_tmp[j];
                my_chunk[input_idx+j] = tmp;
                my_maximum = max(my_maximum, tmp);
            }
#endif
        }
    }
#endif

    for (; input_idx < items_num; input_idx++)
    {
        INPUT0_TYPE tmp = input[aligned_data_offset + get_sub_group_local_id() + input_idx * get_sub_group_size()];
        my_chunk[input_idx] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

    if (in_data_set_idx < aligned_offset)
    {
        INPUT0_TYPE tmp = input[data_set_offset + in_data_set_idx];
        my_chunk[input_idx++] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

    if (in_data_set_idx < actual_leftovers)
    {
        INPUT0_TYPE tmp = input[leftover_idx];
        my_chunk[input_idx++] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

#if !IS_DYNAMIC
    #if LWS == SUB_GROUP_SIZE
        my_maximum = sub_group_reduce_max(my_maximum);
    #else
        my_maximum = work_group_reduce_max(my_maximum);
    #endif
#else
    my_maximum = work_group_reduce_max(my_maximum);
#endif

    uint num_iters = items_num;

    for (uint j=0; j<num_iters; ++j)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[j] - my_maximum);
        my_sum += tmp;
        my_chunk[j] = tmp;
    }

    if (in_data_set_idx < aligned_offset)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[num_iters] - my_maximum);
        my_sum += tmp;
        my_chunk[num_iters] = tmp;

        num_iters++;
    }

    if (in_data_set_idx < actual_leftovers)
    {
        INPUT0_TYPE tmp = native_exp(my_chunk[num_iters] - my_maximum);
        my_sum += tmp;
        my_chunk[num_iters] = tmp;
    }

#if !IS_DYNAMIC
    #if LWS == SUB_GROUP_SIZE
        my_sum = sub_group_reduce_add(my_sum);
    #else
        my_sum = work_group_reduce_add(my_sum);
    #endif
#else
    my_sum = work_group_reduce_add(my_sum);
#endif

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
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++) {
                ACTIVATION_TYPE dequantized = my_chunk[output_idx + j] / my_sum;
                FUSED_OPS_MAIN;
                vec_tmp[j] = FUSED_OPS_RESULT_MAIN;
            }
            BLOCK_WRITE_OPT(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }

        for (; output_idx<items_num; output_idx++)
        {
            BLOCK_TYPE vec_tmp;
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
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++){
                vec_tmp[j] = ACTIVATION(my_chunk[output_idx + j] / my_sum, ACTIVATION_PARAMS);
            }
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

    if (in_data_set_idx < aligned_offset) {
        output[data_set_offset + in_data_set_idx] = ACTIVATION(my_chunk[output_idx++] / my_sum, ACTIVATION_PARAMS);
    }

    if (in_data_set_idx < actual_leftovers) {
        output[leftover_idx] = ACTIVATION(my_chunk[output_idx++] / my_sum, ACTIVATION_PARAMS);
    }
#endif
#if IS_DYNAMIC
    } else { // Case for output[] directly when (items_num + 2) > STACK_SIZE
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; input_idx < num_iters; input_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp = BLOCK_READ_OPT(input, aligned_data_offset + input_idx * get_sub_group_size());
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++)
            {
                output[aligned_data_offset + get_sub_group_local_id() + (input_idx + j) * get_sub_group_size()] = vec_tmp[j];
                my_maximum = max(my_maximum, vec_tmp[j]);
            }
        }

        for (; input_idx < items_num; input_idx++)
        {
            BLOCK_TYPE vec_tmp = BLOCK_READ(input, aligned_data_offset + input_idx * get_sub_group_size());
            output[aligned_data_offset + get_sub_group_local_id() + input_idx * get_sub_group_size()] = vec_tmp;
            my_maximum = max(my_maximum, vec_tmp);
        }
    }

    for (; input_idx < items_num; input_idx++)
    {
        INPUT0_TYPE tmp = input[aligned_data_offset + get_sub_group_local_id() + input_idx * get_sub_group_size()];
        output[aligned_data_offset + get_sub_group_local_id() + input_idx * get_sub_group_size()] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

    if (in_data_set_idx < aligned_offset)
    {
        INPUT0_TYPE tmp = input[data_set_offset + in_data_set_idx];
        output[data_set_offset + in_data_set_idx] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

    if (in_data_set_idx < actual_leftovers)
    {
        INPUT0_TYPE tmp = input[leftover_idx];
        output[leftover_idx] = tmp;
        my_maximum = max(my_maximum, tmp);
    }

#if !IS_DYNAMIC
    #if LWS == SUB_GROUP_SIZE
        my_maximum = sub_group_reduce_max(my_maximum);
    #else
        my_maximum = work_group_reduce_max(my_maximum);
    #endif
#else
    my_maximum = work_group_reduce_max(my_maximum);
#endif

    const uint num_iters = input_idx;

    for (uint j=0; j<num_iters; ++j) {
        INPUT0_TYPE tmp = native_exp(output[aligned_data_offset + get_sub_group_local_id() + j * get_sub_group_size()] - my_maximum);
        my_sum += tmp;
        output[aligned_data_offset + get_sub_group_local_id() + j * get_sub_group_size()] = tmp;
    }

    if (in_data_set_idx < aligned_offset) {
        INPUT0_TYPE tmp = native_exp(output[data_set_offset + in_data_set_idx] - my_maximum);
        my_sum += tmp;
        output[data_set_offset + in_data_set_idx] = tmp;
    }

    if (in_data_set_idx < actual_leftovers) {
        INPUT0_TYPE tmp = native_exp(output[leftover_idx] - my_maximum);
        my_sum += tmp;
        output[leftover_idx] = tmp;
    }

#if !IS_DYNAMIC
    #if LWS == SUB_GROUP_SIZE
        my_sum = sub_group_reduce_add(my_sum);
    #else
        my_sum = work_group_reduce_add(my_sum);
    #endif
#else
    my_sum = work_group_reduce_add(my_sum);
#endif

    // Write outputs
    uint output_idx = 0;
#if HAS_FUSED_OPS
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; output_idx < num_iters; output_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp;
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++) {
                ACTIVATION_TYPE dequantized = output[aligned_data_offset + get_sub_group_local_id() + (output_idx + j) * get_sub_group_size()] / my_sum;
                FUSED_OPS_MAIN;
                vec_tmp[j] = FUSED_OPS_RESULT_MAIN;
            }
            BLOCK_WRITE_OPT(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }

        for (; output_idx<items_num; output_idx++)
        {
            BLOCK_TYPE vec_tmp;
            ACTIVATION_TYPE dequantized = output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()] / my_sum;
            FUSED_OPS_MAIN;
            vec_tmp = FUSED_OPS_RESULT_MAIN;
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }

    for (; output_idx < items_num; output_idx++)
    {
        ACTIVATION_TYPE dequantized = output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()] / my_sum;
        FUSED_OPS_MAIN;
        output[aligned_data_offset + get_sub_group_local_id() + i * get_sub_group_size()] = FUSED_OPS_RESULT_MAIN;
    }

    if (in_data_set_idx < aligned_offset)
    {
        ACTIVATION_TYPE dequantized = output[data_set_offset + in_data_set_idx] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[data_set_offset + in_data_set_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }

    if (in_data_set_idx < actual_leftovers)
    {
        ACTIVATION_TYPE dequantized = output[leftover_idx] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[leftover_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }
#else
    if (workers_per_data_set > SUB_GROUP_SIZE)
    {
        const uint num_iters = items_num - (items_num % OPT_BLOCK_SIZE);
        for (; output_idx < num_iters; output_idx += OPT_BLOCK_SIZE)
        {
            BLOCK_TYPE_OPT vec_tmp;
            unroll_for (int j = 0; j < OPT_BLOCK_SIZE; j++){
                vec_tmp[j] = ACTIVATION(output[aligned_data_offset + get_sub_group_local_id() + (output_idx + j) * get_sub_group_size()] / my_sum, ACTIVATION_PARAMS);
            }
            BLOCK_WRITE_OPT(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }

        for (; output_idx < items_num; output_idx++)
        {
            BLOCK_TYPE vec_tmp;
            vec_tmp = ACTIVATION(output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()] / my_sum, ACTIVATION_PARAMS);
            BLOCK_WRITE(output, aligned_data_offset + output_idx * get_sub_group_size(), vec_tmp);
        }
    }

    for (; output_idx < items_num; output_idx++)
    {
        output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()]
        = ACTIVATION(output[aligned_data_offset + get_sub_group_local_id() + output_idx * get_sub_group_size()] / my_sum, ACTIVATION_PARAMS);
    }

    if (in_data_set_idx < aligned_offset) {
        output[data_set_offset + in_data_set_idx] = ACTIVATION(output[data_set_offset + in_data_set_idx] / my_sum, ACTIVATION_PARAMS);
    }

    if (in_data_set_idx < actual_leftovers) {
        output[leftover_idx] = ACTIVATION(output[leftover_idx] / my_sum, ACTIVATION_PARAMS);
    }
#endif
    }
#endif
}
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef BLOCK_TYPE
