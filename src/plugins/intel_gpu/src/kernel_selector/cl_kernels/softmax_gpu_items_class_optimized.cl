// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define DATA_PER_WORKITEM ( (INPUT0_CLASS_NUM + (WORKITEMS_PER_CLASSES - 1) ) / WORKITEMS_PER_CLASSES)
#define FULL_ITERATIONS_NUM (INPUT0_CLASS_NUM / WORKITEMS_PER_CLASSES)

#if FULL_ITERATIONS_NUM / 8 > 0
#define BLOCK_SIZE 8
#elif FULL_ITERATIONS_NUM / 4 > 0
#define BLOCK_SIZE 4
#elif FULL_ITERATIONS_NUM / 2 > 0
#define BLOCK_SIZE 2
#else
#define BLOCK_SIZE 1
#endif

#if BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define BLOCK_TYPE INPUT0_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, BLOCK_SIZE)(ptr, offset, val)
#define BLOCK_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, BLOCK_SIZE)
#endif

#define SUB_GROUP_SIZE 16

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(softmax_items_class_optimized)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if INPUT0_DIMS == 5
    const uint other0 = (uint)get_group_id(0) % INPUT0_OTHER0_SIZE;
    const uint other2 = (uint)get_group_id(0) / INPUT0_OTHER0_SIZE;
#else
    const uint other0 = get_group_id(0);
    const uint other2 = 0;
#endif
    const uint other1 = get_group_id(1);
    const uint other3  = get_group_id(2);
    const uint simd_lane = get_sub_group_local_id();

    const uint in_depth_offset  = other3*INPUT0_OTHER3_PITCH + other2*INPUT0_OTHER2_PITCH + other1*INPUT0_OTHER1_PITCH + other0*INPUT0_OTHER0_PITCH + INPUT0_OFFSET;
    const uint out_depth_offset = other3*OUTPUT_OTHER3_PITCH + other2*OUTPUT_OTHER2_PITCH + other1*OUTPUT_OTHER1_PITCH + other0*OUTPUT_OTHER0_PITCH + OUTPUT_OFFSET;

    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
    ACCUMULATOR_TYPE data[DATA_PER_WORKITEM];

    // PART 1. Calculate MAX value
    uint input_idx = in_depth_offset;

    uint cls = 0;
#if FULL_ITERATIONS_NUM >= SUB_GROUP_SIZE && IS_SUBGROUP_BLOCK_IO_ENABLED
    for (; cls < FULL_ITERATIONS_NUM - (FULL_ITERATIONS_NUM % BLOCK_SIZE); cls += BLOCK_SIZE)
    {
        BLOCK_TYPE vec = BLOCK_READ(input, input_idx);
#if BLOCK_SIZE > 1
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            ACCUMULATOR_TYPE in = vec[i];
            max_value = max(max_value, in);
            data[cls + i] = in;
        }
#else
        ACCUMULATOR_TYPE in = vec;
        max_value = max(max_value, in);
        data[cls] = in;
#endif
        input_idx += BLOCK_SIZE * WORKITEMS_PER_CLASSES * INPUT0_CLASS_PITCH;
    }
#endif
    input_idx += simd_lane * INPUT0_CLASS_PITCH;
    for (; cls < FULL_ITERATIONS_NUM; cls++)
    {
        ACCUMULATOR_TYPE in = input[input_idx];
        max_value = max(max_value, in);
        data[cls] = in;
        input_idx += WORKITEMS_PER_CLASSES*INPUT0_CLASS_PITCH;
    }

    if(simd_lane < LEFTOVERS)
    {
        ACCUMULATOR_TYPE in = input[input_idx];
        max_value = max(max_value, in);
        data[DATA_PER_WORKITEM-1] = in;
    }
    max_value = sub_group_reduce_max(max_value);

    // PART 2. Calculate DENOMINATOR
    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
    for (uint cls = 0; cls < FULL_ITERATIONS_NUM; cls++)
    {
// This is a temporary solution for unresolved problem when ocl kernels compilation step doesn't produce actual binaries
// for current kernel but driver doesn't report any errors (JIRA 32211)
#if HAS_DRIVER_PROBLEMS
        data[cls] = data[cls] == max_value ? 1.0 : native_exp(data[cls] - max_value);
#else
        data[cls] = native_exp(data[cls] - max_value);
#endif
        denominator += data[cls];
    }
    if(simd_lane < LEFTOVERS)
    {
// This is a temporary solution for unresolved problem when ocl kernels compilation step doesn't produce actual binaries
// for current kernel but driver doesn't report any errors (JIRA 32211)
#if HAS_DRIVER_PROBLEMS
        data[DATA_PER_WORKITEM-1] = data[DATA_PER_WORKITEM-1] == max_value ? 1.0 : native_exp(data[DATA_PER_WORKITEM-1] - max_value);
#else
        data[DATA_PER_WORKITEM-1] = native_exp(data[DATA_PER_WORKITEM-1] - max_value);
#endif
        denominator += data[DATA_PER_WORKITEM-1];
    }

    denominator = sub_group_reduce_add(denominator);

    // PART 3. Write out results
    uint output_idx = out_depth_offset;
    cls = 0;
#if FULL_ITERATIONS_NUM >= SUB_GROUP_SIZE && IS_SUBGROUP_BLOCK_IO_ENABLED
    for (; cls < FULL_ITERATIONS_NUM - (FULL_ITERATIONS_NUM % BLOCK_SIZE); cls += BLOCK_SIZE)
    {
        BLOCK_TYPE vec;
#if BLOCK_SIZE > 1
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            const ACCUMULATOR_TYPE res = data[cls + i] / denominator;
#if HAS_FUSED_OPS
            FUSED_OPS;
            vec[i] = FUSED_OPS_RESULT;
#else
            vec[i] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
        }
#else
        const ACCUMULATOR_TYPE res = data[cls] / denominator;
#if HAS_FUSED_OPS
        FUSED_OPS;
        vec = FUSED_OPS_RESULT;
#else
        vec = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
#endif
        BLOCK_WRITE(output, output_idx, vec);
        output_idx += BLOCK_SIZE * WORKITEMS_PER_CLASSES;
    }
#endif
    output_idx += simd_lane * OUTPUT_CLASS_PITCH;
    for (; cls < FULL_ITERATIONS_NUM; cls++)
    {
        const ACCUMULATOR_TYPE res = data[cls] / denominator;
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
        output_idx += WORKITEMS_PER_CLASSES * OUTPUT_CLASS_PITCH;
    }
    if(simd_lane < LEFTOVERS)
    {
        const ACCUMULATOR_TYPE res = data[DATA_PER_WORKITEM-1] / denominator;
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
    }
}

#undef FULL_ITERATIONS_NUM
#undef DATA_PER_WORKITEM
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef BLOCK_TYPE
#undef SUB_GROUP_SIZE
