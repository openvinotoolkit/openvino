// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BRING_INTO_RANGE(VAL, MAX) clamp((long)VAL < 0l ? (long)VAL + (long)MAX : (long)VAL, 0l, (long)MAX-1l);

KERNEL(slice_ref)(OPTIONAL_SHAPE_INFO_ARG 
                  const __global INPUT0_TYPE* input,
                  SLICE_BEGIN_BUFFER
                  SLICE_STEP_BUFFER
                  __global OUTPUT_TYPE* output)
{
    //printf("OUTPUT_SIZE_X: %i\n", OUTPUT_SIZE_X);
    //printf("SLICE_BEGIN_X: %i\n", SLICE_BEGIN_X);

    const long batch = get_global_id(0);
    const long feature = get_global_id(1);

    const long slice_begin_batch =   BRING_INTO_RANGE(SLICE_BEGIN_BATCH, INPUT0_BATCH_NUM);
    const long slice_begin_feature = BRING_INTO_RANGE(SLICE_BEGIN_FEATURE, INPUT0_FEATURE_NUM);
    const long slice_begin_y =       BRING_INTO_RANGE(SLICE_BEGIN_Y, INPUT0_SIZE_Y);
    const long slice_begin_x =       BRING_INTO_RANGE(SLICE_BEGIN_X, INPUT0_SIZE_X);

    // printf("slice_begin_x: %i\n", slice_begin_x);
    // printf("SLICE_STEP_X: %i\n", SLICE_STEP_X);

    // printf("OUTPUT_BATCH_NUM: %i\n", OUTPUT_BATCH_NUM);
    // printf("OUTPUT_FEATURE_NUM: %i\n", OUTPUT_FEATURE_NUM);
    // printf("OUTPUT_SIZE_Y: %i\n", OUTPUT_SIZE_Y);
    // printf("OUTPUT_SIZE_X: %i\n", OUTPUT_SIZE_X);

    // printf("SLICE_STEP_BATCH: %i\n", SLICE_STEP_BATCH);
    // printf("SLICE_STEP_FEATURE: %i\n", SLICE_STEP_FEATURE);
    // printf("SLICE_STEP_Y: %i\n", SLICE_STEP_Y);
    // printf("SLICE_STEP_X: %i\n", SLICE_STEP_X);


#if INPUT0_DIMS <= 4
    const long xy = get_global_id(2);
    const long y = xy / OUTPUT_SIZE_X;
    const long x = xy % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_batch + batch * SLICE_STEP_BATCH,
        slice_begin_feature + feature * SLICE_STEP_FEATURE,
        slice_begin_y + y * SLICE_STEP_Y,
        slice_begin_x + x * SLICE_STEP_X);
#elif INPUT0_DIMS == 5
    const long slice_begin_z = BRING_INTO_RANGE(SLICE_BEGIN_Z, INPUT0_SIZE_Z);
    const long xyz = get_global_id(2);
    const long yx = xyz % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long z = xyz / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long y = yx / OUTPUT_SIZE_X;
    const long x = yx % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_batch + batch * SLICE_STEP_BATCH,
        slice_begin_feature + feature * SLICE_STEP_FEATURE,
        slice_begin_z + z * SLICE_STEP_Z,
        slice_begin_y + y * SLICE_STEP_Y,
        slice_begin_x + x * SLICE_STEP_X);
#endif
    //printf("input_index: %i, input[input_index]: %i\n",input_index, input[input_index]);
    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}

#undef BRING_INTO_RANGE;