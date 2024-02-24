// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BRING_INTO_RANGE(VAL, MAX) \
    clamp((long)VAL < 0l ? (long)VAL + (long)MAX : (long)VAL, 0l, (long)MAX-1l);

KERNEL(slice_ref)(OPTIONAL_SHAPE_INFO_ARG 
                  const __global INPUT0_TYPE* restrict input,
                  SLICE_BEGIN_BUFFER
                  SLICE_STEP_BUFFER
                  SLICE_AXES_BUFFER
                  __global OUTPUT_TYPE* restrict output)
{
    long axes[INPUT0_DIMS];
    axes[0] = SLICE_AXES_DIM0;
    axes[1] = SLICE_AXES_DIM1;
    axes[2] = SLICE_AXES_DIM2;
    axes[3] = SLICE_AXES_DIM3;
#if INPUT0_DIMS == 5
    axes[4] = SLICE_AXES_DIM4;
#endif

    long slice_step_init[INPUT0_DIMS];
    slice_step_init[0] = SLICE_STEP_DIM0;
    slice_step_init[1] = SLICE_STEP_DIM1;
    slice_step_init[2] = SLICE_STEP_DIM2;
    slice_step_init[3] = SLICE_STEP_DIM3;
#if INPUT0_DIMS == 5
    slice_step_init[4] = SLICE_STEP_DIM4;
#endif

    long slice_start_init[INPUT0_DIMS];
    slice_start_init[0] = SLICE_BEGIN_DIM0;
    slice_start_init[1] = SLICE_BEGIN_DIM1;
    slice_start_init[2] = SLICE_BEGIN_DIM2;
    slice_start_init[3] = SLICE_BEGIN_DIM3;
#if INPUT0_DIMS == 5
    slice_start_init[4] = SLICE_BEGIN_DIM4;
#endif

    long slice_step[INPUT0_DIMS];
    long slice_start[INPUT0_DIMS];
    #pragma unroll
    for(int i = 0; i < INPUT0_DIMS; ++i) {
        slice_step[i] = 1;
        slice_start[i] = 0;
    }

    #pragma unroll
    for(int i = 0; i < SLICE_AXES_BUFFER_SIZE; ++i) {
        const long axis = axes[i];
        slice_step[axis] = slice_step_init[i];
        slice_start[axis] = slice_start_init[i];
    }

    const long output_dim0 = get_global_id(0);
    const long output_dim1 = get_global_id(1);
    const long slice_begin_dim0 = BRING_INTO_RANGE(slice_start[0], INPUT0_BATCH_NUM);
    const long slice_begin_dim1 = BRING_INTO_RANGE(slice_start[1], INPUT0_FEATURE_NUM);

#if INPUT0_DIMS <= 4
    const long slice_begin_dim2 = BRING_INTO_RANGE(slice_start[2], INPUT0_SIZE_Y);
    const long slice_begin_dim3 = BRING_INTO_RANGE(slice_start[3], INPUT0_SIZE_X);
    const long output_dim23 = get_global_id(2);
    const long output_dim2 = output_dim23 / OUTPUT_SIZE_X;
    const long output_dim3 = output_dim23 % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(output_dim0, output_dim1, output_dim2, output_dim3);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_dim0 + output_dim0 * slice_step[0],
        slice_begin_dim1 + output_dim1 * slice_step[1],
        slice_begin_dim2 + output_dim2 * slice_step[2],
        slice_begin_dim3 + output_dim3 * slice_step[3]);
#elif INPUT0_DIMS == 5
    const long slice_begin_dim2 = BRING_INTO_RANGE(slice_start[2], INPUT0_SIZE_Z);
    const long slice_begin_dim3 = BRING_INTO_RANGE(slice_start[3], INPUT0_SIZE_Y);
    const long slice_begin_dim4 = BRING_INTO_RANGE(slice_start[4], INPUT0_SIZE_X);
    const long output_dim234 = get_global_id(2);
    const long output_dim34 = output_dim234 % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long output_dim2 = output_dim234 / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long output_dim3 = output_dim34 / OUTPUT_SIZE_X;
    const long output_dim4 = output_dim34 % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(output_dim0, output_dim1, output_dim2, output_dim3, output_dim4);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_dim0 + output_dim0 * slice_step[0],
        slice_begin_dim1 + output_dim1 * slice_step[1],
        slice_begin_dim2 + output_dim2 * slice_step[2],
        slice_begin_dim3 + output_dim3 * slice_step[3],
        slice_begin_dim4 + output_dim4 * slice_step[4]);
#endif

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}

#undef BRING_INTO_RANGE;