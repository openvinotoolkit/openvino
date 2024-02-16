// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BRING_INTO_RANGE(VAL, MAX) \
    clamp((long)VAL < 0l ? (long)VAL + (long)MAX : (long)VAL, 0l, (long)MAX-1l);

KERNEL(slice_ref)(OPTIONAL_SHAPE_INFO_ARG 
                  const __global INPUT0_TYPE* input,
                  SLICE_BEGIN_BUFFER
                  SLICE_STEP_BUFFER
                  __global OUTPUT_TYPE* output)
{
    const long output_dim0 = get_global_id(0);
    const long output_dim1 = get_global_id(1);
    const long slice_begin_dim0 = BRING_INTO_RANGE(SLICE_BEGIN_DIM0, INPUT0_BATCH_NUM);
    const long slice_begin_dim1 = BRING_INTO_RANGE(SLICE_BEGIN_DIM1, INPUT0_FEATURE_NUM);

#if INPUT0_DIMS <= 4
    const long slice_begin_dim2 = BRING_INTO_RANGE(SLICE_BEGIN_DIM2, INPUT0_SIZE_Y);
    const long slice_begin_dim3 = BRING_INTO_RANGE(SLICE_BEGIN_DIM3, INPUT0_SIZE_X);
    const long output_dim23 = get_global_id(2);
    const long output_dim2 = output_dim23 / OUTPUT_SIZE_X;
    const long output_dim3 = output_dim23 % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(output_dim0, output_dim1, output_dim2, output_dim3);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_dim0 + output_dim0 * SLICE_STEP_DIM0,
        slice_begin_dim1 + output_dim1 * SLICE_STEP_DIM1,
        slice_begin_dim2 + output_dim2 * SLICE_STEP_DIM2,
        slice_begin_dim3 + output_dim3 * SLICE_STEP_DIM3);
#elif INPUT0_DIMS == 5
    const long slice_begin_dim2 = BRING_INTO_RANGE(SLICE_BEGIN_DIM2, INPUT0_SIZE_Z);
    const long slice_begin_dim3 = BRING_INTO_RANGE(SLICE_BEGIN_DIM3, INPUT0_SIZE_Y);
    const long slice_begin_dim4 = BRING_INTO_RANGE(SLICE_BEGIN_DIM4, INPUT0_SIZE_X);
    const long output_dim234 = get_global_id(2);
    const long output_dim34 = output_dim234 % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long output_dim2 = output_dim234 / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const long output_dim3 = output_dim34 / OUTPUT_SIZE_X;
    const long output_dim4 = output_dim34 % OUTPUT_SIZE_X;
    const long output_index = OUTPUT_GET_INDEX(output_dim0, output_dim1, output_dim2, output_dim3, output_dim4);
    const long input_index = INPUT0_GET_INDEX(
        slice_begin_dim0 + output_dim0 * SLICE_STEP_DIM0,
        slice_begin_dim1 + output_dim1 * SLICE_STEP_DIM1,
        slice_begin_dim2 + output_dim2 * SLICE_STEP_DIM2,
        slice_begin_dim3 + output_dim3 * SLICE_STEP_DIM3,
        slice_begin_dim4 + output_dim4 * SLICE_STEP_DIM4);
#endif

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}

#undef BRING_INTO_RANGE;