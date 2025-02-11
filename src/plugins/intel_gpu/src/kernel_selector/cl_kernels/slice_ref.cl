// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BRING_INTO_RANGE(VAL, MAX) \
    clamp((long)VAL < 0l ? (long)VAL + (long)MAX : (long)VAL, 0l, (long)MAX-1l);

#if INPUT0_DIMS < 5
#define LOAD_BUFFER(in_prefix, out_name)  \
    long out_name[INPUT0_DIMS];           \
    out_name[0] = in_prefix##_VAL0;       \
    out_name[1] = in_prefix##_VAL1;       \
    out_name[2] = in_prefix##_VAL2;       \
    out_name[3] = in_prefix##_VAL3;
#else
#define LOAD_BUFFER(in_prefix, out_name)  \
    long out_name[INPUT0_DIMS];           \
    out_name[0] = in_prefix##_VAL0;       \
    out_name[1] = in_prefix##_VAL1;       \
    out_name[2] = in_prefix##_VAL2;       \
    out_name[3] = in_prefix##_VAL3;       \
    out_name[4] = in_prefix##_VAL4;
#endif

KERNEL(slice_ref)(OPTIONAL_SHAPE_INFO_ARG
                  const __global INPUT0_TYPE* restrict input,
                  START_BUFFER
                  STEP_BUFFER
                  AXES_BUFFER
                  __global OUTPUT_TYPE* restrict output)
{
    LOAD_BUFFER(START, start_buff);
    LOAD_BUFFER(STEP, step_buff);
    LOAD_BUFFER(AXES, axes_buff);

    long slice_step[INPUT0_DIMS];
    long slice_start[INPUT0_DIMS];

    unroll_for(int i = 0; i < INPUT0_DIMS; ++i) {
        slice_step[i] = 1;
        slice_start[i] = 0;
    }

    unroll_for(int i = 0; i < AXES_BUFFER_SIZE; ++i) {
        const long axis = axes_buff[i];
        slice_step[axis] = step_buff[i];
        slice_start[axis] = start_buff[i];
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

#undef LOAD_BUFFER;
#undef BRING_INTO_RANGE;
