// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// SliceScatter optimized kernel.
// Uses sub-group block writes for contiguous inner-dimension scatters.
// Condition: step == 1 on all axes, so the slice region is contiguous.
// Dispatched over updates (INPUT1) dimensions with vectorized inner-dim processing.

#include "include/batch_headers/fetch_data.cl"

#define BRING_INTO_RANGE(VAL, MAX) \
    clamp((long)VAL < 0l ? (long)VAL + (long)MAX : (long)VAL, 0l, (long)MAX - 1l);

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

#define VEC_SIZE    SLICE_SCATTER_VEC_SIZE
#define SUBGROUP_SIZE get_sub_group_size()

#if VEC_SIZE == 8
    #define VEC_TYPE MAKE_VECTOR_TYPE(INPUT1_TYPE, 8)
    #define VLOAD(offset, ptr) vload8(offset, ptr)
    #define VSTORE(val, offset, ptr) vstore8(val, offset, ptr)
#elif VEC_SIZE == 4
    #define VEC_TYPE MAKE_VECTOR_TYPE(INPUT1_TYPE, 4)
    #define VLOAD(offset, ptr) vload4(offset, ptr)
    #define VSTORE(val, offset, ptr) vstore4(val, offset, ptr)
#else
    #define VEC_TYPE INPUT1_TYPE
    #define VLOAD(offset, ptr) (ptr[offset])
    #define VSTORE(val, offset, ptr) (ptr[offset] = val)
#endif

REQD_SUB_GROUP_SIZE(SIMD_SIZE)
KERNEL(slice_scatter_opt)(OPTIONAL_SHAPE_INFO_ARG
                          const __global INPUT0_TYPE* restrict data,
                          const __global INPUT1_TYPE* restrict updates,
                          START_BUFFER
                          STEP_BUFFER
                          AXES_BUFFER
                          __global OUTPUT_TYPE* restrict output)
{
    LOAD_BUFFER(START, start_buff);
    LOAD_BUFFER(AXES, axes_buff);

    // Compute start offsets per dimension (step is always 1 for opt kernel)
    long slice_start[INPUT0_DIMS];
    unroll_for(int i = 0; i < INPUT0_DIMS; ++i) {
        slice_start[i] = 0;
    }
    unroll_for(int i = 0; i < AXES_BUFFER_SIZE; ++i) {
        const long axis = axes_buff[i];
        slice_start[axis] = start_buff[i];
    }

    // Global IDs: dim0=batch, dim1=feature, dim2=spatial(Z*Y*X or Y*X)
    const long upd_dim0 = get_global_id(0);  // batch
    const long upd_dim1 = get_global_id(1);  // feature
    const long slice_begin_dim0 = BRING_INTO_RANGE(slice_start[0], INPUT0_BATCH_NUM);
    const long slice_begin_dim1 = BRING_INTO_RANGE(slice_start[1], INPUT0_FEATURE_NUM);

#if INPUT0_DIMS <= 4
    const long slice_begin_dim2 = BRING_INTO_RANGE(slice_start[2], INPUT0_SIZE_Y);
    const long slice_begin_dim3 = BRING_INTO_RANGE(slice_start[3], INPUT0_SIZE_X);

    // Each work-item in the sub-group processes VEC_SIZE elements along X
    const long upd_dim23_base = get_global_id(2);
    const long total_spatial = (long)INPUT1_SIZE_Y * (long)INPUT1_SIZE_X;

    // Bounds check - skip excess work-items
    if (upd_dim23_base >= total_spatial / VEC_SIZE)
        return;

    const long linear_pos = upd_dim23_base * VEC_SIZE;
    const long upd_dim2 = linear_pos / INPUT1_SIZE_X;
    const long upd_dim3_start = linear_pos % INPUT1_SIZE_X;

    // Compute base indices for vectorized access
    const long updates_base = INPUT1_GET_INDEX(upd_dim0, upd_dim1, upd_dim2, upd_dim3_start);
    const long output_base = OUTPUT_GET_INDEX(
        slice_begin_dim0 + upd_dim0,
        slice_begin_dim1 + upd_dim1,
        slice_begin_dim2 + upd_dim2,
        slice_begin_dim3 + upd_dim3_start);

#if VEC_SIZE > 1
    // Check if remaining elements on this row are enough for full vector
    if (upd_dim3_start + VEC_SIZE <= INPUT1_SIZE_X) {
        VEC_TYPE val = VLOAD(0, &updates[updates_base]);
        VSTORE(val, 0, &output[output_base]);
    } else {
        // Scalar fallback for tail elements
        for (long i = 0; i < VEC_SIZE && upd_dim3_start + i < INPUT1_SIZE_X; ++i) {
            output[output_base + i] = updates[updates_base + i];
        }
    }
#else
    output[output_base] = updates[updates_base];
#endif

#elif INPUT0_DIMS == 5
    const long slice_begin_dim2 = BRING_INTO_RANGE(slice_start[2], INPUT0_SIZE_Z);
    const long slice_begin_dim3 = BRING_INTO_RANGE(slice_start[3], INPUT0_SIZE_Y);
    const long slice_begin_dim4 = BRING_INTO_RANGE(slice_start[4], INPUT0_SIZE_X);

    const long upd_dim234_base = get_global_id(2);
    const long total_spatial = (long)INPUT1_SIZE_Z * (long)INPUT1_SIZE_Y * (long)INPUT1_SIZE_X;

    if (upd_dim234_base >= total_spatial / VEC_SIZE)
        return;

    const long linear_pos = upd_dim234_base * VEC_SIZE;
    const long upd_dim4_start = linear_pos % INPUT1_SIZE_X;
    const long upd_dim34 = linear_pos / INPUT1_SIZE_X;
    const long upd_dim3 = upd_dim34 % INPUT1_SIZE_Y;
    const long upd_dim2 = upd_dim34 / INPUT1_SIZE_Y;

    const long updates_base = INPUT1_GET_INDEX(upd_dim0, upd_dim1, upd_dim2, upd_dim3, upd_dim4_start);
    const long output_base = OUTPUT_GET_INDEX(
        slice_begin_dim0 + upd_dim0,
        slice_begin_dim1 + upd_dim1,
        slice_begin_dim2 + upd_dim2,
        slice_begin_dim3 + upd_dim3,
        slice_begin_dim4 + upd_dim4_start);

#if VEC_SIZE > 1
    if (upd_dim4_start + VEC_SIZE <= INPUT1_SIZE_X) {
        VEC_TYPE val = VLOAD(0, &updates[updates_base]);
        VSTORE(val, 0, &output[output_base]);
    } else {
        for (long i = 0; i < VEC_SIZE && upd_dim4_start + i < INPUT1_SIZE_X; ++i) {
            output[output_base + i] = updates[updates_base + i];
        }
    }
#else
    output[output_base] = updates[updates_base];
#endif

#endif
}

#undef LOAD_BUFFER
#undef BRING_INTO_RANGE
#undef VEC_SIZE
#undef VEC_TYPE
#undef VLOAD
#undef VSTORE
