// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// Memcpy-style broadcast kernel: each work-group copies one output row (X dimension)
// from input to output. Outer dimensions (B, F, W, Z, Y) are broadcast via modulo.
// This avoids per-element get_idx_pos() computation — address is computed once per row.

KERNEL(broadcast_gpu_memcpy)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
)
{
    const uint row_id = get_global_id(1);
    const uint lid = get_global_id(0);

    // Decode row_id into (b, f, w, z, y) output coordinates
#if OUTPUT_DIMS == 6
    const uint out_y = row_id % OUTPUT_SIZE_Y;
    const uint out_z = (row_id / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const uint out_w = (row_id / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z) % OUTPUT_SIZE_W;
    const uint out_f = (row_id / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_SIZE_W) % OUTPUT_FEATURE_NUM;
    const uint out_b = row_id / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_SIZE_W / OUTPUT_FEATURE_NUM;
#elif OUTPUT_DIMS == 5
    const uint out_y = row_id % OUTPUT_SIZE_Y;
    const uint out_z = (row_id / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const uint out_f = (row_id / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z) % OUTPUT_FEATURE_NUM;
    const uint out_b = row_id / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_FEATURE_NUM;
    const uint out_w = 0;
#else
    const uint out_y = row_id % OUTPUT_SIZE_Y;
    const uint out_f = (row_id / OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM;
    const uint out_b = row_id / OUTPUT_SIZE_Y / OUTPUT_FEATURE_NUM;
    const uint out_z = 0;
    const uint out_w = 0;
#endif

    // Compute input row coordinates (broadcast via modulo for outer dims only)
    const uint in_b = out_b % INPUT0_BATCH_NUM;
    const uint in_f = out_f % INPUT0_FEATURE_NUM;
#if OUTPUT_DIMS >= 5
    const uint in_z = out_z % INPUT0_SIZE_Z;
#else
    const uint in_z = 0;
#endif
#if OUTPUT_DIMS >= 6
    const uint in_w = out_w % INPUT0_SIZE_W;
#else
    const uint in_w = 0;
#endif
    const uint in_y = out_y % INPUT0_SIZE_Y;

    // Base addresses for this row (computed once per row, not per element)
#if OUTPUT_DIMS == 6
    const uint out_row_base = OUTPUT_GET_INDEX(out_b, out_f, out_w, out_z, out_y, 0);
    const uint in_row_base = INPUT0_GET_INDEX(in_b, in_f, in_w, in_z, in_y, 0);
#elif OUTPUT_DIMS == 5
    const uint out_row_base = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, 0);
    const uint in_row_base = INPUT0_GET_INDEX(in_b, in_f, in_z, in_y, 0);
#else
    const uint out_row_base = OUTPUT_GET_INDEX(out_b, out_f, out_y, 0);
    const uint in_row_base = INPUT0_GET_INDEX(in_b, in_f, in_y, 0);
#endif

    // Linear copy of X dimension: each work-item handles consecutive chunks
    const uint x_per_wi = MEMCPY_VEC_SIZE;
    const uint x_start = lid * x_per_wi;

    if (x_start + x_per_wi <= OUTPUT_SIZE_X) {
        // Full vector load/store
        MAKE_VECTOR_TYPE(INPUT0_TYPE, MEMCPY_VEC_SIZE) in_vec = CAT(vload, MEMCPY_VEC_SIZE)(0, &input[in_row_base + x_start]);
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, MEMCPY_VEC_SIZE) out_vec;
        out_vec.s0 = TO_OUTPUT_TYPE(in_vec.s0);
        out_vec.s1 = TO_OUTPUT_TYPE(in_vec.s1);
        out_vec.s2 = TO_OUTPUT_TYPE(in_vec.s2);
        out_vec.s3 = TO_OUTPUT_TYPE(in_vec.s3);
        out_vec.s4 = TO_OUTPUT_TYPE(in_vec.s4);
        out_vec.s5 = TO_OUTPUT_TYPE(in_vec.s5);
        out_vec.s6 = TO_OUTPUT_TYPE(in_vec.s6);
        out_vec.s7 = TO_OUTPUT_TYPE(in_vec.s7);
        CAT(vstore, MEMCPY_VEC_SIZE)(out_vec, 0, &output[out_row_base + x_start]);
    } else if (x_start < OUTPUT_SIZE_X) {
        // Scalar tail
        for (uint x = x_start; x < OUTPUT_SIZE_X; x++) {
            output[out_row_base + x] = TO_OUTPUT_TYPE(input[in_row_base + x]);
        }
    }
}
