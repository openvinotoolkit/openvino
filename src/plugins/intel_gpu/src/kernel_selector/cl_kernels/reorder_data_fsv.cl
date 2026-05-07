// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Reorder kernel for single-blocked fsv format conversions.
// Converts between b_fs_{yx,zyx}_fsv{4,8,16,32} formats with different fsv sizes.
// Both input and output are blocked only on the feature dimension.

#include "include/batch_headers/fetch_data.cl"

#ifdef FSV_VECTORIZED
    #define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
    #define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
    #define CONVERT_OUT_VEC CAT(convert_, OUTPUT_VEC_TYPE)
    #define VLOAD_VEC       CAT(vload, VEC_SIZE)
    #define VSTORE_VEC      CAT(vstore, VEC_SIZE)
#endif

KERNEL (reorder_data_fsv)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    const uint x = (uint)get_global_id(0);
    const uint yz = (uint)get_global_id(1);
    const uint b_fs = (uint)get_global_id(2);

    const uint fs_out = b_fs % OUT_FEATURE_SLICE_NUM;
    const uint b = b_fs / OUT_FEATURE_SLICE_NUM;

#ifdef INPUT0_DIMS_5
    const uint z = yz % INPUT0_SIZE_Z;
    const uint y = yz / INPUT0_SIZE_Z;
#else
    const uint y = yz;
    const uint z = 0;
#endif

    const uint f_base = fs_out * OUT_FSV;

    const uint in_x_pitch = IN_FSV;
    const uint in_y_pitch = in_x_pitch * INPUT0_SIZE_X;
#ifdef INPUT0_DIMS_5
    const uint in_z_pitch = in_y_pitch * INPUT0_SIZE_Y;
    const uint in_fs_pitch = in_z_pitch * INPUT0_SIZE_Z;
#else
    const uint in_fs_pitch = in_y_pitch * INPUT0_SIZE_Y;
#endif
    const uint in_b_pitch = in_fs_pitch * IN_FEATURE_SLICE_NUM;

    const uint out_x_pitch = OUT_FSV;
    const uint out_y_pitch = out_x_pitch * OUTPUT_SIZE_X;
#ifdef INPUT0_DIMS_5
    const uint out_z_pitch = out_y_pitch * OUTPUT_SIZE_Y;
    const uint out_fs_pitch = out_z_pitch * OUTPUT_SIZE_Z;
#else
    const uint out_fs_pitch = out_y_pitch * OUTPUT_SIZE_Y;
#endif
    const uint out_b_pitch = out_fs_pitch * OUT_FEATURE_SLICE_NUM;

#ifdef INPUT0_DIMS_5
    const uint out_offset_base = b * out_b_pitch + fs_out * out_fs_pitch + z * out_z_pitch + y * out_y_pitch + x * OUT_FSV;
    const uint in_spatial_base = b * in_b_pitch + z * in_z_pitch + y * in_y_pitch + x * IN_FSV;
#else
    const uint out_offset_base = b * out_b_pitch + fs_out * out_fs_pitch + y * out_y_pitch + x * OUT_FSV;
    const uint in_spatial_base = b * in_b_pitch + y * in_y_pitch + x * IN_FSV;
#endif

#ifdef FSV_VECTORIZED
    // Vectorized path: read/write VEC_SIZE elements at a time.
    // RATIO = max(IN_FSV, OUT_FSV) / min(IN_FSV, OUT_FSV)
    #if (OUT_FSV > IN_FSV)
        // Upscale: e.g. fsv16 -> fsv32. Read RATIO input slices into one output slice.
        unroll_for (uint r = 0; r < RATIO; ++r) {
            const uint fs_in = fs_out * RATIO + r;
            const uint f_chunk = f_base + r * VEC_SIZE;
            if (f_chunk < INPUT0_FEATURE_NUM) {
                INPUT_VEC_TYPE v = VLOAD_VEC(0, input + in_spatial_base + fs_in * in_fs_pitch);
                VSTORE_VEC(CONVERT_OUT_VEC(v), 0, output + out_offset_base + r * VEC_SIZE);
            } else {
                OUTPUT_VEC_TYPE zero = (OUTPUT_VEC_TYPE)(0);
                VSTORE_VEC(zero, 0, output + out_offset_base + r * VEC_SIZE);
            }
        }
    #else
        // Downscale: e.g. fsv32 -> fsv16. Read one chunk from a larger input slice.
        const uint fs_in = fs_out / RATIO;
        const uint sub = fs_out % RATIO;
        if (f_base < INPUT0_FEATURE_NUM) {
            INPUT_VEC_TYPE v = VLOAD_VEC(0, input + in_spatial_base + fs_in * in_fs_pitch + sub * VEC_SIZE);
            VSTORE_VEC(CONVERT_OUT_VEC(v), 0, output + out_offset_base);
        } else {
            OUTPUT_VEC_TYPE zero = (OUTPUT_VEC_TYPE)(0);
            VSTORE_VEC(zero, 0, output + out_offset_base);
        }
    #endif
#else
    // Scalar fallback for small fsv sizes (fsv4, fsv8) or non-power-of-2 ratios.
    for (uint i = 0; i < OUT_FSV; ++i) {
        const uint f = f_base + i;
        if (f < INPUT0_FEATURE_NUM) {
            const uint fs_in = f / IN_FSV;
            const uint fsv_in = f % IN_FSV;
            const uint in_offset = in_spatial_base + fs_in * in_fs_pitch + fsv_in;
            output[out_offset_base + i] = ACTIVATION(TO_OUTPUT_TYPE(input[in_offset]), ACTIVATION_PARAMS);
        } else {
            output[out_offset_base + i] = TO_OUTPUT_TYPE(0);
        }
    }
#endif
}
