// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef INPUT0_TYPE data_t;
typedef INPUT1_TYPE grid_t;
typedef OUTPUT_TYPE output_t;

typedef INPUT0_TYPE data_et;
typedef float grid_et;
typedef OUTPUT_TYPE output_et;

#if defined(ALIGN_CORNERS)
#    define rescale_align FUNC(denormalize)
inline grid_et rescale_align(const grid_et value, const size_t range) {
    return (value + 1) * ((grid_et)(range)-1) / 2;
}
#else
#    define rescale_noalign FUNC(denormalize)
inline grid_et rescale_noalign(const grid_et value, const size_t range) {
    return ((value + 1) * (grid_et)(range)-1) / 2;
}
#endif
#define denormalize FUNC_CALL(denormalize)

inline const bool FUNC(is_between)(int val, int min, int max) {
    return (val >= min) && (val < max);
}
#define is_between FUNC_CALL(is_between)

// __attribute__((reqd_work_group_size(1,256,1)))
KERNEL(grid_sample_opt_bilinear_zeros)(const __global data_t* restrict data,
                                       const __global grid_t* restrict grid,
                                       __global output_t* restrict output
#if HAS_FUSED_OPS_DECLS
                                       ,
                                       FUSED_OPS_DECLS
#endif
) {

#if !defined(INTERPOLATION_MODE_BILINEAR)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support bilinear interppolation mode.
#endif

#if !defined(PADDING_MODE_ZEROS)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support zeros padding mode.
#endif

    const int n = get_global_id(0);
    const int GRID_OFFSET_FOR_THIS_BLOCK = GRID_ITEMS_PER_BLOCK * 2 * get_group_id(1);
    const int BLOCK_SIZE = get_local_size(1);
    const grid_t* restrict grid_for_this_block = grid + GRID_OFFSET_FOR_THIS_BLOCK;
    const int GRID_ITEMS_FOR_THIS_BLOCK =
        min(OUTPUT_SIZE_Y * OUTPUT_SIZE_X * 2 - GRID_OFFSET_FOR_THIS_BLOCK, GRID_ITEMS_PER_BLOCK * 2);

    const int INPUT_OFFSET_THIS_THREAD = n * INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X;
    const int INPUT_C_STRIDE = INPUT0_SIZE_Y * INPUT0_SIZE_X;
    const int OUTPUT_C_STRIDE = OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int GRID_VAL_PER_THREAD = GRID_ITEMS_PER_THREAD * 2;  //< Each item has 2 vals(h,w).

    // The basic idea is to cache and reuse grid vals for getting close to
    // optimal numer of loads(and stores).
    for (int thisThreadHW = get_local_linear_id() * GRID_VAL_PER_THREAD; thisThreadHW < GRID_ITEMS_FOR_THIS_BLOCK;
         thisThreadHW += 2 * BLOCK_SIZE) {
        const int globalThisThreadHW = (thisThreadHW + GRID_OFFSET_FOR_THIS_BLOCK) / 2;
        const int h = globalThisThreadHW / OUTPUT_SIZE_X;
        const int w = globalThisThreadHW % OUTPUT_SIZE_X;

        const int OUTPUT_OFFSET_THIS_THREAD =
            n * OUTPUT_FEATURE_NUM * OUTPUT_SIZE_Y * OUTPUT_SIZE_X + h * OUTPUT_SIZE_X + w;

        const grid_et x_n = grid_for_this_block[thisThreadHW];
        const grid_et y_n = grid_for_this_block[thisThreadHW + 1];
        const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
        const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
        const int y_topleft = (int)floor(y_d);
        const int x_topleft = (int)floor(x_d);
        const grid_et dy = y_d - y_topleft;
        const grid_et dx = x_d - x_topleft;

        const bool y_topleft_valid = is_between(y_topleft, 0, INPUT0_SIZE_Y);
        const bool y_topleft_plus_valid = is_between(y_topleft + 1, 0, INPUT0_SIZE_Y);
        const bool x_topleft_valid = is_between(x_topleft, 0, INPUT0_SIZE_X);
        const bool x_topleft_plus_valid = is_between(x_topleft + 1, 0, INPUT0_SIZE_X);

        const bool v00_valid = y_topleft_valid && x_topleft_valid;
        const bool v01_valid = y_topleft_valid && x_topleft_plus_valid;
        const bool v10_valid = y_topleft_plus_valid && x_topleft_valid;
        const bool v11_valid = y_topleft_plus_valid && x_topleft_plus_valid;

        const int v00_OFFSET = v00_valid ? (INPUT_OFFSET_THIS_THREAD + y_topleft * INPUT0_SIZE_X + x_topleft) : 0;
        const int v01_OFFSET = v01_valid ? (INPUT_OFFSET_THIS_THREAD + y_topleft * INPUT0_SIZE_X + x_topleft + 1) : 0;
        const int v10_OFFSET = v10_valid ? (INPUT_OFFSET_THIS_THREAD + (y_topleft + 1) * INPUT0_SIZE_X + x_topleft) : 0;
        const int v11_OFFSET =
            v11_valid ? (INPUT_OFFSET_THIS_THREAD + (y_topleft + 1) * INPUT0_SIZE_X + x_topleft + 1) : 0;

#pragma unroll
        for (int c = 0; c < OUTPUT_FEATURE_NUM; ++c) {
            const int INPUT_OFFSET_FOR_THIS_C = c * INPUT_C_STRIDE;

            // WARNING: This loads may read from 'wrong' location - this
            // is done intentianally to keep warp without need to sync
            // and allows for having multiple such loads on the fly - if
            // compiler is smart enough.
            // Otherwise, if load is done conditionally, software pipelinging
            // is hindered by having warp sync due to warp divergence.
            // Tested on a770 GPU with ocl 3.0
            const data_et v00_d = data[v00_OFFSET + INPUT_OFFSET_FOR_THIS_C];
            const data_et v01_d = data[v01_OFFSET + INPUT_OFFSET_FOR_THIS_C];
            const data_et v10_d = data[v10_OFFSET + INPUT_OFFSET_FOR_THIS_C];
            const data_et v11_d = data[v11_OFFSET + INPUT_OFFSET_FOR_THIS_C];

            const data_et v00 = v00_valid ? v00_d * (1 - dx) : 0;
            const data_et v01 = v01_valid ? v01_d * dx : 0;
            const data_et v10 = v10_valid ? v10_d * (1 - dx) : 0;
            const data_et v11 = v11_valid ? v11_d * dx : 0;

            const data_et q0 = v00 + v01;
            const data_et q1 = v10 + v11;
            const data_et out = dy * q1 + (1 - dy) * q0;

            output[OUTPUT_OFFSET_THIS_THREAD + c * OUTPUT_C_STRIDE] = out;
        }
    }
}
#undef interpolate
#undef get_padded
#undef denormalize
