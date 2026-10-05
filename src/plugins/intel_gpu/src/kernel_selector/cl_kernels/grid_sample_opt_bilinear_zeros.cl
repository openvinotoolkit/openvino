// Copyright (C) 2018-2026 Intel Corporation
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

#define PRE_CALC_VALID_OFFSETS_FOR_INPUT_LOAD(x_n, y_n)                \
    const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);               \
    const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);               \
    const int y_topleft = (int)floor(y_d);                             \
    const int x_topleft = (int)floor(x_d);                             \
    const grid_et dy = y_d - y_topleft;                                \
    const grid_et dx = x_d - x_topleft;                                \
                                                                        \
    const bool y0_valid = is_between(y_topleft, 0, INPUT0_SIZE_Y);     \
    const bool y1_valid = is_between(y_topleft + 1, 0, INPUT0_SIZE_Y); \
    const bool x0_valid = is_between(x_topleft, 0, INPUT0_SIZE_X);     \
    const bool x1_valid = is_between(x_topleft + 1, 0, INPUT0_SIZE_X); \
                                                                        \
    const bool v00_valid = y0_valid && x0_valid;                       \
    const bool v01_valid = y0_valid && x1_valid;                       \
    const bool v10_valid = y1_valid && x0_valid;                       \
    const bool v11_valid = y1_valid && x1_valid;                       \
                                                                        \
    const int y0c = y0_valid ? y_topleft : 0;                          \
    const int y1c = y1_valid ? (y_topleft + 1) : 0;                    \
    const int x0c = x0_valid ? x_topleft : 0;                          \
    const int x1c = x1_valid ? (x_topleft + 1) : 0;

// WARNING: This loads may read from 'wrong' location
// (in sense that is has nothing to do with
// sampling point being calculated) - this is done
// intentianally to keep warp without need to sync
// and allows for having multiple such loads on the fly - if
// compiler is smart enough.
// Otherwise, if load is done conditionally, software pipelinging
// is hindered by having warp sync due to warp divergence.
// Tested on a770 GPU with ocl 3.0
#define LOAD_INPUT(c)                                             \
    const data_et v00_d = data[INPUT0_GET_INDEX(n, c, y0c, x0c)]; \
    const data_et v01_d = data[INPUT0_GET_INDEX(n, c, y0c, x1c)]; \
    const data_et v10_d = data[INPUT0_GET_INDEX(n, c, y1c, x0c)]; \
    const data_et v11_d = data[INPUT0_GET_INDEX(n, c, y1c, x1c)];

#define INTERPOLATE()                                      \
    const data_et v00 = v00_valid ? v00_d * (1 - dx) : 0;  \
    const data_et v01 = v01_valid ? v01_d * dx : 0;        \
    const data_et v10 = v10_valid ? v10_d * (1 - dx) : 0;  \
    const data_et v11 = v11_valid ? v11_d * dx : 0;        \
                                                           \
    const data_et q0 = v00 + v01;                          \
    const data_et q1 = v10 + v11;                          \
    const data_et out = dy * q1 + (1 - dy) * q0;

#define STORE(c) output[OUTPUT_GET_INDEX(n, c, h, w)] = out;

// ====================================================================
//
// GRID SAMPLE KERNEL
//
// ====================================================================

KERNEL(grid_sample_opt_bilinear_zeros)(const __global data_t* restrict data,
                                       const __global grid_t* restrict grid,
                                       __global output_t* restrict output) {
#if !defined(INTERPOLATION_MODE_BILINEAR)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support bilinear interppolation mode.
#endif

#if !defined(PADDING_MODE_ZEROS)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support zeros padding mode.
#endif

    const int n = get_global_id(0);

    const int OUTPUT_C_STRIDE = OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int LOCAL_GRID_OFFSET_FOR_THIS_BLOCK = GRID_ITEMS_PER_BLOCK * 2 * get_group_id(1);
    const int BLOCK_SIZE = get_local_size(1);
    const int GRID_ITEMS_FOR_THIS_BLOCK =
        min(OUTPUT_C_STRIDE * 2 - LOCAL_GRID_OFFSET_FOR_THIS_BLOCK, GRID_ITEMS_PER_BLOCK * 2);

    for (int thisThreadHW = get_local_linear_id() * 2; thisThreadHW < GRID_ITEMS_FOR_THIS_BLOCK;
         thisThreadHW += 2 * BLOCK_SIZE) {
        const int globalThisThreadHW = (thisThreadHW + LOCAL_GRID_OFFSET_FOR_THIS_BLOCK) / 2;
        const int h = globalThisThreadHW / OUTPUT_SIZE_X;
        const int w = globalThisThreadHW % OUTPUT_SIZE_X;

        const grid_et x_n = grid[INPUT1_GET_INDEX(n, h, w, 0)];
        const grid_et y_n = grid[INPUT1_GET_INDEX(n, h, w, 1)];

        PRE_CALC_VALID_OFFSETS_FOR_INPUT_LOAD(x_n, y_n);

#pragma unroll
        for (int c = 0; c < OUTPUT_FEATURE_NUM; ++c) {
            LOAD_INPUT(c);
            INTERPOLATE();
            STORE(c);
        }
    }
}

#undef denormalize
#undef is_between
#undef STORE
#undef INTERPOLATE
#undef LOAD_INPUT
#undef PRE_CALC_VALID_OFFSETS_FOR_INPUT_LOAD
