// Copyright (C) 2025 Intel Corporation
// SPdx_1-License-Identifier: Apache-2.0
//

typedef INPUT0_TYPE data_t;
typedef INPUT1_TYPE grid_t;
typedef OUTPUT_TYPE output_t;

typedef INPUT0_TYPE data_et;
typedef float grid_et;
typedef OUTPUT_TYPE output_et;

#define REPEAT_1(FN)  FN(0)
#define REPEAT_2(FN)  REPEAT_1(FN) FN(1)
#define REPEAT_3(FN)  REPEAT_2(FN) FN(2)
#define REPEAT_4(FN)  REPEAT_3(FN) FN(3)
#define REPEAT_5(FN)  REPEAT_4(FN) FN(4)
#define REPEAT_6(FN)  REPEAT_5(FN) FN(5)
#define REPEAT_7(FN)  REPEAT_6(FN) FN(6)
#define REPEAT_8(FN)  REPEAT_7(FN) FN(7)
#define REPEAT_9(FN)  REPEAT_8(FN) FN(8)
#define REPEAT_10(FN) REPEAT_9(FN) FN(9)

#define REPEAT(FN, N)        REPEAT_##N(FN)
#define REPEAT_EXPAND(FN, N) REPEAT(FN, N)

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

#define LOAD_GRID(INDEX)                                                       \
    const grid_et x_n_##INDEX = grid_for_this_block[thisThreadHW + 2 * INDEX]; \
    const grid_et y_n_##INDEX = grid_for_this_block[thisThreadHW + 1 + 2 * INDEX];

#define CALC_VALID_OFFSETS(INDEX)                                                                                      \
    const grid_et y_d_##INDEX = denormalize(y_n_##INDEX, INPUT0_SIZE_Y);                                               \
    const grid_et x_d_##INDEX = denormalize(x_n_##INDEX, INPUT0_SIZE_X);                                               \
    const int y_topleft_##INDEX = (int)floor(y_d_##INDEX);                                                             \
    const int x_topleft_##INDEX = (int)floor(x_d_##INDEX);                                                             \
    const grid_et dy_##INDEX = y_d_##INDEX - y_topleft_##INDEX;                                                        \
    const grid_et dx_##INDEX = x_d_##INDEX - x_topleft_##INDEX;                                                        \
                                                                                                                       \
    const bool y_topleft_##INDEX##_valid = is_between(y_topleft_##INDEX, 0, INPUT0_SIZE_Y);                            \
    const bool y_topleft_##INDEX##_plus_valid = is_between(y_topleft_##INDEX + 1, 0, INPUT0_SIZE_Y);                   \
    const bool x_topleft_##INDEX##_valid = is_between(x_topleft_##INDEX, 0, INPUT0_SIZE_X);                            \
    const bool x_topleft_##INDEX##_plus_valid = is_between(x_topleft_##INDEX + 1, 0, INPUT0_SIZE_X);                   \
                                                                                                                       \
    const bool v00_##INDEX##_valid = y_topleft_##INDEX##_valid && x_topleft_##INDEX##_valid;                           \
    const bool v01_##INDEX##_valid = y_topleft_##INDEX##_valid && x_topleft_##INDEX##_plus_valid;                      \
    const bool v10_##INDEX##_valid = y_topleft_##INDEX##_plus_valid && x_topleft_##INDEX##_valid;                      \
    const bool v11_##INDEX##_valid = y_topleft_##INDEX##_plus_valid && x_topleft_##INDEX##_plus_valid;                 \
                                                                                                                       \
    const int v00_##INDEX##_OFFSET =                                                                                   \
        v00_##INDEX##_valid ? (INPUT_OFFSET_THIS_THREAD + y_topleft_##INDEX * INPUT0_SIZE_X + x_topleft_##INDEX) : 0;  \
    const int v01_##INDEX##_OFFSET =                                                                                   \
        v01_##INDEX##_valid ? (INPUT_OFFSET_THIS_THREAD + y_topleft_##INDEX * INPUT0_SIZE_X + x_topleft_##INDEX + 1)   \
                            : 0;                                                                                       \
    const int v10_##INDEX##_OFFSET =                                                                                   \
        v10_##INDEX##_valid ? (INPUT_OFFSET_THIS_THREAD + (y_topleft_##INDEX + 1) * INPUT0_SIZE_X + x_topleft_##INDEX) \
                            : 0;                                                                                       \
    const int v11_##INDEX##_OFFSET =                                                                                   \
        v11_##INDEX##_valid                                                                                            \
            ? (INPUT_OFFSET_THIS_THREAD + (y_topleft_##INDEX + 1) * INPUT0_SIZE_X + x_topleft_##INDEX + 1)             \
            : 0;

// WARNING: This loads may read from 'wrong' location - this
// is done intentianally to keep warp without need to sync
// and allows for having multiple such loads on the fly - if
// compiler is smart enough.
// Otherwise, if load is done conditionally, software pipelinging
// is hindered by having warp sync due to warp divergence.
// Tested on a770 GPU with ocl 3.0
#define INTERPOLATE(INDEX)                                                                    \
    const data_et v00_##INDEX##_d = data[v00_##INDEX##_OFFSET + INPUT_OFFSET_FOR_THIS_C];     \
    const data_et v01_##INDEX##_d = data[v01_##INDEX##_OFFSET + INPUT_OFFSET_FOR_THIS_C];     \
    const data_et v10_##INDEX##_d = data[v10_##INDEX##_OFFSET + INPUT_OFFSET_FOR_THIS_C];     \
    const data_et v11_##INDEX##_d = data[v11_##INDEX##_OFFSET + INPUT_OFFSET_FOR_THIS_C];     \
                                                                                              \
    const data_et v00_##INDEX = v00_##INDEX##_valid ? v00_##INDEX##_d * (1 - dx_##INDEX) : 0; \
    const data_et v01_##INDEX = v01_##INDEX##_valid ? v01_##INDEX##_d * dx_##INDEX : 0;       \
    const data_et v10_##INDEX = v10_##INDEX##_valid ? v10_##INDEX##_d * (1 - dx_##INDEX) : 0; \
    const data_et v11_##INDEX = v11_##INDEX##_valid ? v11_##INDEX##_d * dx_##INDEX : 0;       \
                                                                                              \
    const data_et q0_##INDEX = v00_##INDEX + v01_##INDEX;                                     \
    const data_et q1_##INDEX = v10_##INDEX + v11_##INDEX;                                     \
    const data_et out_##INDEX = dy_##INDEX * q1_##INDEX + (1 - dy_##INDEX) * q0_##INDEX;

#define STORE(INDEX) output[OUTPUT_OFFSET_THIS_THREAD + c * OUTPUT_C_STRIDE + INDEX] = out_##INDEX;

// ---------------------------------------------------------------------------------
//
// GRID SAMPLE KERNEL
//
// ---------------------------------------------------------------------------------

//__attribute__((reqd_work_group_size(1, 256, 1)))
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

        REPEAT_EXPAND(LOAD_GRID, GRID_ITEMS_PER_THREAD);
        REPEAT_EXPAND(CALC_VALID_OFFSETS, GRID_ITEMS_PER_THREAD);

#pragma unroll
        for (int c = 0; c < OUTPUT_FEATURE_NUM; ++c) {
            const int INPUT_OFFSET_FOR_THIS_C = c * INPUT_C_STRIDE;

            REPEAT_EXPAND(INTERPOLATE, GRID_ITEMS_PER_THREAD);
            REPEAT_EXPAND(STORE, GRID_ITEMS_PER_THREAD);
        }
    }
}

#undef denormalize
#undef STORE
#undef INTERPOLATE
#undef CALC_VALID_OFFSETS
#undef LOAD_GRID
