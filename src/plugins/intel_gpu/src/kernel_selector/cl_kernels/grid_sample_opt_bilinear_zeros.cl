// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef INPUT0_TYPE data_t;
typedef INPUT1_TYPE grid_t;
typedef OUTPUT_TYPE output_t;

typedef INPUT0_TYPE data_et;
typedef float grid_et;
typedef OUTPUT_TYPE output_et;

inline const data_et FUNC(
    get_data_single_value)(const data_t* buffer, const size_t n, const size_t c, const size_t h, const size_t w) {
    const size_t idx = INPUT0_GET_INDEX(n, c, h, w);
    return buffer[idx];
}
#define get_data_single_value FUNC_CALL(get_data_single_value)

inline const grid_et FUNC(
    get_grid_single_value)(const grid_t* buffer, const size_t n, const size_t c, const size_t h, const size_t w) {
    const size_t idx = INPUT1_GET_INDEX(n, h, w, c);
    return buffer[idx];
}
#define get_grid_single_value FUNC_CALL(get_grid_single_value)

inline void FUNC(set_output_single_value)(const output_et value,
                                          output_t* buffer,
                                          const size_t n,
                                          const size_t c,
                                          const size_t h,
                                          const size_t w) {
    const size_t idx = OUTPUT_GET_INDEX(n, c, h, w);
    buffer[idx] = value;
}
#define set_output_single_value FUNC_CALL(set_output_single_value)

#if defined(ALIGN_CORNERS)
#define rescale_align FUNC(denormalize)
inline grid_et rescale_align(const grid_et value, const size_t range) {
    return (value + 1) * ((grid_et)(range)-1) / 2;
}
#else
#define rescale_noalign FUNC(denormalize)
inline grid_et rescale_noalign(const grid_et value, const size_t range) {
    return ((value + 1) * (grid_et)(range)-1) / 2;
}
#endif
#define denormalize FUNC_CALL(denormalize)

#if defined(PADDING_MODE_ZEROS)
#define zeros_padding FUNC(get_padded)
inline data_et zeros_padding(const data_t* data, const size_t n, const size_t c, const long y_d, const long x_d) {
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    if (y_d < 0 || x_d < 0 || y_d >= H || x_d >= W) {
        return 0;
    } else {
        const size_t y = (size_t)(y_d);
        const size_t x = (size_t)(x_d);
        return get_data_single_value(data, n, c, y, x);
    }
}
#undef zeros_padding
#else
#error [clDNN grid_sample_ref.cl]: undefined padding mode
#endif

#define get_padded FUNC_CALL(get_padded)

KERNEL(grid_sample_opt_bilinear_zeros)(const __global data_t* restrict data,
                        const __global grid_t* restrict grid,
                        __global output_t* restrict output
                        #if HAS_FUSED_OPS_DECLS
                        , FUSED_OPS_DECLS
                        #endif
                       ) {

#if !defined(INTERPOLATION_MODE_BILINEAR)
#error [clDNN grid_sample_opt_bilinear.cl]: This kernel only support bilinear interppolation mode.
#endif

#if !defined(PADDING_MODE_ZEROS)
#error [clDNN grid_sample_opt_bilinear.cl]: This kernel only support zeros padding mode.
#endif

    const int n = get_global_id(0);
    const int GRID_OFFSET_FOR_THIS_BLOCK = GRID_ELEMENTS_PER_BLOCK*2*get_group_id(1);
    const int BLOCK_SIZE = get_local_size(1);
    const grid_t* restrict grid_for_this_block = grid + GRID_OFFSET_FOR_THIS_BLOCK;
    const int GRID_ELEMS_FOR_THIS_BLOCK = min(OUTPUT_SIZE_Y*OUTPUT_SIZE_X*2 - GRID_OFFSET_FOR_THIS_BLOCK, GRID_ELEMENTS_PER_BLOCK*2); 
    
    for(int thisThreadHW = get_local_linear_id()*2; thisThreadHW < GRID_ELEMS_FOR_THIS_BLOCK; thisThreadHW += 2*BLOCK_SIZE) {
        const int globalThisThreadHW = (thisThreadHW + GRID_OFFSET_FOR_THIS_BLOCK)/2;
        const int h = globalThisThreadHW / OUTPUT_SIZE_X;
        const int w = globalThisThreadHW % OUTPUT_SIZE_X;

        const grid_et x_n = grid_for_this_block[thisThreadHW];
        const grid_et y_n = grid_for_this_block[thisThreadHW+1];
        const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
        const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
        const grid_et y_topleft = floor(y_d);
        const grid_et x_topleft = floor(x_d);
        const grid_et dy = y_d - y_topleft;
        const grid_et dx = x_d - x_topleft;
    
        #pragma unroll
        for(int c = 0; c < OUTPUT_FEATURE_NUM; ++c) {
            const data_et v00 = get_padded(data, n, c, y_topleft, x_topleft);
            const data_et v01 = get_padded(data, n, c, y_topleft, x_topleft + 1);
            const data_et v10 = get_padded(data, n, c, y_topleft + 1, x_topleft);
            const data_et v11 = get_padded(data, n, c, y_topleft + 1, x_topleft + 1);

            const data_et q0 = (1 - dx) * v00 + dx * v01;
            const data_et q1 = (1 - dx) * v10 + dx * v11;
            const data_et out = dy * q1 + (1 - dy) * q0;
            set_output_single_value(out, output, n, c, h, w);
        }
    }
}
#undef interpolate
#undef get_padded
#undef denormalize
