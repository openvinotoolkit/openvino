// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)
KERNEL (permute_tile_8x8_4x4)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = get_global_id(0);
#if IS_DYNAMIC
#if INPUT0_SIZE_X < DEFAULT_TILE_SIZE
#define VEC_WIDTH MIN_TILE_SIZE
#else
#define VEC_WIDTH TILE_SIZE
#endif
    const uint tile_size = VEC_WIDTH;
    const uint f = (uint)get_global_id(2) % CEIL_DIV(INPUT0_FEATURE_NUM, tile_size);
    const uint b = (uint)get_global_id(2) / CEIL_DIV(INPUT0_FEATURE_NUM, tile_size);
#else
    const uint tile_size = TILE_SIZE;
    const uint f = (uint)get_global_id(2) % NFEATURE_TILES;
    const uint b = (uint)get_global_id(2) / NFEATURE_TILES;
#endif

#if INPUT0_DIMS == 4
    //|dim2:bf|dim1:y|dim0:x
    const uint y = get_global_id(1);
#elif INPUT0_DIMS == 5
    //|dim2:bf|dim1:yz|dim0:x
    const uint z = get_global_id(1) / INPUT0_SIZE_Y;
    const uint y = get_global_id(1) % INPUT0_SIZE_Y;
#elif INPUT0_DIMS == 6
    //|dim2:bf|dim1:wyz|dim0:x
    const uint y = get_global_id(1) % INPUT0_SIZE_Y;
    const uint z = get_global_id(1) / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = get_global_id(1) / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
#endif
#if IS_DYNAMIC
    const uint local_buf_stride = tile_size;

    bool x_remainder = false;
    bool f_remainder = false;
    bool normal_tile_cond = true;
    bool x_remainder_cond = true;
    bool f_remainder_cond = true;
    uint x_remainder_item, f_remainder_item;
    uint x_remainder_size, f_remainder_size;
    if (INPUT0_SIZE_X % tile_size) {
        x_remainder = true;
        x_remainder_item = INPUT0_SIZE_X / tile_size;
        x_remainder_size = INPUT0_SIZE_X % tile_size;
        normal_tile_cond = normal_tile_cond && (x < x_remainder_item);
        x_remainder_cond = x_remainder_cond && (x == x_remainder_item);
        f_remainder_cond = f_remainder_cond && (x < x_remainder_item);
    }
    if (INPUT0_FEATURE_NUM % tile_size) {
        f_remainder = true;
        f_remainder_item = INPUT0_FEATURE_NUM / tile_size;
        f_remainder_size = INPUT0_FEATURE_NUM % tile_size;
        normal_tile_cond = normal_tile_cond && (f < f_remainder_item);
        x_remainder_cond = x_remainder_cond && (f < f_remainder_item);
        f_remainder_cond = f_remainder_cond && (f == f_remainder_item);
    }
#else
    const uint local_buf_stride = LOCAL_BUF_STRIDE;
#endif
    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];

    int local_id = get_local_id(0) * get_local_size(2) * get_local_size(1)
                    + get_local_id(1) * get_local_size(2)
                    + get_local_id(2);

    int local_buf_offset = local_id * local_buf_stride;

#if IS_DYNAMIC
    if (normal_tile_cond) {
#else
    if (NORMAL_TILE_CONDITION) {
#endif
        for (int lh = 0; lh < tile_size; ++lh) {
            // vectorwidth == tilesize
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            unsigned int dst_w = lh / tile_size;
#if IS_DYNAMIC
            for (int i = 0; i < tile_size; ++i) {
#else
            unroll_for (int i = 0; i < TILE_SIZE; ++i) {
#endif
                unsigned int dst = local_buf_offset + i;
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[dst][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[dst][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for(int lh = 0; lh < tile_size; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    }
#if IS_DYNAMIC
    else if (f_remainder && f_remainder_cond) {
        for (int lh = 0; lh < f_remainder_size; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            for (int i = 0; i < tile_size; ++i) {
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for (int lh = 0; lh < tile_size; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            for ( int i = 0; i < f_remainder_size; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
#else
#ifdef F_REMAINDER_ITEM
    else if (F_REMAINDER_CONDITION) {
        for (int lh = 0; lh < F_REMAINDER_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            unroll_for (int i = 0; i < TILE_SIZE; ++i) {
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for (int lh = 0; lh < TILE_SIZE; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            for ( int i = 0; i < F_REMAINDER_SIZE; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
#endif
#endif
#if IS_DYNAMIC
    else if (x_remainder && x_remainder_cond) {
        // read
        for (int lh = 0; lh < tile_size; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            for (int i = 0; i < x_remainder_size; ++i) {
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for (int lh = 0; lh < x_remainder_size; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    }
#else
#ifdef X_REMAINDER_ITEM
    else if (X_REMAINDER_CONDITION) {
        // read
        for (int lh = 0; lh < TILE_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            unroll_for (int i = 0; i < X_REMAINDER_SIZE; ++i) {
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for (int lh = 0; lh < X_REMAINDER_SIZE; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    }
#endif
#endif
#if IS_DYNAMIC
    else if (f_remainder && x_remainder && f == f_remainder_item && x == x_remainder_item) {
        // point by point
        for (int lh = 0; lh < f_remainder_size; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            for (int i = 0; i < x_remainder_size; ++i) {
                unsigned int dst = local_buf_offset + i;
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for(int lh = 0; lh < x_remainder_size; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            for ( int i = 0; i < f_remainder_size; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
    #undef VEC_WIDTH
#else
#if defined(X_REMAINDER_ITEM) && defined(F_REMAINDER_ITEM)
     else if (f == F_REMAINDER_ITEM && x == X_REMAINDER_ITEM) {
        // point by point
        for (int lh = 0; lh < F_REMAINDER_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            for (int i = 0; i < X_REMAINDER_SIZE; ++i) {
                unsigned int dst = local_buf_offset + i;
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[local_buf_offset + i][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[local_buf_offset + i][lh] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
        for(int lh = 0; lh < X_REMAINDER_SIZE; ++lh) {
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            for ( int i = 0; i < F_REMAINDER_SIZE; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
#endif
#endif
}
