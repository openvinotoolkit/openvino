// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL (permute_bfzyx_to_bfyxz)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint b = get_global_id(2) / (NZ_TILES * INPUT0_FEATURE_NUM) % INPUT0_BATCH_NUM;
    const uint f = get_global_id(2) / NZ_TILES % INPUT0_FEATURE_NUM;
    const uint z = get_global_id(2) % NZ_TILES;

    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];

    int local_id = get_local_id(0) * get_local_size(2) * get_local_size(1)
                   + get_local_id(1) * get_local_size(2)
                   + get_local_id(2);

    int local_buf_offset = local_id * LOCAL_BUF_STRIDE;

    if (NORMAL_TILE_CONDITION) {
        for (int lh = 0; lh < TILE_SIZE; ++lh) {
            // vectorwidth == tilesize
            // read
            unsigned int input_idx = INPUT0_GET_INDEX(b, f, (z * TILE_SIZE + lh), y, (x * TILE_SIZE));
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // transpose
            unroll_for (int i = 0; i < TILE_SIZE; ++i) {
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
        for(int lh = 0; lh < TILE_SIZE; ++lh) {
            unsigned int output_idx = OUTPUT_GET_INDEX(b, f, y, (x * TILE_SIZE + lh), (z * TILE_SIZE));
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    }
#ifdef Z_REMAINDER_ITEM
    else if (Z_REMAINDER_CONDITION) {
        for (int lh = 0; lh < Z_REMAINDER_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_INDEX(b, f, (z * TILE_SIZE + lh), y, (x * TILE_SIZE));
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
            unsigned int output_idx = OUTPUT_GET_INDEX(b, f, y, (x * TILE_SIZE + lh), (z * TILE_SIZE));
            for ( int i = 0; i < Z_REMAINDER_SIZE; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
#endif
#ifdef X_REMAINDER_ITEM
    else if (X_REMAINDER_CONDITION) {
        // read
        for (int lh = 0; lh < TILE_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_INDEX(b, f, (z * TILE_SIZE + lh), y, (x * TILE_SIZE));
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
            unsigned int output_idx = OUTPUT_GET_INDEX(b, f, y, (x * TILE_SIZE + lh), (z * TILE_SIZE));
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    }
#endif
#if defined(X_REMAINDER_ITEM) && defined(Z_REMAINDER_ITEM)
     else if (z == Z_REMAINDER_ITEM && x == X_REMAINDER_ITEM) {
        // point by point
        for (int lh = 0; lh < Z_REMAINDER_SIZE; ++lh) {
            // read
            unsigned int input_idx = INPUT0_GET_INDEX(b, f, (z * TILE_SIZE + lh), y, (x * TILE_SIZE));
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
            unsigned int output_idx = OUTPUT_GET_INDEX(b, f, y, (x * TILE_SIZE + lh), (z * TILE_SIZE));
            for ( int i = 0; i < Z_REMAINDER_SIZE; ++i) {
                output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
            }
        }
    }
#endif
}

