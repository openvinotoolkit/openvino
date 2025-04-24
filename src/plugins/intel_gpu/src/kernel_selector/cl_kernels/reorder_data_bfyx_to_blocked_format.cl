// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)

#define INPUTVTYPE CAT(INPUT0_TYPE, TILE_SIZE)
#define OUTPUTVTYPE CAT(OUTPUT_TYPE, TILE_SIZE)
#define VLOAD CAT(vload, TILE_SIZE)
#define VSTORE CAT(vstore, TILE_SIZE)
#define AS_INPUTVTYPE CAT(as_, INPUTVTYPE)
#define TO_OUTPUTVTYPE CAT(convert_, OUTPUTVTYPE)

#define GET_GLOBAL_ID(IDX) ((uint)get_global_id(IDX))
#define GET_LOCAL_ID(IDX) ((uint)get_local_id(IDX))
#define GET_LOCAL_SIZE(IDX) ((uint)get_local_size(IDX))

#define FUNC_VLOAD(inner, outer)    unroll_for (uint lh = 0; lh < outer; ++lh) { \
                                        const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER); \
                                        INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx)); \
                                        unroll_for (uint lw = 0; lw < inner; ++lw) { \
                                            const uint dst = local_buf_offset + lw; \
                                            transpose_buf[dst][lh] = read_data[lw]; \
                                        } \
                                    }

#define FUNC_LOAD_LEFTOVERS(inner, outer)    unroll_for (uint lh = 0; lh < outer; ++lh) { \
                                        const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER); \
                                        INPUTVTYPE read_data; \
                                        unroll_for (uint lw = 0; lw < inner; ++lw) { \
                                            read_data[lw] = input[input_idx + lw]; \
                                        } \
                                        unroll_for (uint lw = 0; lw < inner; ++lw) { \
                                            const uint dst = local_buf_offset + lw; \
                                            transpose_buf[dst][lh] = read_data[lw]; \
                                        } \
                                    }

#define FUNC_VSTORE(loop)           unroll_for (uint lw = 0; lw < loop; ++lw) { \
                                        const uint output_idx = output_idx_tile + (lw * x_pitch); \
                                        VSTORE(TO_OUTPUTVTYPE(transpose_buf[local_buf_offset + lw]), 0, output + output_idx); \
                                    }

#define FUNC_WRITE(inner, outer)    unroll_for (uint lw = 0; lw < outer; ++lw) { \
                                        const uint output_idx = output_idx_tile + (lw * x_pitch); \
                                        unroll_for (uint i = 0; i < inner; ++i) { \
                                            output[output_idx + i] = ACTIVATION(TO_OUTPUT_TYPE(transpose_buf[local_buf_offset + lw][i]), ACTIVATION_PARAMS); \
                                        } \
                                    }

KERNEL (reorder_data_bfyx_to_blocked_format)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
    )
{
#if INPUT0_DIMS == 4
    const uint y = GET_GLOBAL_ID(1) % INPUT0_SIZE_Y;
    const uint x = (GET_GLOBAL_ID(1) / INPUT0_SIZE_Y) * TILE_SIZE;
#elif INPUT0_DIMS == 5
    const uint z = GET_GLOBAL_ID(1) % INPUT0_SIZE_Z;
    const uint yx = GET_GLOBAL_ID(1) / INPUT0_SIZE_Z;
    const uint y = yx % INPUT0_SIZE_Y;
    const uint x = (yx / INPUT0_SIZE_Y) * TILE_SIZE;
#else
#error reorder_data_bfyx_to_blocked_format.cl: input format - not supported
#endif

    const uint fsv = GET_GLOBAL_ID(0) * TILE_SIZE;
    const uint fs = GET_GLOBAL_ID(2) % INPUT0_FEATURE_SLICE_NUM;
    const uint b = GET_GLOBAL_ID(2) / INPUT0_FEATURE_SLICE_NUM;
    const uint f = fsv + fs * FSV_ALIGNMENT;

#if DOUBLE_BLOCKED_FORMAT
    const uint bs = b / BSV_ALIGNMENT;
    const uint bsv = b % BSV_ALIGNMENT;
    const uint x_pitch = BSV_ALIGNMENT * FSV_ALIGNMENT;
#else
    const uint x_pitch = FSV_ALIGNMENT;
#endif
    const uint y_pitch = x_pitch * (OUTPUT_SIZE_X);

#if INPUT0_DIMS == 4
    #if DOUBLE_BLOCKED_FORMAT
        const uint bsv_pitch = FSV_ALIGNMENT;
        const uint fs_pitch = y_pitch * (OUTPUT_SIZE_Y);
        const uint bs_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
        const uint output_idx_tile = (bs * bs_pitch) + (fs * fs_pitch) + (y * y_pitch) + (x * x_pitch) + (bsv * bsv_pitch) + (fsv);
    #else
        #if FS_B_YX_FSV
        const uint b_pitch = y_pitch * (OUTPUT_SIZE_Y);
        const uint fs_pitch = b_pitch * (INPUT0_BATCH_NUM);
        #else
        const uint fs_pitch = y_pitch * (OUTPUT_SIZE_Y);
        const uint b_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
        #endif
        const uint output_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (y * y_pitch) + (x * x_pitch) + (fsv);
    #endif
#elif INPUT0_DIMS == 5
     #if DOUBLE_BLOCKED_FORMAT
        const uint bsv_pitch = FSV_ALIGNMENT;
        const uint z_pitch = y_pitch * (OUTPUT_SIZE_Y);
        const uint fs_pitch = z_pitch * (OUTPUT_SIZE_Z);
        const uint bs_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
        const uint output_idx_tile = (bs * bs_pitch) + (fs * fs_pitch) + (z * z_pitch) + (y * y_pitch) + (x * x_pitch) + (bsv * bsv_pitch) + (fsv);
    #else
        const uint z_pitch = y_pitch * (OUTPUT_SIZE_Y);
        const uint fs_pitch = z_pitch * (OUTPUT_SIZE_Z);
        const uint b_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
        const uint output_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (z * z_pitch) + (y * y_pitch) + (x * x_pitch) + (fsv);
    #endif
#endif

    // get local buf offset
    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];
    const uint local_id = GET_LOCAL_ID(0) * GET_LOCAL_SIZE(2) * GET_LOCAL_SIZE(1)
                    + GET_LOCAL_ID(1) * GET_LOCAL_SIZE(2)
                    + GET_LOCAL_ID(2);
    const uint local_buf_offset = local_id * TILE_SIZE;

    if (F_NO_REMAINDER_CONDITION) {
        // read and transpose
#ifdef X_REMAINDER_CONDITION
        if (X_NO_REMAINDER_CONDITION) {
            FUNC_VLOAD(TILE_SIZE, TILE_SIZE)
        } else {
            FUNC_LOAD_LEFTOVERS(X_REMAINDER_SIZE, TILE_SIZE)
        }
#else
        FUNC_VLOAD(TILE_SIZE, TILE_SIZE)
#endif

        // write to ddr
#ifdef X_REMAINDER_CONDITION
        if (X_NO_REMAINDER_CONDITION) {
            FUNC_VSTORE(TILE_SIZE)
        } else {
            FUNC_VSTORE(X_REMAINDER_SIZE)
        }
#else
        FUNC_VSTORE(TILE_SIZE)
#endif
    }
#ifdef F_REMAINDER_CONDITION
    else if (F_REMAINDER_CONDITION) {
        // read and transpose
    #ifdef X_REMAINDER_CONDITION
        if (X_NO_REMAINDER_CONDITION) {
            FUNC_VLOAD(TILE_SIZE, F_REMAINDER_SIZE)
        } else {
            FUNC_LOAD_LEFTOVERS(X_REMAINDER_SIZE, F_REMAINDER_SIZE)
        }
    #else
        FUNC_VLOAD(TILE_SIZE, F_REMAINDER_SIZE)
    #endif

        // write to ddr
    #ifdef X_REMAINDER_CONDITION
        if (X_NO_REMAINDER_CONDITION) {
            FUNC_WRITE(F_REMAINDER_SIZE, TILE_SIZE)
        } else {
            FUNC_WRITE(F_REMAINDER_SIZE, X_REMAINDER_SIZE)
        }
    #else
        FUNC_WRITE(F_REMAINDER_SIZE, TILE_SIZE)
    #endif
    }
#endif
}

#undef FUNC_WRITE
#undef FUNC_VSTORE
#undef FUNC_VLOAD

#undef GET_LOCAL_SIZE
#undef GET_LOCAL_ID
#undef GET_GLOBAL_ID

#undef TO_OUTPUTVTYPE
#undef AS_INPUTVTYPE
#undef VSTORE
#undef VLOAD
#undef OUTPUTVTYPE
#undef INPUTVTYPE

#undef INPUT0_GET_TILED_INDEX
