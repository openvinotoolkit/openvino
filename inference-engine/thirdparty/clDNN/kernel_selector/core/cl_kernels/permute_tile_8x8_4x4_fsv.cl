// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch.cl"
#include "include/common.cl"
#include "include/data_types.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)
#define YZ_REMAINDER_LESS_THAN_TILE_SIZE ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE < ( TILE_SIZE /2)))
#define YZ_REMAINDER_MORE_THAN_TILE_SIZE ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE >= ( TILE_SIZE /2)))

#define INPUTVTYPE CAT(INPUT0_TYPE, TILE_SIZE)
#define OUTPUTVTYPE CAT(OUTPUT_TYPE, TILE_SIZE)
#define VLOAD CAT(vload, TILE_SIZE)
#define VSTORE CAT(vstore, TILE_SIZE)
#define AS_INPUTVTYPE CAT(as_, INPUTVTYPE)

#define GET_GLOBAL_ID(IDX) ((uint)get_global_id(IDX))
#define GET_LOCAL_ID(IDX) ((uint)get_local_id(IDX))
#define GET_LOCAL_SIZE(IDX) ((uint)get_local_size(IDX))

KERNEL (permute_tile_8x8_4x4_fsv)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if INPUT0_DIMS == 4
    const uint y = (GET_GLOBAL_ID(1) / INPUT0_SIZE_X) * TILE_SIZE;
    const uint x = (GET_GLOBAL_ID(1)) % INPUT0_SIZE_X;
#elif INPUT0_DIMS == 5
    const uint z = (GET_GLOBAL_ID(1)/ (INPUT0_SIZE_X * INPUT0_SIZE_Y)) * TILE_SIZE;
    const uint yx = GET_GLOBAL_ID(1) % (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint y = yx / INPUT0_SIZE_X ;
    const uint x = yx % INPUT0_SIZE_X;
#endif
    const uint fsv = GET_GLOBAL_ID(0) * TILE_SIZE;
    const uint fs = GET_GLOBAL_ID(2) % INPUT0_FEATURE_SLICE_NUM;
    const uint b = GET_GLOBAL_ID(2) / INPUT0_FEATURE_SLICE_NUM;
    const uint f = fsv + fs * FSV_ALIGNMENT;

    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];
    const uint local_id = GET_LOCAL_ID(0) * GET_LOCAL_SIZE(2) * GET_LOCAL_SIZE(1)
                    + GET_LOCAL_ID(1) * GET_LOCAL_SIZE(2)
                    + GET_LOCAL_ID(2);
    const uint local_buf_offset = local_id * TILE_SIZE;

    if (F_NO_REMAINDER_CONDITION) {
        // read and transpose
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
            const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));

            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                const uint dst = local_buf_offset + lw;
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[lw];
                FUSED_OPS;
                transpose_buf[dst][lh] = FUSED_OPS_RESULT;
#else
                transpose_buf[dst][lh] = ACTIVATION(read_data[lw], ACTIVATION_PARAMS);
#endif
            }
        }
        // write to ddr
#ifdef YZ_REMAINDER_CONDITION
        if (YZ_REMAINDER_LESS_THAN_TILE_SIZE) {
            // copy one by one when z % TILE_SIZE < TILE_SIZE/2
            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                unroll_for (uint lh = 0; lh < YZ_REMAINDER_SIZE; ++lh) {
                    output[output_idx + lh] = transpose_buf[local_buf_offset + lw][lh];
                }
            }
        } else if (YZ_REMAINDER_MORE_THAN_TILE_SIZE) {
            // use vstore and fill zero when z % TILE_SIZE > TILE_SIZE/2
            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
                unroll_for (uint lh = YZ_REMAINDER_SIZE; lh < TILE_SIZE; ++lh) {
                    output[output_idx + lh] = 0.f;
                }
            }
        } else if (YZ_NO_REMAINDER_CONDITION) {
            // use vstore when z % TILE_SIZE == 0
            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
            }
        }
#else
        unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
        }
#endif
    }
#ifdef F_REMAINDER_CONDITION
    else if (F_REMAINDER_CONDITION) {
        // read and transpose
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
            const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                uint dst = local_buf_offset + lw;
    #if HAS_FUSED_OPS
                    INPUT0_TYPE input_var = read_data[lw];
                    FUSED_OPS;
                    transpose_buf[dst][lh] = FUSED_OPS_RESULT;
    #else
                    transpose_buf[dst][lh] = ACTIVATION(read_data[lw], ACTIVATION_PARAMS);
    #endif
            }
        }
        // write to ddr
#ifdef YZ_REMAINDER_CONDITION
        if (YZ_REMAINDER_LESS_THAN_TILE_SIZE) {
            // copy one by one when z % TILE_SIZE < TILE_SIZE/2
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                unroll_for (uint lh = 0; lh < YZ_REMAINDER_SIZE; ++lh) {
                    output[output_idx + lh] = transpose_buf[local_buf_offset + lw][lh];
                }
            }
        } else if (YZ_REMAINDER_MORE_THAN_TILE_SIZE) {
            // use vstore and fill zero when z % TILE_SIZE > TILE_SIZE/2
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
                // zero fill for unaligned
                unroll_for (uint lh = YZ_REMAINDER_SIZE; lh < TILE_SIZE; ++lh) {
                    output[output_idx + lh] = 0.f;
                }
            }
        } else if (YZ_NO_REMAINDER_CONDITION) {
            // use vstore when z % TILE_SIZE == 0
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
            }
        }
#else
        unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
        }
#endif
    }
#endif
}
