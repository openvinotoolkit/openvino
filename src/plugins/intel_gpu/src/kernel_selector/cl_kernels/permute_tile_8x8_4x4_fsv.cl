// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)
#define YZ_REMAINDER_LESS_THAN_TILE_SIZE ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE < ( TILE_SIZE /2)))
#define YZ_REMAINDER_MORE_THAN_TILE_SIZE ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE >= ( TILE_SIZE /2)))

#define INPUTVTYPE CAT(INPUT0_TYPE, TILE_SIZE)
#define OUTPUTVTYPE CAT(OUTPUT_TYPE, TILE_SIZE)
#define VLOAD CAT(vload, TILE_SIZE)
#define VSTORE CAT(vstore, TILE_SIZE)
#define AS_INPUTVTYPE CAT(as_, INPUTVTYPE)
#define AS_OUTPUTVTYPE CAT(as_, OUTPUTVTYPE)
#define TO_OUTPUTVTYPE CAT(convert_, OUTPUTVTYPE)

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

#ifdef REORDERED_OUTPUT_TILED_ORDER
    if (F_NO_REMAINDER_CONDITION) {
#ifdef YZ_REMAINDER_CONDITION
        unroll_for (uint lh = 0; lh < (((YZ_REMAINDER_CONDITION)) ? YZ_REMAINDER_SIZE : TILE_SIZE); ++lh) {
#else
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
#endif
            // read
            const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            // write to ddr
          #if HAS_FUSED_OPS
            OUTPUTVTYPE out_data;
            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                INPUT0_TYPE input_var = read_data[lw];
                FUSED_OPS;
                out_data[lw] = FUSED_OPS_RESULT;
            }
            const uint output_idx = OUTPUT_GET_TILED_INDEX(REORDERED_OUTPUT_TILED_ORDER);
            VSTORE(out_data, 0, output + output_idx);
          #else
            const uint output_idx = OUTPUT_GET_TILED_INDEX(REORDERED_OUTPUT_TILED_ORDER);
            VSTORE(ACTIVATION(TO_OUTPUTVTYPE(read_data), ACTIVATION_PARAMS), 0, output + output_idx);
          #endif
        }
    }
#ifdef F_REMAINDER_CONDITION
    else if (F_REMAINDER_CONDITION) {
#ifdef YZ_REMAINDER_CONDITION
        unroll_for (uint lh = 0; lh < (((YZ_REMAINDER_CONDITION)) ? YZ_REMAINDER_SIZE : TILE_SIZE); ++lh) {
#else
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
#endif
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                // read
                const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
                INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
                // write to ddr
                const uint output_idx = OUTPUT_GET_TILED_INDEX(REORDERED_OUTPUT_TILED_ORDER);
              #if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[lw];
                FUSED_OPS;
                output[output_idx + lw] = FUSED_OPS_RESULT;
              #else
                output[output_idx + lw] = TO_OUTPUT_TYPE(read_data[lw]);
              #endif
            }
        }
    }
#endif // F_REMAINDER_CONDITION
#else // !REORDERED_OUTPUT_TILED_ORDER
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
#else // YZ_REMAINDER_CONDITION
        // write to ddr
        unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
        }
#endif //YZ_REMAINDER_CONDITION
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
#else //  !YZ_REMAINDER_CONDITION
        unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
        }
#endif // YZ_REMAINDER_CONDITION
    }
#endif // F_REMAINDER_CONDITION
#endif // REORDERED_OUTPUT)TILED_ORDER
}

#undef INPUT0_GET_TILED_INDEX(ORDER)
#undef OUTPUT_GET_TILED_INDEX(ORDER)
#undef YZ_REMAINDER_LESS_THAN_TILE_SIZE
#undef YZ_REMAINDER_MORE_THAN_TILE_SIZE
#undef INPUTVTYPE
#undef OUTPUTVTYPE
#undef VLOAD
#undef VSTORE
#undef AS_INPUTVTYPE
#undef AS_OUTPUTVTYPE
#undef TO_OUTPUTVTYPE
#undef GET_GLOBAL_ID(IDX)
#undef GET_LOCAL_ID(IDX)
#undef GET_LOCAL_SIZE(IDX)
