// Copyright (c) 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "include/fetch.cl"
#include "include/common.cl"
#include "include/data_types.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)
#define YZ_REMAINDER_LESS_THAN_TILE_HEIGHT ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE < ( TILE_HEIGHT /2)))
#define YZ_REMAINDER_MORE_THAN_TILE_HEIGHT ((YZ_REMAINDER_CONDITION) && (YZ_REMAINDER_SIZE >= ( TILE_HEIGHT /2)))

KERNEL (permute_tile_8x8_4x4_fsv)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if INPUT0_DIMS == 4
    const uint y = (get_global_id(1) / INPUT0_SIZE_X) * TILE_HEIGHT;
    const uint x = get_global_id(1) % INPUT0_SIZE_X;
#elif INPUT0_DIMS == 5
    const uint z = (get_global_id(1)/ (INPUT0_SIZE_X*INPUT0_SIZE_Y)) * TILE_HEIGHT;
    const uint yx = get_global_id(1) % (INPUT0_SIZE_X*INPUT0_SIZE_Y);
    const uint y = yx / INPUT0_SIZE_X ;
    const uint x = yx % INPUT0_SIZE_X;
#endif
    const uint fsv = get_global_id(0) * TILE_WIDTH;
    const uint fs = get_global_id(2) % (INPUT0_SIZE_FS);
    const uint b = get_global_id(2) / (INPUT0_SIZE_FS);
    const uint f = fsv + fs*FSV_ALIGNMENT;

    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];
    const uint local_id = get_local_id(0) * get_local_size(2) * get_local_size(1)
                    + get_local_id(1) * get_local_size(2)
                    + get_local_id(2);
    const uint local_buf_offset = local_id * TILE_WIDTH;

    if (F_NO_REMAINDER_CONDITION)
    {
        // read and transpose
        unroll_for (uint lh = 0; lh < TILE_HEIGHT; ++lh) {
            const uint input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));

            unroll_for (uint lw = 0; lw < TILE_WIDTH; ++lw) {
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
        if (YZ_REMAINDER_LESS_THAN_TILE_HEIGHT)
        {
            // copy one by one when z % TILE_HEIGHT < TILE_HEIGHT/2
            unroll_for (uint lw = 0; lw < TILE_WIDTH; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                unroll_for (uint lh = 0; lh < YZ_REMAINDER_SIZE; ++lh) {
                    output[output_idx + lh] = transpose_buf[local_buf_offset + lw][lh];
                }
            }
        }
        else if (YZ_REMAINDER_MORE_THAN_TILE_HEIGHT)
        {
            // use vstore and fill zero when z % TILE_HEIGHT > TILE_HEIGHT/2
            unroll_for (uint lw = 0; lw < TILE_WIDTH; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
                unroll_for (uint lh = YZ_REMAINDER_SIZE; lh < TILE_HEIGHT; ++lh) {
                    output[output_idx + lh] = 0.f;
                }
            }
        }
        else if (YZ_NO_REMAINDER_CONDITION)
        {
            // use vstore when z % TILE_HEIGHT == 0
            unroll_for (uint lw = 0; lw < TILE_WIDTH; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
            }
        }
#else
        unroll_for (uint lw = 0; lw < TILE_WIDTH; ++lw) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
        }
#endif
    }
#ifdef F_REMAINDER_CONDITION
    else if (F_REMAINDER_CONDITION) {
        // read and transpose
        unroll_for (uint lh = 0; lh < TILE_HEIGHT; ++lh) {
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
        if (YZ_REMAINDER_LESS_THAN_TILE_HEIGHT)
        {
            // copy one by one when z % TILE_HEIGHT < TILE_HEIGHT/2
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                unroll_for (uint lh = 0; lh < YZ_REMAINDER_SIZE; ++lh) {
                    output[output_idx + lh] = transpose_buf[local_buf_offset + lw][lh];
                }
            }
        }
        else if (YZ_REMAINDER_MORE_THAN_TILE_HEIGHT)
        {
            // use vstore and fill zero when z % TILE_HEIGHT > TILE_HEIGHT/2
            unroll_for (uint lw = 0; lw < F_REMAINDER_SIZE; ++lw) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                VSTORE(transpose_buf[local_buf_offset + lw], 0, output + output_idx);
                // zero fill for unaligned
                unroll_for (uint lh = YZ_REMAINDER_SIZE; lh < TILE_HEIGHT; ++lh) {
                    output[output_idx + lh] = 0.f;
                }
            }
        }
        else if (YZ_NO_REMAINDER_CONDITION)
        {
            // use vstore when z % TILE_HEIGHT == 0
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
