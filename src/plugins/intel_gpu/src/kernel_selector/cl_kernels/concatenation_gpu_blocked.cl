// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define WORK_GROUP_SIZE 16
#define IC_BLOCK 16

#define INPUT_VEC_TYPE                          MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_XY)
#define OUTPUT_VEC_TYPE                         MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_XY)
#define TO_OUTPUT_VEC_TYPE(x)                   CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define INPUT_BLOCK_READ(ptr, offset)           MAKE_VECTOR_TYPE(DT_INPUT_BLOCK_READ, TILE_XY)(ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val)    MAKE_VECTOR_TYPE(DT_OUTPUT_BLOCK_WRITE, TILE_XY)(ptr, offset, val)

#if !ALIGNED
// For non-aligned case process two features together to mitigate misalignment
#   define TILE_F 2
#else
#   define TILE_F 1
#endif

__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
REQD_SUB_GROUP_SIZE(WORK_GROUP_SIZE)
KERNEL (concatenation_gpu_blocked)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    uint output_offset_in_concat_axis)
{
    const int xy = (uint)get_global_id(0) * TILE_XY;
    const int f_block = (uint)get_group_id(1) * TILE_F;
    const int b = get_group_id(2);
    const int lid = get_sub_group_local_id();

    const int x = xy % OUTPUT_SIZE_X;
    const int y = xy / OUTPUT_SIZE_X;

    const uint input_offset = INPUT0_GET_INDEX(b, f_block*IC_BLOCK, y, x);

#if ALIGNED
    INPUT_VEC_TYPE src = INPUT_BLOCK_READ(input, input_offset);
    const uint dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + output_offset_in_concat_axis), y, x);

    bool do_block_write = (INPUT0_FEATURE_NUM % IC_BLOCK == 0)
                        || (f_block * IC_BLOCK + IC_BLOCK <= INPUT0_FEATURE_NUM);

    if (do_block_write) {
        OUTPUT_VEC_TYPE res = TO_OUTPUT_VEC_TYPE(ACTIVATION(src, ACTIVATION_PARAMS));
        OUTPUT_BLOCK_WRITE(output, dst_index, res);
    } else {
        if (lid < INPUT0_FEATURE_NUM % IC_BLOCK) {
            unroll_for(uint tx = 0; tx < TILE_XY; ++tx) {
                OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(((INPUT0_TYPE*)&src)[tx], ACTIVATION_PARAMS));
                output[dst_index + tx * IC_BLOCK + lid] = res;
            }
        }
    }
#else

#if TILE_F != 1
    bool full_write = (INPUT0_FEATURE_NUM % (IC_BLOCK * TILE_F) == 0) || (f_block * IC_BLOCK + TILE_F * IC_BLOCK <= INPUT0_FEATURE_NUM);
    if (full_write) {
        INPUT_VEC_TYPE src0 = INPUT_BLOCK_READ(input, input_offset + 0 * INPUT0_FEATURE_PITCH * IC_BLOCK);
        INPUT_VEC_TYPE src1 = INPUT_BLOCK_READ(input, input_offset + 1 * INPUT0_FEATURE_PITCH * IC_BLOCK);
    #if TILE_F == 4
        INPUT_VEC_TYPE src2 = INPUT_BLOCK_READ(input, input_offset + 2 * INPUT0_FEATURE_PITCH * IC_BLOCK);
        INPUT_VEC_TYPE src3 = INPUT_BLOCK_READ(input, input_offset + 3 * INPUT0_FEATURE_PITCH * IC_BLOCK);
    #endif

        uint dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + (IC_BLOCK - MISALIGNMENT) + output_offset_in_concat_axis), y, x);

        INPUT_VEC_TYPE src_al0 = 0;
    #if TILE_F == 4
        INPUT_VEC_TYPE src_al1 = 0;
        INPUT_VEC_TYPE src_al2 = 0;
    #endif
        unroll_for(uint tx = 0; tx < TILE_XY; ++tx) {
            ((INPUT0_TYPE*)&src_al0)[tx] = _sub_group_shuffle_down(((INPUT0_TYPE*)&src0)[tx], ((INPUT0_TYPE*)&src1)[tx], (IC_BLOCK - MISALIGNMENT));
    #if TILE_F == 4
            ((INPUT0_TYPE*)&src_al1)[tx] = _sub_group_shuffle_down(((INPUT0_TYPE*)&src1)[tx], ((INPUT0_TYPE*)&src2)[tx], (IC_BLOCK - MISALIGNMENT));
            ((INPUT0_TYPE*)&src_al2)[tx] = _sub_group_shuffle_down(((INPUT0_TYPE*)&src2)[tx], ((INPUT0_TYPE*)&src3)[tx], (IC_BLOCK - MISALIGNMENT));
    #endif
        }
        OUTPUT_VEC_TYPE res_al0 = TO_OUTPUT_VEC_TYPE(ACTIVATION(src_al0, ACTIVATION_PARAMS));
        OUTPUT_BLOCK_WRITE(output, dst_index, res_al0);
    #if TILE_F == 4
        OUTPUT_VEC_TYPE res_al1 = TO_OUTPUT_VEC_TYPE(ACTIVATION(src_al1, ACTIVATION_PARAMS));
        OUTPUT_BLOCK_WRITE(output, dst_index + 1 * OUTPUT_FEATURE_PITCH * IC_BLOCK, res_al1);
        OUTPUT_VEC_TYPE res_al2 = TO_OUTPUT_VEC_TYPE(ACTIVATION(src_al2, ACTIVATION_PARAMS));
        OUTPUT_BLOCK_WRITE(output, dst_index + 2 * OUTPUT_FEATURE_PITCH * IC_BLOCK, res_al2);
    #endif
        uint lid_f_offset = lid;
        INPUT_VEC_TYPE src_unal = 0;

        lid_f_offset += lid < (IC_BLOCK - MISALIGNMENT) ? 0 : IC_BLOCK * (TILE_F - 1);
    #if TILE_F == 2
        src_unal = lid < (IC_BLOCK - MISALIGNMENT) ? src0 : src1;
    #elif TILE_F == 4
        src_unal = lid < (IC_BLOCK - MISALIGNMENT) ? src0 : src3;
    #endif

        dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + lid_f_offset + output_offset_in_concat_axis), y, x);
        unroll_for(uint tx = 0; tx < TILE_XY; ++tx) {
            OUTPUT_TYPE res_unal = TO_OUTPUT_TYPE(ACTIVATION(((INPUT0_TYPE*)&src_unal)[tx], ACTIVATION_PARAMS));
            output[dst_index + tx * IC_BLOCK] = res_unal;
        }
    } else
#endif  // TILE_F != 1
    {
        const uint dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + lid + output_offset_in_concat_axis), y, x);

        unroll_for(uint fw = 0; fw < TILE_F; ++fw) {
            if (TILE_F != 1 && CEIL_DIV(INPUT0_FEATURE_NUM, IC_BLOCK) % TILE_F != 0 && CEIL_DIV(INPUT0_FEATURE_NUM, IC_BLOCK) % TILE_F == fw)
                break;

            bool do_leftover_write = INPUT0_FEATURE_NUM % IC_BLOCK == 0 || f_block * IC_BLOCK + fw * IC_BLOCK + lid < INPUT0_FEATURE_NUM;
            if (do_leftover_write) {
                unroll_for(uint tx = 0; tx < TILE_XY; ++tx) {
                    INPUT0_TYPE src = input[input_offset + lid + tx * IC_BLOCK + fw * INPUT0_FEATURE_PITCH * IC_BLOCK];
                    OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(src, ACTIVATION_PARAMS));
                    output[dst_index + tx * IC_BLOCK + fw * OUTPUT_FEATURE_PITCH * IC_BLOCK] = res;
                }
            }
        }
    }
#endif
}

#undef WORK_GROUP_SIZE
#undef IC_BLOCK

#undef INPUT_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef INPUT_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef TILE_F
