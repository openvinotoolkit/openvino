// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define READ_VEC(offset, ptr)          CAT(vload, VEC_SIZE)(offset, ptr)
#define WRITE_VEC(val, offset, ptr)    CAT(vstore, VEC_SIZE)(val, offset, ptr)

#define IN_VEC_TYPE                     MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define TO_IN_VEC_TYPE(x)               CAT(convert_, IN_VEC_TYPE)(x)
#define ACC_VEC_TYPE                    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)
#define TO_ACC_VEC_TYPE(x)              CAT(convert_, ACC_VEC_TYPE)(x)
#define OUT_VEC_TYPE                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define TO_OUT_VEC_TYPE(x)              CAT(convert_, OUT_VEC_TYPE)(x)

#if defined (PERMUTE_SIMPLE_MEM_COPY)
KERNEL (permute_f_y_axes)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const int bf = get_global_id(2);
    const int f_idx = bf % INPUT0_FEATURE_NUM;
    const int b_idx = bf / INPUT0_FEATURE_NUM;
    const int x_start = get_global_id(0) * BLOCK_SIZE;
    const int y_idx = get_global_id(1);

    __attribute__((opencl_unroll_hint(J_TIMES)))
    for (int j = 0; j < J_TIMES; ++j) {
        const int x_idx = x_start + j * VEC_SIZE;
        const int f_out_idx = get_global_id(1);
        const int y_out_idx = bf % INPUT0_FEATURE_NUM;
#if HAS_FUSED_OPS
        OUT_VEC_TYPE result;
        IN_VEC_TYPE res = READ_VEC(0, &input[INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx)]);
        FUSED_OPS_VEC;
        result = FUSED_OPS_RESULT_VEC;
#else
        IN_VEC_TYPE res = READ_VEC(0, &input[INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx)]);
        OUT_VEC_TYPE result = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
        const int output_idx = OUTPUT_GET_INDEX(b_idx, f_out_idx, y_out_idx, x_idx);
        WRITE_VEC(result, 0, &output[output_idx]);
    }
}

#elif defined (THREE_DIM_TRANSPOSE)

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
#endif
KERNEL (permute_f_y_axes)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    __local OUTPUT_TYPE transpose_buf[FEATURE_BLOCK_SIZE][FEATURE_BLOCK_SIZE][TILE_SIZE];
    const int bf = get_global_id(2);
    const int f_idx = bf % INPUT0_FEATURE_NUM;
    const int b_idx = bf / INPUT0_FEATURE_NUM;

    int bf_local = get_local_id(2);
    int y_local = get_local_id(1);

    const int x_begin = get_global_id(0) * TILE_SIZE;
    const int y_begin = get_global_id(1) * FEATURE_BLOCK_SIZE;
    const int f_begin = get_local_size(2) * get_group_id(2);

    __attribute__((opencl_unroll_hint(FEATURE_BLOCK_SIZE)))
    for (int j = 0; j < FEATURE_BLOCK_SIZE; ++j) {
        __attribute__((opencl_unroll_hint(TILE_SIZE)))
        for (int i = 0; i < TILE_SIZE; ++i) {
            const int x_idx = x_begin + i;
            const int y_idx = y_begin + j;
            const uint input_offset = INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx) - get_sub_group_local_id();
            INPUT0_TYPE res = DT_INPUT_BLOCK_READ(input, input_offset);
    #if HAS_FUSED_OPS
            const int f_out_idx = y_begin + j;
            const int y_out_idx = bf % INPUT0_FEATURE_NUM;
            FUSED_OPS;
            transpose_buf[bf_local][j][i]  = FUSED_OPS_RESULT;
    #else
            transpose_buf[bf_local][j][i] = ACTIVATION(res, ACTIVATION_PARAMS);
    #endif
        }
    }

    __attribute__((opencl_unroll_hint(FEATURE_BLOCK_SIZE)))
    for (int j = 0; j < FEATURE_BLOCK_SIZE; ++j) {
        __attribute__((opencl_unroll_hint(TILE_SIZE)))
        for (int i = 0; i < TILE_SIZE; ++i) {
            const int x_idx = x_begin + i;
            const int f = (f_begin + j) % INPUT0_FEATURE_NUM;
            const int y_idx = y_begin + bf_local;
            const uint output_offset = OUTPUT_GET_INDEX(b_idx, y_idx, f, x_idx) - get_sub_group_local_id();
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, transpose_buf[j][bf_local][i]);
        }
    }
}

#else

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
#endif
KERNEL (permute_f_y_axes)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    __local OUTPUT_TYPE transpose_buf[TILE_SIZE][TILE_SIZE+1];

    const int bf = get_global_id(2);
    const int b_idx = bf / INPUT0_FEATURE_NUM;
    const int f_idx = bf % INPUT0_FEATURE_NUM;
    const int bf_local = get_local_id(2);
    const int x_idx = get_global_id(0);
    const int y_begin = get_global_id(1) * TILE_SIZE;
    const int f_begin = get_local_size(2) * get_group_id(2);
#if INPUT0_SIMPLE == 1
    __attribute__((opencl_unroll_hint(J_TIMES)))
    for (int j = 0; j < J_TIMES; ++j) {
        const int j_vec = j * VEC_SIZE;
        const int y_idx = y_begin + j_vec;
#if HAS_FUSED_OPS
        IN_VEC_TYPE res = READ_VEC(0, &input[INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx)]);
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int k = 0; k < VEC_SIZE; ++k) {
            transpose_buf[j_vec + k][bf_local] = res[k];
        }
#else
        IN_VEC_TYPE res = READ_VEC(0, &input[INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx)]);
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (int k = 0; k < VEC_SIZE; ++k) {
            transpose_buf[j_vec + k][bf_local] = ACTIVATION(res[k], ACTIVATION_PARAMS);
        }
#endif
    }
#if HAS_FUSED_OPS
    __attribute__((opencl_unroll_hint(J_TIMES)))
    for (int j = 0; j < J_TIMES; ++j) {
        const int j_vec = j * VEC_SIZE;
        OUT_VEC_TYPE res = READ_VEC(0, &transpose_buf[bf_local][j_vec]);
        const int f_out_idx = y_begin + bf_local;
        const int y_out_idx = (f_begin + j_vec) % INPUT0_FEATURE_NUM;;
        FUSED_OPS_VEC;
        OUT_VEC_TYPE result = FUSED_OPS_RESULT_VEC;
        const int output_idx = OUTPUT_GET_INDEX(b_idx, f_out_idx, y_out_idx, x_idx);
        WRITE_VEC(result, 0 , &output[output_idx]);
    }
#else
    __attribute__((opencl_unroll_hint(J_TIMES)))
    for (int j = 0; j < J_TIMES; ++j) {
        const int j_vec = j * VEC_SIZE;
        const int f = (f_begin + j_vec) % INPUT0_FEATURE_NUM;;
        const int y_idx = y_begin + bf_local;
        const int output_idx = OUTPUT_GET_INDEX(b_idx, y_idx, f, x_idx);
        WRITE_VEC(READ_VEC(0, &transpose_buf[bf_local][j_vec]), 0, &output[output_idx]);
    }
#endif

#else
    __attribute__((opencl_unroll_hint(TILE_SIZE)))
    for (int j = 0; j < TILE_SIZE; ++j) {
        int y_idx = y_begin + j;
        const uint input_offset = INPUT0_GET_INDEX(b_idx, f_idx, y_idx, x_idx) - get_sub_group_local_id();
        INPUT0_TYPE res = DT_INPUT_BLOCK_READ(input, input_offset);
#if HAS_FUSED_OPS
        const int y_out_idx = bf % INPUT0_FEATURE_NUM;
        const int f_out_idx = y_begin + j;
        FUSED_OPS;
        transpose_buf[bf_local][j]  = FUSED_OPS_RESULT;
#else
        transpose_buf[bf_local][j] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
    }
    __attribute__((opencl_unroll_hint(TILE_SIZE)))
    for (int j = 0; j < TILE_SIZE; ++j) {
        const int f = (f_begin + j) % INPUT0_FEATURE_NUM;
        const int y_idx = y_begin + bf_local;
        const uint output_offset = OUTPUT_GET_INDEX(b_idx, y_idx, f, x_idx) - get_sub_group_local_id();
        DT_OUTPUT_BLOCK_WRITE(output, output_offset, transpose_buf[j][bf_local]);
    }
#endif
}

#endif // #if defined (SIMPLE_MEM_COPY)
