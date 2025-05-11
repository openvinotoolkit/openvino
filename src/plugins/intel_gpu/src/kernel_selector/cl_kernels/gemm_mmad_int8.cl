// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/mmad.cl"

#define PACK_SIZE               4

#define ACCUMULATOR_TYPE_VEC    CAT(ACCUMULATOR_TYPE, SUB_GROUP_SIZE)
#define ACTIVATION_TYPE_VEC     CAT(ACTIVATION_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT0_TYPE_VEC  CAT(PACKED_INPUT0_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT1_TYPE_VEC  CAT(PACKED_INPUT1_TYPE, SUB_GROUP_SIZE)
#define BLOCK_READ(ptr)         _sub_group_block_read((const __global uint*)(ptr))
#define BLOCK_SHUFFLE           _sub_group_shuffle

#if SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_8x8
#else // SUB_GROUP_SIZE == 8
#define MMAD                    MMAD_16x16
#define TILE_SIZE_M_DIV         (TILE_SIZE_M / 2)
#endif // SUB_GROUP_SIZE == 8

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_mmad_int8.cl : Unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_mmad_int8.cl : Unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_mmad_int8.cl : Unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_mmad_int8.cl : Unsupported output format
#endif // OUTPUT_SIMPLE
}

inline uint FUNC(get_common_input1_offset)(uint batch_offset_input1, uint k, uint i, uint output_x_tile, uint lid) {
#if !TRANSPOSE_INPUT1
    return batch_offset_input1 + (k * TILE_SIZE_K + i * PACK_SIZE) * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;
#else // !TRANSPOSE_INPUT1
    return batch_offset_input1 + (output_x_tile * TILE_SIZE_N + lid) * INPUT1_SIZE_X + k * TILE_SIZE_K + i * PACK_SIZE;
#endif // !TRANSPOSE_INPUT1
}

inline uint FUNC(get_current_input1_offset)(uint common_input1_offset, uint i, uint lid) {
#if !TRANSPOSE_INPUT1
    return common_input1_offset + INPUT1_SIZE_X * i + lid;
#else // !TRANSPOSE_INPUT1
    return common_input1_offset + i;
#endif // !TRANSPOSE_INPUT1
}

inline uint FUNC(get_common_input0_offset)(uint batch_offset_input0, uint k, uint i, uint output_y_tile, uint lid) {
#if !TRANSPOSE_INPUT0
    return batch_offset_input0 + (output_y_tile * TILE_SIZE_M + i) * INPUT0_SIZE_X + k * TILE_SIZE_K;
#else // !TRANSPOSE_INPUT0
    return batch_offset_input0 + (k * TILE_SIZE_K + lid * PACK_SIZE) * INPUT0_SIZE_X + output_y_tile * TILE_SIZE_M + i;
#endif // !TRANSPOSE_INPUT0
}

inline uint FUNC(get_current_input0_offset)(uint common_input0_offset, uint i, uint lid) {
#if !TRANSPOSE_INPUT0
    return common_input0_offset + lid * PACK_SIZE + i;
#else // !TRANSPOSE_INPUT0
    return common_input0_offset + INPUT0_SIZE_X * i;
#endif // !TRANSPOSE_INPUT0
}

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(gemm_mmad_int8)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif // INPUT2_TYPE
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif // HAS_FUSED_OPS_DECLS
    )

// ***************************************************************************************** //
// Kernel with leftovers for all sizes of input matrices and all transposition combinations. //
// ***************************************************************************************** //

#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
{
    const uint output_x = (uint)get_global_id(0);
    const uint output_x_tile = output_x / TILE_SIZE_N;
    const uint output_y_tile = (uint)get_global_id(1);
#if HAS_FUSED_OPS
    uint output_y = output_y_tile * TILE_SIZE_M;
#endif // HAS_FUSED_OPS
    uint batch = get_global_id(2);
    const uint lid = (uint)get_local_id(0);

    const uint z = batch % OUTPUT_SIZE_Z;
    batch /= OUTPUT_SIZE_Z;
    const uint w = batch % OUTPUT_SIZE_W;
    batch /= OUTPUT_SIZE_W;
    const uint f = batch % OUTPUT_FEATURE_NUM;
    batch /= OUTPUT_FEATURE_NUM;
    const uint b = batch % OUTPUT_BATCH_NUM;

    const uint batch_offset_input0 = FUNC_CALL(get_input0_batch_offset)(b, f, w, z);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_batch_offset)(b, f, w, z);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(b, f, w, z);
#endif // INPUT2_TYPE
    const uint batch_offset_output = FUNC_CALL(get_output_batch_offset)(b, f, w, z);

    PACKED_INPUT0_TYPE_VEC tile_input0;
    PACKED_INPUT1_TYPE_VEC tile_input1;
#ifdef INPUT2_TYPE
    ACTIVATION_TYPE_VEC tile_input2;
#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        if (output_y_tile * TILE_SIZE_M + i >= OUTPUT_SIZE_Y) continue;
        if (output_x_tile * TILE_SIZE_N + lid >= OUTPUT_SIZE_X) continue;

        tile_input2[i] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X +
                                                   output_x_tile * TILE_SIZE_N + lid]);
    }
#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        tile_input2[i] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X +
                                                   output_x_tile * TILE_SIZE_N + lid]);
    }
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
#endif // INPUT2_TYPE

    ACCUMULATOR_TYPE_VEC tile_output = (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO);

#if !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = (INPUT0_SIZE_X - 1) / TILE_SIZE_K + 1;
    const uint K_SIZE = INPUT0_SIZE_X;
#else // !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = (INPUT0_SIZE_Y - 1) / TILE_SIZE_K + 1;
    const uint K_SIZE = INPUT0_SIZE_Y;
#endif // !TRANSPOSE_INPUT0

    for (uint k = 0; k < K_BLOCK_NUM; k++) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input0[SUB_GROUP_SIZE];
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1[SUB_GROUP_SIZE];

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            const uint common_input1_offset = FUNC_CALL(get_common_input1_offset)(batch_offset_input1, k, i, output_x_tile, lid);

#if OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
            const uint cur_n = output_x_tile * TILE_SIZE_N + lid;
            const uint cur_k = k * TILE_SIZE_K + i * PACK_SIZE;

            temp_input1[i] = 0;

            if (cur_n < OUTPUT_SIZE_X) {
                if (cur_k + 3 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                    temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
                    temp_input1[i].s3 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 3, lid)];
                } else if (cur_k + 2 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                    temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
                } else if (cur_k + 1 < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                    temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
                } else if (cur_k < K_SIZE) {
                    temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
                }
            }
#else // OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
            temp_input1[i].s0 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 0, lid)];
            temp_input1[i].s1 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 1, lid)];
            temp_input1[i].s2 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 2, lid)];
            temp_input1[i].s3 = input1[FUNC_CALL(get_current_input1_offset)(common_input1_offset, 3, lid)];
#endif // OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K

            tile_input1[i] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1[i]);
        }

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            const uint common_input0_offset = FUNC_CALL(get_common_input0_offset)(batch_offset_input0, k, i, output_y_tile, lid);

#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K
            const uint cur_m = output_y_tile * TILE_SIZE_M + i;
            const uint cur_k = k * TILE_SIZE_K + lid * PACK_SIZE;

            temp_input0[i] = 0;

            if (cur_m < OUTPUT_SIZE_Y) {
                if (cur_k + 3 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                    temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
                    temp_input0[i].s3 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 3, lid)];
                } else if (cur_k + 2 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                    temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
                } else if (cur_k + 1 < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                    temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
                } else if (cur_k < K_SIZE) {
                    temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
                }
            }

            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K

#if !TRANSPOSE_INPUT0
            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ(input0 + common_input0_offset));
#else // !TRANSPOSE_INPUT0
            temp_input0[i].s0 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 0, lid)];
            temp_input0[i].s1 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 1, lid)];
            temp_input0[i].s2 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 2, lid)];
            temp_input0[i].s3 = input0[FUNC_CALL(get_current_input0_offset)(common_input0_offset, 3, lid)];

            tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
#endif // !TRANSPOSE_INPUT0

#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_K
        }

        tile_output = MMAD(tile_input0, tile_input1, tile_output);
    }

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif // HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
#if OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N
        if (output_y_tile * TILE_SIZE_M + i >= OUTPUT_SIZE_Y) continue;
        if (output_x_tile * TILE_SIZE_N + lid >= OUTPUT_SIZE_X) continue;
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N

        ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(tile_output[i]);
        dequantized *= TO_ACTIVATION_TYPE(ALPHA);

#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * tile_input2[i];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS;
#endif // FUSED_OPS_CAN_USE_PRELOAD
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = res;
        output_y++;
#else // HAS_FUSED_OPS
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = dequantized;
#endif // HAS_FUSED_OPS
    }
}

// ******************************************************************************************************************************** //
// Optimized kernel without leftovers (for tiling parameters M = 8, N = 8, K = 32; M = 16, N = 16, K = 64; M = 32, N = 16, K = 64). //
// ******************************************************************************************************************************** //

#else // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K
{
    const uint output_x = (uint)get_global_id(0);
    const uint output_x_tile = output_x / TILE_SIZE_N;
    const uint output_y_tile = (uint)get_global_id(1);
#if HAS_FUSED_OPS
    uint output_y = output_y_tile * TILE_SIZE_M;
#endif // HAS_FUSED_OPS
    uint batch = get_global_id(2);
    const uint lid = (uint)get_local_id(0);

    const uint z = batch % OUTPUT_SIZE_Z;
    batch /= OUTPUT_SIZE_Z;
    const uint w = batch % OUTPUT_SIZE_W;
    batch /= OUTPUT_SIZE_W;
    const uint f = batch % OUTPUT_FEATURE_NUM;
    batch /= OUTPUT_FEATURE_NUM;
    const uint b = batch % OUTPUT_BATCH_NUM;

    const uint batch_offset_input0 = FUNC_CALL(get_input0_batch_offset)(b, f, w, z);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_batch_offset)(b, f, w, z);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(b, f, w, z);
#endif // INPUT2_TYPE
    const uint batch_offset_output = FUNC_CALL(get_output_batch_offset)(b, f, w, z);

    PACKED_INPUT0_TYPE_VEC tile_input00;
    PACKED_INPUT1_TYPE_VEC tile_input10;

#ifdef INPUT2_TYPE
    ACTIVATION_TYPE_VEC tile_input20;

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        tile_input20[i] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X +
                                                    output_x_tile * TILE_SIZE_N + lid]);
    }
#endif // INPUT2_TYPE

    ACCUMULATOR_TYPE_VEC tile_output00 = (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO);
#if TILE_NUM == 2
    ACCUMULATOR_TYPE_VEC tile_output01 = (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO);
#endif // TILE_NUM == 2

#if !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = INPUT0_SIZE_X / TILE_SIZE_K;
#else // !TRANSPOSE_INPUT0
    const uint K_BLOCK_NUM = INPUT0_SIZE_Y / TILE_SIZE_K;
#endif // !TRANSPOSE_INPUT0

    for (uint k = 0; k < K_BLOCK_NUM; k++) {
#if !TRANSPOSE_INPUT1
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1[SUB_GROUP_SIZE];
        const uint common_input1_offset = batch_offset_input1 + k * TILE_SIZE_K * INPUT1_SIZE_X + output_x_tile * TILE_SIZE_N;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            temp_input1[i].s0 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + lid];
            temp_input1[i].s1 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + INPUT1_SIZE_X + lid];
            temp_input1[i].s2 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + 2 * INPUT1_SIZE_X + lid];
            temp_input1[i].s3 = input1[common_input1_offset + i * PACK_SIZE * INPUT1_SIZE_X + 3 * INPUT1_SIZE_X + lid];

            tile_input10[i] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1[i]);
        }
#else // !TRANSPOSE_INPUT1
        const uint common_input1_offset = batch_offset_input1 + output_x_tile * TILE_SIZE_N * INPUT1_SIZE_X + k * TILE_SIZE_K;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            tile_input10[i] = AS_TYPE(PACKED_INPUT1_TYPE, BLOCK_READ(input1 + common_input1_offset  + i * INPUT1_SIZE_X));
        }

        PACKED_INPUT1_TYPE_VEC tile_input1_col0 = BLOCK_SHUFFLE(tile_input10, 0);
        PACKED_INPUT1_TYPE_VEC tile_input1_col1 = BLOCK_SHUFFLE(tile_input10, 1);
        PACKED_INPUT1_TYPE_VEC tile_input1_col2 = BLOCK_SHUFFLE(tile_input10, 2);
        PACKED_INPUT1_TYPE_VEC tile_input1_col3 = BLOCK_SHUFFLE(tile_input10, 3);
        PACKED_INPUT1_TYPE_VEC tile_input1_col4 = BLOCK_SHUFFLE(tile_input10, 4);
        PACKED_INPUT1_TYPE_VEC tile_input1_col5 = BLOCK_SHUFFLE(tile_input10, 5);
        PACKED_INPUT1_TYPE_VEC tile_input1_col6 = BLOCK_SHUFFLE(tile_input10, 6);
        PACKED_INPUT1_TYPE_VEC tile_input1_col7 = BLOCK_SHUFFLE(tile_input10, 7);
#if SUB_GROUP_SIZE == 16
        PACKED_INPUT1_TYPE_VEC tile_input1_col8 = BLOCK_SHUFFLE(tile_input10, 8);
        PACKED_INPUT1_TYPE_VEC tile_input1_col9 = BLOCK_SHUFFLE(tile_input10, 9);
        PACKED_INPUT1_TYPE_VEC tile_input1_col10 = BLOCK_SHUFFLE(tile_input10, 10);
        PACKED_INPUT1_TYPE_VEC tile_input1_col11 = BLOCK_SHUFFLE(tile_input10, 11);
        PACKED_INPUT1_TYPE_VEC tile_input1_col12 = BLOCK_SHUFFLE(tile_input10, 12);
        PACKED_INPUT1_TYPE_VEC tile_input1_col13 = BLOCK_SHUFFLE(tile_input10, 13);
        PACKED_INPUT1_TYPE_VEC tile_input1_col14 = BLOCK_SHUFFLE(tile_input10, 14);
        PACKED_INPUT1_TYPE_VEC tile_input1_col15 = BLOCK_SHUFFLE(tile_input10, 15);
#endif // SUB_GROUP_SIZE == 16

        tile_input10.s0 = tile_input1_col0[lid];
        tile_input10.s1 = tile_input1_col1[lid];
        tile_input10.s2 = tile_input1_col2[lid];
        tile_input10.s3 = tile_input1_col3[lid];
        tile_input10.s4 = tile_input1_col4[lid];
        tile_input10.s5 = tile_input1_col5[lid];
        tile_input10.s6 = tile_input1_col6[lid];
        tile_input10.s7 = tile_input1_col7[lid];
#if SUB_GROUP_SIZE == 16
        tile_input10.s8 = tile_input1_col8[lid];
        tile_input10.s9 = tile_input1_col9[lid];
        tile_input10.sa = tile_input1_col10[lid];
        tile_input10.sb = tile_input1_col11[lid];
        tile_input10.sc = tile_input1_col12[lid];
        tile_input10.sd = tile_input1_col13[lid];
        tile_input10.se = tile_input1_col14[lid];
        tile_input10.sf = tile_input1_col15[lid];
#endif // SUB_GROUP_SIZE == 16

#endif // !TRANSPOSE_INPUT1

#if !TRANSPOSE_INPUT0
        const uint common_input0_offset = batch_offset_input0 + output_y_tile * TILE_SIZE_M * INPUT0_SIZE_X + k * TILE_SIZE_K;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            tile_input00[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ(input0 + common_input0_offset + i * INPUT0_SIZE_X));
        }

        tile_output00 = MMAD(tile_input00, tile_input10, tile_output00);

#if TILE_NUM == 2
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            tile_input00[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ(input0 + common_input0_offset + (TILE_SIZE_M_DIV + i) * INPUT0_SIZE_X));
        }

        tile_output01 = MMAD(tile_input00, tile_input10, tile_output01);
#endif // TILE_NUM == 2

#else // !TRANSPOSE_INPUT0
        MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input0[SUB_GROUP_SIZE];
        const uint common_input0_offset = batch_offset_input0 + (k * TILE_SIZE_K + lid * PACK_SIZE) * INPUT0_SIZE_X + output_y_tile * TILE_SIZE_M;

        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            temp_input0[i].s0 = input0[common_input0_offset + i];
            temp_input0[i].s1 = input0[common_input0_offset + 1 * INPUT0_SIZE_X + i];
            temp_input0[i].s2 = input0[common_input0_offset + 2 * INPUT0_SIZE_X + i];
            temp_input0[i].s3 = input0[common_input0_offset + 3 * INPUT0_SIZE_X + i];

            tile_input00[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
        }

        tile_output00 = MMAD(tile_input00, tile_input10, tile_output00);

#if TILE_NUM == 2
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            temp_input0[i].s0 = input0[common_input0_offset + TILE_SIZE_M_DIV + i];
            temp_input0[i].s1 = input0[common_input0_offset + 1 * INPUT0_SIZE_X + TILE_SIZE_M_DIV + i];
            temp_input0[i].s2 = input0[common_input0_offset + 2 * INPUT0_SIZE_X + TILE_SIZE_M_DIV + i];
            temp_input0[i].s3 = input0[common_input0_offset + 3 * INPUT0_SIZE_X + TILE_SIZE_M_DIV + i];

            tile_input00[i] = AS_TYPE(PACKED_INPUT0_TYPE, temp_input0[i]);
        }

        tile_output01 = MMAD(tile_input00, tile_input10, tile_output01);
#endif // TILE_NUM == 2

#endif // !TRANSPOSE_INPUT0
    }

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif // HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD

    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(tile_output00[i]);
        dequantized *= TO_ACTIVATION_TYPE(ALPHA);
#ifdef INPUT2_TYPE
        dequantized += TO_ACTIVATION_TYPE(BETA) * tile_input20[i];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS;
#endif // FUSED_OPS_CAN_USE_PRELOAD

        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = res;
        output_y++;
#else // HAS_FUSED_OPS
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = dequantized;
#endif // HAS_FUSED_OPS
    }

#if TILE_NUM == 2
    for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
        ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(tile_output01[i]);
        dequantized *= TO_ACTIVATION_TYPE(ALPHA);

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS;
#endif // FUSED_OPS_CAN_USE_PRELOAD

        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + TILE_SIZE_M_DIV + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = res;
        output_y++;
#else // HAS_FUSED_OPS
        output[batch_offset_output + (output_y_tile * TILE_SIZE_M + TILE_SIZE_M_DIV + i) * OUTPUT_SIZE_X + output_x_tile * TILE_SIZE_N + lid] = dequantized;
#endif // HAS_FUSED_OPS
    }
#endif // TILE_NUM == 2

}
#endif // OUTPUT_LEFTOVERS_M || OUTPUT_LEFTOVERS_N || OUTPUT_LEFTOVERS_K

#undef PACK_SIZE
#undef ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef PACKED_INPUT0_TYPE_VEC
#undef PACKED_INPUT1_TYPE_VEC
#undef BLOCK_READ
#undef BLOCK_SHUFFLE
#undef MMAD
#undef TILE_SIZE_M_DIV
