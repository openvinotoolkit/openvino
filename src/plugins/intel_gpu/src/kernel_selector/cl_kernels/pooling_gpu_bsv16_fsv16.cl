// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define INPUT0_SIZE_Z_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Z + INPUT0_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_Z_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Z + OUTPUT_SIZE_Z + OUTPUT_PAD_AFTER_SIZE_Z)

#define HAS_PAD_Z (PADDING_SIZE_Z != 0)
#define HAS_PAD_Y (PADDING_SIZE_Y != 0)
#define HAS_PAD_X (PADDING_SIZE_X != 0)

#define INPUT_VEC8 MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

#define ACCUMULATOR_VEC8 MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8)
#define TO_ACCUMULATOR_VEC8 CAT(convert_, ACCUMULATOR_VEC8)

#define ACTIVATION_VEC8 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 8)
#define TO_ACTIVATION_VEC8 CAT(convert_, ACTIVATION_VEC8)

#define OUTPUT_VEC8 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)
#define TO_OUTPUT_VEC8 CAT(convert_, OUTPUT_VEC8)

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#endif

inline ACCUMULATOR_VEC8 FUNC(apply_pooling)(ACCUMULATOR_VEC8 tmp, ACCUMULATOR_VEC8 in)
{
#if MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
#if SUB_GROUP_SIZE != 1
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
#endif
KERNEL(pooling_gpu_bsv16_fsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int oc = get_group_id(0) * OC_BLOCK;
    const int sp = get_group_id(1);
    int b = get_group_id(2) * MB_BLOCK;

#if INPUT0_DIMS == 5
    const int z = sp / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const int yx = sp % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const int in_z = z * STRIDE_SIZE_Z - PADDING_SIZE_Z;
#else
    const int z = 0;
    const int in_z = 0;
    const int yx = sp;
#endif
    const int y = yx / OUTPUT_SIZE_X;
    const int x = yx % OUTPUT_SIZE_X;

    int in_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    int in_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    int pool_elementes = 0;

    __global OUTPUT_TYPE *dst_write0 = output
            + b * OUTPUT_FEATURE_NUM * (OUTPUT_SIZE_Z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X)
            + oc * (OUTPUT_SIZE_Z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X) * OC_BLOCK
            + z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X * OC_BLOCK * MB_BLOCK
            + y * OUTPUT_SIZE_X * OC_BLOCK * MB_BLOCK
            + x * OC_BLOCK * MB_BLOCK;

    input += b * INPUT0_FEATURE_NUM * (INPUT0_SIZE_Z_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING * INPUT0_SIZE_X_WITH_PADDING)
            + oc * (INPUT0_SIZE_Z_WITH_PADDING * INPUT0_SIZE_Y_WITH_PADDING * INPUT0_SIZE_X_WITH_PADDING) * IC_BLOCK
            + in_x * IC_BLOCK * MB_BLOCK
            + in_y * INPUT0_SIZE_X_WITH_PADDING * IC_BLOCK * MB_BLOCK
            + in_z * INPUT0_SIZE_Y_WITH_PADDING * INPUT0_SIZE_X_WITH_PADDING * IC_BLOCK * MB_BLOCK;

    ACCUMULATOR_VEC8 blockC00 = (ACCUMULATOR_VEC8)(INIT_VAL);
    ACCUMULATOR_VEC8 blockC01 = (ACCUMULATOR_VEC8)(INIT_VAL);

#if ((HAS_PAD_Z && POOL_SIZE_Z == 1) || (HAS_PAD_Y && POOL_SIZE_Y == 1) || (HAS_PAD_X && POOL_SIZE_X == 1))
    if (!(in_z < 0 || in_z >= INPUT0_SIZE_Z_WITH_PADDING || in_y < 0 || in_y >= INPUT0_SIZE_Y_WITH_PADDING || in_x < 0 || in_x >= INPUT0_SIZE_X_WITH_PADDING)) {
#endif
#if POOL_SIZE_Y != 1 || POOL_SIZE_X != 1 || POOL_SIZE_Z != 1
    unroll_for(int p_z = 0; p_z < POOL_SIZE_Z; ++p_z)
        unroll_for(int p_y = 0; p_y < POOL_SIZE_Y; ++p_y)
            unroll_for(int p_x = 0; p_x < POOL_SIZE_X; ++p_x) {
                if (in_y + p_y < INPUT0_PAD_BEFORE_SIZE_Y || in_y + p_y >= INPUT0_SIZE_Y + INPUT0_PAD_BEFORE_SIZE_Y
                    || in_x + p_x < INPUT0_PAD_BEFORE_SIZE_X
                    || in_x + p_x >= INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X
#if INPUT0_DIMS == 5
                    || in_z + p_z < INPUT0_PAD_BEFORE_SIZE_Z
                    || in_z + p_z >= INPUT0_SIZE_Z + INPUT0_PAD_BEFORE_SIZE_Z) {
#else
                ) {
#endif
                    continue;
                }
                const uint idx = p_z * INPUT0_SIZE_Y_WITH_PADDING * INPUT0_SIZE_X_WITH_PADDING * IC_BLOCK * MB_BLOCK
                                 + p_y * INPUT0_SIZE_X_WITH_PADDING * IC_BLOCK * MB_BLOCK
                                 + p_x * IC_BLOCK * MB_BLOCK;
                const __global INPUT0_TYPE *src1 = input + idx;
#else
                const __global INPUT0_TYPE *src1 = input;
#endif
                INPUT_VEC8 blockA;

                blockA = DT_INPUT_BLOCK_READ8(src1, 0);

                blockC00 = FUNC_CALL(apply_pooling)(blockC00, TO_ACCUMULATOR_VEC8(blockA));

                blockA = DT_INPUT_BLOCK_READ8(src1, 8 * IC_BLOCK);

                blockC01 = FUNC_CALL(apply_pooling)(blockC01, TO_ACCUMULATOR_VEC8(blockA));

                pool_elementes++;

#if POOL_SIZE_Y != 1 || POOL_SIZE_X != 1 || POOL_SIZE_Z != 1
            }
#endif
#if ((HAS_PAD_Z && POOL_SIZE_Z == 1) || (HAS_PAD_Y && POOL_SIZE_Y == 1) || (HAS_PAD_X && POOL_SIZE_X == 1))
    }
#endif

#if defined AVG_POOLING

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    blockC00 /= (ACCUMULATOR_TYPE)max(pool_elementes, (int)1);
    blockC01 /= (ACCUMULATOR_TYPE)max(pool_elementes, (int)1);
#else
    blockC00 /= (ACCUMULATOR_TYPE)POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X;
    blockC01 /= (ACCUMULATOR_TYPE)POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X;
#endif

#endif
    ACTIVATION_VEC8 pool_result;
    OUTPUT_VEC8 final_result;

    #if HAS_FUSED_OPS
    {
        #define BLOCK_NUM 0
        pool_result = TO_ACTIVATION_VEC8(blockC00);
        FUSED_OPS;
        final_result = FUSED_OPS_RESULT;
        DT_OUTPUT_BLOCK_WRITE8(dst_write0, 0, final_result);
        #undef BLOCK_NUM
    }
    {
        #define BLOCK_NUM 1
        pool_result = TO_ACTIVATION_VEC8(blockC01);
        FUSED_OPS;
        final_result = FUSED_OPS_RESULT;
        DT_OUTPUT_BLOCK_WRITE8(dst_write0, 8 * OC_BLOCK, final_result);
        #undef BLOCK_NUM
    }
    #else
        pool_result = TO_ACTIVATION_VEC8(blockC00);
        final_result = TO_OUTPUT_VEC8(ACTIVATION(pool_result, ACTIVATION_PARAMS));
        DT_OUTPUT_BLOCK_WRITE8(dst_write0, 0, final_result);

        pool_result = TO_ACTIVATION_VEC8(blockC01);
        final_result = TO_OUTPUT_VEC8(ACTIVATION(pool_result, ACTIVATION_PARAMS));
        DT_OUTPUT_BLOCK_WRITE8(dst_write0, 8 * OC_BLOCK, final_result);
    #endif
}

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING
#undef INPUT0_SIZE_Z_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
#undef OUTPUT_SIZE_Z_WITH_PADDING

#undef HAS_PAD_Z
#undef HAS_PAD_Y
#undef HAS_PAD_X

#undef INPUT_VEC8

#undef ACCUMULATOR_VEC8
#undef TO_ACCUMULATOR_VEC8

#undef ACTIVATION_VEC8
#undef TO_ACTIVATION_VEC8

#undef OUTPUT_VEC8
#undef TO_OUTPUT_VEC8
