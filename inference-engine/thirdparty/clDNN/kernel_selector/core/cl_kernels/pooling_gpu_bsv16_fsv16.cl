/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "include/unit_type.cl"
#include "include/include_all.cl"

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define INPUT0_SIZE_Z_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Z + INPUT0_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_Z_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Z + OUTPUT_SIZE_Z + OUTPUT_PAD_AFTER_SIZE_Z)

#define HAS_PAD_Z (PADDING_SIZE_Z != 0)
#define HAS_PAD_Y (PADDING_SIZE_Y != 0)
#define HAS_PAD_X (PADDING_SIZE_X != 0)

#if MAX_POOLING
#define INIT_VAL INPUT0_VAL_MIN
#elif AVG_POOLING
#define INIT_VAL 0
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

inline UNIT_TYPE8 FUNC(apply_pooling)(UNIT_TYPE8 tmp, UNIT_TYPE8 in)
{
#if MAX_POOLING
    return INPUT0_MAX_FUNC(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
KERNEL(pooling_gpu_bsv16_fsv16)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
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

    __global UNIT_TYPE *dst_write0 = output
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

    UNIT_TYPE8 blockC00 = (UNIT_TYPE8)(INIT_VAL);
    UNIT_TYPE8 blockC01 = (UNIT_TYPE8)(INIT_VAL);

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
                const __global UNIT_TYPE *src1 = input + idx;
#else
                const __global UNIT_TYPE *src1 = input;
#endif

                UNIT_TYPE8 blockA;

                blockA = UNIT_BLOCK_READ8(src1, 0);

                blockC00 = FUNC_CALL(apply_pooling)(blockC00, blockA);

                blockA = UNIT_BLOCK_READ8(src1, 8 * IC_BLOCK);

                blockC01 = FUNC_CALL(apply_pooling)(blockC01, blockA);

                pool_elementes++;
#if POOL_SIZE_Y != 1 || POOL_SIZE_X != 1 || POOL_SIZE_Z != 1
            }
#endif
#if ((HAS_PAD_Z && POOL_SIZE_Z == 1) || (HAS_PAD_Y && POOL_SIZE_Y == 1) || (HAS_PAD_X && POOL_SIZE_X == 1))
    }
#endif

#if defined AVG_POOLING

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    blockC00 /= max(pool_elementes, (int)1);
    blockC01 /= max(pool_elementes, (int)1);
#else
    blockC00 /= (POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X);
    blockC01 /= (POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X);
#endif

#endif

    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);

    UNIT_BLOCK_WRITE8(dst_write0, 0, blockC00);
    UNIT_BLOCK_WRITE8(dst_write0, 8 * OC_BLOCK, blockC01);
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

#undef unroll_for
