// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#else
    #error No correct pooling mode defined
#endif

#define INPUT_VEC2 MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)

#define ACCUMULATOR_VEC2 MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 2)
#define TO_ACCUMULATOR_VEC2 CAT(convert_, ACCUMULATOR_VEC2)

#define ACTIVATION_VEC2 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 2)
#define TO_ACTIVATION_VEC2 CAT(convert_, ACTIVATION_VEC2)

#define OUTPUT_VEC2 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2)
#define TO_OUTPUT_VEC2 CAT(convert_, OUTPUT_VEC2)

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define INPUT0_SIZE_B_WITH_PADDING (INPUT0_PAD_BEFORE_BATCH_NUM + INPUT0_BATCH_NUM + INPUT0_PAD_AFTER_BATCH_NUM)

#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_B_WITH_PADDING (OUTPUT_PAD_BEFORE_BATCH_NUM + OUTPUT_BATCH_NUM + OUTPUT_PAD_AFTER_BATCH_NUM)

// Kernel works only for sub_group size of 16 with 32 features slice size and process 2 features per WI
#define SUB_GROUP_SIZE 16
#define REQD_FEATURE_SLICE_SIZE 32
#define REQD_FEATURES_PER_WORK_ITEM 2

inline ACCUMULATOR_VEC2 FUNC(apply_pooling)(ACCUMULATOR_VEC2 tmp, ACCUMULATOR_VEC2 in)
{
#if MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pooling_gpu_fs_b_yx_fsv32)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint out_x    = (uint)get_global_id(0);
    const uint out_y    = (uint)get_global_id(1);
    const uint bf       = (uint)get_global_id(2);
    const uint bfs      = bf / (REQD_FEATURE_SLICE_SIZE / REQD_FEATURES_PER_WORK_ITEM);
    const uint sglid    = get_sub_group_local_id();

    const uint b  = bfs % INPUT0_BATCH_NUM;
    const uint fs = bfs / INPUT0_BATCH_NUM;

    ACCUMULATOR_VEC2 results  = (ACCUMULATOR_VEC2)(INIT_VAL,INIT_VAL);

    const uint x_pitch = REQD_FEATURE_SLICE_SIZE;                        // difference in location between (x+1) and (x)
    const uint y_pitch = x_pitch * INPUT0_SIZE_X_WITH_PADDING;           // difference in location between (y+1) and (y)
    const uint b_pitch = y_pitch * INPUT0_SIZE_Y_WITH_PADDING;           // difference in location between (b+1) and (b)
    const uint fs_pitch = b_pitch * INPUT0_SIZE_B_WITH_PADDING;          // difference in location between (fs+1) and (fs)

    const int offset_x = (int)out_x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)out_y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const size_t padding_offset = INPUT0_PAD_BEFORE_SIZE_X * x_pitch +
                                  INPUT0_PAD_BEFORE_SIZE_Y * y_pitch +
                                  INPUT0_PAD_BEFORE_BATCH_NUM * b_pitch +
                                  INPUT0_PAD_BEFORE_FEATURE_NUM / REQD_FEATURE_SLICE_SIZE * fs_pitch;
    const size_t fs_offset = fs * fs_pitch; // locate beginning of feature tile
    const size_t b_offset = b * b_pitch;   // locate beginning of batch

#ifdef CHECK_BOUNDARY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elements = 0;
#endif
    unroll_for(uint in_dy = 0; in_dy < POOL_SIZE_Y; in_dy++)
    {
        if(offset_y + in_dy < INPUT0_SIZE_Y && offset_y + (int)in_dy >= 0)
        {
            const size_t input_offset_y = (offset_y + in_dy) * y_pitch;
            unroll_for(uint in_dx = 0; in_dx < POOL_SIZE_X; in_dx++)
            {
                if(offset_x + in_dx < INPUT0_SIZE_X && offset_x + (int)in_dx >= 0)
                {
                    const size_t input_offset_x = (offset_x + in_dx) * x_pitch;
                    const size_t total_input_offset = padding_offset + fs_offset + b_offset + input_offset_y + input_offset_x;
                    INPUT_VEC2 tmp_input = DT_INPUT_BLOCK_READ2(input, total_input_offset);
                    results  = FUNC_CALL(apply_pooling)(results , TO_ACCUMULATOR_VEC2(tmp_input));

                    #ifdef DYNAMIC_KERNEL_DIVIDER
                        num_elements++;
                    #endif
                }
            }
        }
    }

#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
    const uint num_elements = (hend - offset_y) * (wend - offset_x);
#endif
#else // !CHECK_BOUNDARY
    for(uint in_dy = 0; in_dy < POOL_SIZE_Y; in_dy++)
    {
        const size_t input_offset_y = (offset_y + in_dy) * y_pitch;
        unroll_for(uint in_dx = 0; in_dx < POOL_SIZE_X; in_dx++)
        {
            const size_t input_offset_x = (offset_x + in_dx) * x_pitch;
            const size_t total_input_offset = padding_offset + fs_offset + b_offset + input_offset_y + input_offset_x;
            INPUT_VEC2 tmp_input = DT_INPUT_BLOCK_READ2(input, total_input_offset);
            results = FUNC_CALL(apply_pooling)(results , TO_ACCUMULATOR_VEC2(tmp_input));
        }
    }
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elements = POOL_SIZE_X*POOL_SIZE_Y;
    #endif
#endif

#if defined AVG_POOLING
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        results /= max(num_elements, (uint)1);
    #else
        results /= POOL_SIZE_Y * POOL_SIZE_X;
    #endif
#endif

    const size_t out_x_pitch = REQD_FEATURE_SLICE_SIZE;
    const size_t out_y_pitch = out_x_pitch * OUTPUT_SIZE_X_WITH_PADDING;
    const size_t out_b_pitch = out_y_pitch * OUTPUT_SIZE_Y_WITH_PADDING;
    const size_t out_fs_pitch = out_b_pitch * OUTPUT_SIZE_B_WITH_PADDING;

    const size_t out_pad_before_fs = (OUTPUT_PAD_BEFORE_FEATURE_NUM / REQD_FEATURE_SLICE_SIZE);
    const size_t out_x_offset = (out_x + OUTPUT_PAD_BEFORE_SIZE_X) * out_x_pitch;
    const size_t out_y_offset = (out_y + OUTPUT_PAD_BEFORE_SIZE_Y) * out_y_pitch;
    const size_t out_b_offset = (b + OUTPUT_PAD_BEFORE_BATCH_NUM) * out_b_pitch;
    const size_t out_fs_offset = (fs + out_pad_before_fs) * out_fs_pitch;

    const size_t output_offset = out_fs_offset + out_b_offset + out_y_offset + out_x_offset;

    const bool full_f = OUTPUT_FEATURE_NUM % REQD_FEATURE_SLICE_SIZE == 0 ||
                        fs * REQD_FEATURE_SLICE_SIZE + REQD_FEATURE_SLICE_SIZE <= OUTPUT_FEATURE_NUM;

    OUTPUT_VEC2 final_result;
    ACTIVATION_VEC2 pool_result = TO_ACTIVATION_VEC2(results);

    #if HAS_FUSED_OPS
        FUSED_OPS;
        final_result = FUSED_OPS_RESULT;
    #else
        final_result = TO_OUTPUT_VEC2(ACTIVATION(pool_result , ACTIVATION_PARAMS));
    #endif

    if (full_f)
    {
        DT_OUTPUT_BLOCK_WRITE2(output, output_offset, final_result);
    }
    else
    {
        unroll_for (uint ofi = 0; ofi < REQD_FEATURES_PER_WORK_ITEM; ++ofi)
        {
            if (fs * REQD_FEATURE_SLICE_SIZE + ofi * SUB_GROUP_SIZE + sglid < OUTPUT_FEATURE_NUM)
            {
                output[output_offset + ofi * SUB_GROUP_SIZE + sglid] = (OUTPUT_TYPE)final_result[ofi];
            }
        }
    }
}

#undef FEATURE_SLICE_SIZE
#undef INIT_VAL
#undef INPUT_VEC2

#undef ACCUMULATOR_VEC2
#undef TO_ACCUMULATOR_VEC2

#undef ACTIVATION_VEC2
#undef TO_ACTIVATION_VEC2

#undef OUTPUT_VEC2
#undef TO_OUTPUT_VEC2

#undef INPUT0_SIZE_X_WITH_PADDING
#undef INPUT0_SIZE_Y_WITH_PADDING
#undef INPUT0_SIZE_B_WITH_PADDING

#undef OUTPUT_SIZE_X_WITH_PADDING
#undef OUTPUT_SIZE_Y_WITH_PADDING
#undef OUTPUT_SIZE_B_WITH_PADDING

#undef SUB_GROUP_SIZE
#undef REQD_FEATURE_SLICE_SIZE
#undef REQD_FEATURES_PER_WORK_ITEM
