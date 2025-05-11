// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
    #define INPUT_VAR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, X_BLOCK_SIZE)
    #define OUTPUT_VAR_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, X_BLOCK_SIZE)
    #define ACCUMULATOR_VAR_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, X_BLOCK_SIZE)
    #define ACTIVATION_VAR_TYPE MAKE_VECTOR_TYPE(ACTIVATION_TYPE, X_BLOCK_SIZE)
#else
    #define INPUT_VAR_TYPE INPUT0_TYPE
    #define OUTPUT_VAR_TYPE OUTPUT_TYPE
    #define ACCUMULATOR_VAR_TYPE ACCUMULATOR_TYPE
    #define ACTIVATION_VAR_TYPE ACTIVATION_TYPE
#endif

#define TO_OUTPUT_VAR_TYPE(x) CAT(convert_, OUTPUT_VAR_TYPE)(x)
#define TO_ACCUMULATOR_VAR_TYPE CAT(convert_, ACCUMULATOR_VAR_TYPE)
#define TO_ACTIVATION_VAR_TYPE CAT(convert_, ACTIVATION_VAR_TYPE)

#if   defined MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif defined AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(pooling_gpu_blocked)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int lid = get_sub_group_local_id();
#if SUB_GROUP_SIZE == 16
    const int f_block = get_group_id(1);
    const int f_val = 0;
#else
    const int f_block = (uint)get_group_id(1) / (FEATURE_SLICE_SIZE / SUB_GROUP_SIZE);
    const int f_val = (uint)get_group_id(1) % (FEATURE_SLICE_SIZE / SUB_GROUP_SIZE);
#endif
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * X_BLOCK_SIZE;
    const int y = xy / X_BLOCKS;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              (f_block + input_fs_pad_before) * input_fs_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch +
                              f_val * SUB_GROUP_SIZE;

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (f_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch +
                               f_val * SUB_GROUP_SIZE;


    ACCUMULATOR_VAR_TYPE dst = (ACCUMULATOR_VAR_TYPE)INIT_VAL;

#if AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
    ACCUMULATOR_TYPE count;
    if (lid < X_BLOCK_SIZE)
    {
#if defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        int y_min = max(-PADDING_SIZE_Y, input_y);
        int x_min = max(-PADDING_SIZE_X, input_x + lid*STRIDE_SIZE_X);
        int x_max = min(input_x + lid*STRIDE_SIZE_X + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
        int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
#else
        int y_min = max(0, input_y);
        int x_min = max(0, input_x + lid*STRIDE_SIZE_X);
        int x_max = min(input_x + lid*STRIDE_SIZE_X + POOL_SIZE_X, INPUT0_SIZE_X);
        int y_max = min(input_y + POOL_SIZE_Y, INPUT0_SIZE_Y);
#endif
        count = TO_ACCUMULATOR_TYPE(1.f / (float)((y_max - y_min) * (x_max - x_min)));
    }

    ACCUMULATOR_VAR_TYPE scale;
#if X_BLOCK_SIZE > 1
    for (int i = 0; i < X_BLOCK_SIZE; i++)
        scale[i] = _sub_group_shuffle(count, i);
#else
    scale = _sub_group_shuffle(count, 0);
#endif

#endif

    for (int kh = 0; kh < POOL_SIZE_Y; kh++) {
        if (input_y + kh < 0 || input_y + kh >= INPUT0_SIZE_Y)
            continue;

#if CAN_PRELOAD_FULL_LINE
        INPUT0_TYPE line_cache[INPUT_LINE_SIZE];
        for (int i = 0; i < INPUT_LINE_SIZE; i++) {
            if ((input_x + i) >= 0 && (input_x + i) < INPUT0_SIZE_X)
                line_cache[i] = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + i*input_x_pitch);
            else
                #if defined MAX_POOLING
                    line_cache[i] = INPUT0_VAL_MIN;
                #elif defined AVG_POOLING
                    line_cache[i] = INPUT0_VAL_ZERO;
                #endif
        }

        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for (int kw = 0; kw < POOL_SIZE_X; kw++)
        {
            ACCUMULATOR_VAR_TYPE src;
#if X_BLOCK_SIZE > 1
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                src[i] = TO_ACCUMULATOR_TYPE(line_cache[kw + STRIDE_SIZE_X*i]);
            }
#else
            src = TO_ACCUMULATOR_VAR_TYPE(line_cache[kw]);
#endif

#if defined MAX_POOLING
            dst = ACCUMULATOR_MAX_FUNC(dst, src);
#elif defined AVG_POOLING
            dst += src;
#endif
        }

#else // CAN_PRELOAD_FULL_LINE
        // TODO: try partial preload
        for (int kw = 0; kw < POOL_SIZE_X; kw++)
        {
            INPUT_VAR_TYPE src;
#if X_BLOCK_SIZE > 1
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                if ((input_x + kw + STRIDE_SIZE_X*i) >= 0 && (input_x + kw + STRIDE_SIZE_X*i) < INPUT0_SIZE_X)
                    src[i] = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + (kw + STRIDE_SIZE_X*i)*input_x_pitch);
                else
                    #if defined MAX_POOLING
                        src[i] = INPUT0_VAL_MIN;
                    #elif defined AVG_POOLING
                        src[i] = INPUT0_VAL_ZERO;
                    #endif
            }
#else
            src = DT_INPUT_BLOCK_READ(input, input_offset + kh*input_y_pitch + kw*input_x_pitch);
#endif
#if defined MAX_POOLING
            dst = ACCUMULATOR_MAX_FUNC(dst, src);
#elif defined AVG_POOLING
            dst += TO_ACCUMULATOR_VAR_TYPE(src);
#endif
        }
#endif // CAN_PRELOAD_FULL_LINE
    }
    ACTIVATION_VAR_TYPE pool_result;

#if defined MAX_POOLING
    pool_result = TO_ACTIVATION_VAR_TYPE(dst);
#elif defined AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
    pool_result = TO_ACTIVATION_VAR_TYPE(dst*scale);
#elif defined AVG_POOLING
    pool_result = TO_ACTIVATION_VAR_TYPE(dst/(POOL_SIZE_X*POOL_SIZE_Y));
#endif

#if !HAS_FUSED_OPS
    pool_result = ACTIVATION(pool_result, ACTIVATION_PARAMS);
#endif

    OUTPUT_VAR_TYPE final_result;

#if OUTPUT_LEFTOVERS
    if (f_block*FEATURE_SLICE_SIZE + (f_val + 1)*SUB_GROUP_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            if ((f_block*FEATURE_SLICE_SIZE + f_val*SUB_GROUP_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
#if X_BLOCK_SIZE > 1
            #if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
                final_result[i] = FUSED_OPS_RESULT_SCALAR;
            #else
                final_result[i] = TO_OUTPUT_TYPE(pool_result[i]);
            #endif
                output[output_offset + i * output_x_pitch + lid] = final_result[i];
#else
            #if HAS_FUSED_OPS
                FUSED_OPS_VEC;
                final_result = FUSED_OPS_RESULT_VEC;
            #else
                final_result = TO_OUTPUT_VAR_TYPE(pool_result);
            #endif
                output[output_offset + i * output_x_pitch + lid] = final_result;
#endif
            }
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
    {
        #if HAS_FUSED_OPS
                FUSED_OPS_VEC;
                final_result = FUSED_OPS_RESULT_VEC;
        #else
                final_result = TO_OUTPUT_VAR_TYPE(pool_result);
        #endif

#if SUB_GROUP_SIZE == FEATURE_SLICE_SIZE
        #if X_BLOCK_SIZE == 8
                DT_OUTPUT_BLOCK_WRITE8(output, output_offset, final_result);
        #elif X_BLOCK_SIZE == 4
                DT_OUTPUT_BLOCK_WRITE4(output, output_offset, final_result);
        #elif X_BLOCK_SIZE == 2
                DT_OUTPUT_BLOCK_WRITE2(output, output_offset, final_result);
        #elif X_BLOCK_SIZE == 1
                DT_OUTPUT_BLOCK_WRITE(output, output_offset, final_result);
        #endif
#else
    #if X_BLOCK_SIZE > 1
        __attribute__((opencl_unroll_hint(X_BLOCK_SIZE)))
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, final_result[i]);
        }
    #else
        DT_OUTPUT_BLOCK_WRITE(output, output_offset, final_result);
    #endif
#endif
    }
    else
    {
        const int x_tail = OUTPUT_SIZE_X % X_BLOCK_SIZE;
        for (int i = 0; i < x_tail; i++){
#if X_BLOCK_SIZE > 1
        #if HAS_FUSED_OPS
            FUSED_OPS_SCALAR;
            final_result[i] = FUSED_OPS_RESULT_SCALAR;
        #else
            final_result[i] = TO_OUTPUT_TYPE(pool_result[i]);
        #endif
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, final_result[i]);
#else
        #if HAS_FUSED_OPS
            FUSED_OPS_VEC;
            final_result = FUSED_OPS_RESULT_VEC;
        #else
            final_result = TO_OUTPUT_VAR_TYPE(pool_result);
        #endif
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, final_result);
#endif
        }
    }
}

#undef INIT_VAL
#undef FEATURE_SLICE_SIZE

#undef INPUT_VAR_TYPE
#undef OUTPUT_VAR_TYPE
#undef TO_OUTPUT_VAR_TYPE

#undef ACCUMULATOR_VAR_TYPE

#undef ACTIVATION_VAR_TYPE
#undef TO_ACTIVATION_VAR_TYPE
