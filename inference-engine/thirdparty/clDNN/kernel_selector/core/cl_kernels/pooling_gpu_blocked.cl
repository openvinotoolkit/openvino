// Copyright (c) 2018 Intel Corporation
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


#include "include/include_all.cl"
#include "include/unit_type.cl"

#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE > 1
#define vec_t MAKE_VECTOR_TYPE(UNIT_TYPE, X_BLOCK_SIZE)
#else
#define vec_t UNIT_TYPE
#endif

#if   defined MAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif defined AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
#error
#endif

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(pooling_gpu_blocked)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const int lid = get_sub_group_local_id();
    const int f_block = get_group_id(1);
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
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;

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
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;


    vec_t dst = (vec_t)UNIT_INIT_VAL;

#if AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
    UNIT_TYPE count;
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
        count = (UNIT_TYPE)(1.f / (float)((y_max - y_min) * (x_max - x_min)));
    }

    vec_t scale;
#if X_BLOCK_SIZE > 1
    for (int i = 0; i < X_BLOCK_SIZE; i++)
        scale[i] = intel_sub_group_shuffle(count, i);
#else
    scale = intel_sub_group_shuffle(count, 0);
#endif

#endif

    for (int kh = 0; kh < POOL_SIZE_Y; kh++) {
        if (input_y + kh < 0 || input_y + kh >= INPUT0_SIZE_Y)
            continue;

        UNIT_TYPE line_cache[INPUT_LINE_SIZE];
        for (int i = 0; i < INPUT_LINE_SIZE; i++) {
            if ((input_x + i) >= 0 && (input_x + i) < INPUT0_SIZE_X)
                line_cache[i] = UNIT_BLOCK_READ(input, input_offset + kh*input_y_pitch + i*input_x_pitch);
            else
                line_cache[i] = UNIT_INIT_VAL;
        }

        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for (int kw = 0; kw < POOL_SIZE_X; kw++)
        {
            vec_t src;
#if X_BLOCK_SIZE > 1
            for (int i = 0; i < X_BLOCK_SIZE; i++) {
                src[i] = line_cache[kw + STRIDE_SIZE_X*i];
            }
#else
            src = line_cache[kw];
#endif

#if defined MAX_POOLING
            dst = max(dst, src);
#elif defined AVG_POOLING
            dst += src;
#endif
        }
    }

#if defined MAX_POOLING
    dst = ACTIVATION(dst, ACTIVATION_PARAMS);
#elif defined AVG_POOLING && (defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER))
    dst = ACTIVATION((dst*scale), ACTIVATION_PARAMS);
#elif defined AVG_POOLING
    dst = ACTIVATION((dst/(POOL_SIZE_X*POOL_SIZE_Y)), ACTIVATION_PARAMS);
#endif

#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < X_BLOCK_SIZE; i++) {
            if ((f_block*FEATURE_SLICE_SIZE + lid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X)
#if X_BLOCK_SIZE > 1
                output[output_offset + i * output_x_pitch + lid] = dst[i];
#else
                output[output_offset + i * output_x_pitch + lid] = dst;
#endif
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    if (x + X_BLOCK_SIZE <= OUTPUT_SIZE_X)
    {
#if X_BLOCK_SIZE == 8
        UNIT_BLOCK_WRITE8(output, output_offset, dst);
#elif X_BLOCK_SIZE == 4
        UNIT_BLOCK_WRITE4(output, output_offset, dst);
#elif X_BLOCK_SIZE == 2
        UNIT_BLOCK_WRITE2(output, output_offset, dst);
#elif X_BLOCK_SIZE == 1
        UNIT_BLOCK_WRITE(output, output_offset, dst);
#endif
    }
    else
    {
        const int x_tail = OUTPUT_SIZE_X - x;
        for (int i = 0; i < x_tail; i++)
#if X_BLOCK_SIZE > 1
            UNIT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, dst[i]);
#else
            UNIT_BLOCK_WRITE(output, output_offset + i*output_x_pitch, dst);
#endif
    }


}

#undef UNIT_INIT_VAL
#undef FEATURE_SLICE_SIZE
