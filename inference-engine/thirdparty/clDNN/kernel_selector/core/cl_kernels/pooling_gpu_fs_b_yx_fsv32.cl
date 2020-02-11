// Copyright (c) 2019 Intel Corporation
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

#if MAX_POOLING
    #define INIT_VAL UNIT_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL 0
#else
#error No correct pooling mode defined
#endif

#if defined(USE_FLOAT_ACC)
    #define ACC_TYPE2 float2
    #define READ_BLOCK2_INPUT(input, input_total_offset) convert_float2(UNIT_BLOCK_READ2(input,total_input_offset))
    #define TO_UNIT_BLOCK2(values) convert_half2(values)
#else
    #define ACC_TYPE2 UNIT_TYPE2
    #define READ_BLOCK2_INPUT(input, input_total_offset) UNIT_BLOCK_READ2(input,total_input_offset)
    #define TO_UNIT_BLOCK2(values) values
#endif

#define INPUT0_SIZE_X_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define INPUT0_SIZE_Y_WITH_PADDING (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define OUTPUT_SIZE_X_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OUTPUT_SIZE_Y_WITH_PADDING (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)

// Kernel works only for sub_group size of 16 with 32 features slice size and process 2 features per WI
#define REQD_SUB_GROUP_SIZE 16
#define REQD_FEATURE_SLICE_SIZE 32
#define REQD_FEATURES_PER_WORK_ITEM 2

#define unroll_for __attribute__((opencl_unroll_hint)) for

inline ACC_TYPE2 FUNC(apply_pooling)(ACC_TYPE2 tmp, ACC_TYPE2 in)
{
#if MAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((intel_reqd_sub_group_size(REQD_SUB_GROUP_SIZE)))
KERNEL(pooling_gpu_fs_b_yx_fsv32)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    const uint out_x    = (uint)get_global_id(0);
    const uint out_y    = (uint)get_global_id(1);
    const uint bf       = (uint)get_global_id(2);
    const uint bfs      = bf / (REQD_FEATURE_SLICE_SIZE / REQD_FEATURES_PER_WORK_ITEM);
    const uint sglid    = get_sub_group_local_id();

    const uint b  = bfs % INPUT0_BATCH_NUM;
    const uint fs = bfs / INPUT0_BATCH_NUM;

    ACC_TYPE2 results = (ACC_TYPE2)(INIT_VAL,INIT_VAL);

    const uint x_pitch = REQD_FEATURE_SLICE_SIZE;                        // difference in location between (x+1) and (x)
    const uint y_pitch = x_pitch * INPUT0_SIZE_X_WITH_PADDING;           // difference in location between (y+1) and (y)
    const uint b_pitch = y_pitch * INPUT0_SIZE_Y_WITH_PADDING;           // difference in location between (b+1) and (b)
    const uint fs_pitch = b_pitch * INPUT0_BATCH_NUM;                     // difference in location between (fs+1) and (fs)

    const int offset_x = (int)out_x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)out_y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const size_t padding_offset = INPUT0_PAD_BEFORE_SIZE_X * x_pitch + INPUT0_PAD_BEFORE_SIZE_Y * y_pitch;
    const size_t fs_offset = fs * fs_pitch; // locate beginning of feature tile
    const size_t b_offset = b * b_pitch;   // locate beginning of batch
#ifdef CHECK_BOUNDRY
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

                    ACC_TYPE2 tmp_input = READ_BLOCK2_INPUT(input, input_total_offset);
                    
                    results = FUNC_CALL(apply_pooling)(results, tmp_input);

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
#else // !CHECK_BOUNDRY
    for(uint in_dy = 0; in_dy < POOL_SIZE_Y; in_dy++)
    {
        const size_t input_offset_y = (offset_y + in_dy) * y_pitch;
        unroll_for(uint in_dx = 0; in_dx < POOL_SIZE_X; in_dx++)
        {
            const size_t input_offset_x = (offset_x + in_dx) * x_pitch;
            const size_t total_input_offset = padding_offset + fs_offset + b_offset + input_offset_y + input_offset_x;

            ACC_TYPE2 tmp_input = READ_BLOCK2_INPUT(input, input_total_offset);

            results = FUNC_CALL(apply_pooling)(results, tmp_input);
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

    results = ACTIVATION(results, ACTIVATION_PARAMS);

    const size_t out_x_pitch = REQD_FEATURE_SLICE_SIZE;
    const size_t out_y_pitch = out_x_pitch * OUTPUT_SIZE_X_WITH_PADDING;
    const size_t out_b_pitch = out_y_pitch * OUTPUT_SIZE_Y_WITH_PADDING;
    const size_t out_fs_pitch = out_b_pitch * OUTPUT_BATCH_NUM;

    const size_t out_pad_before_fs = (OUTPUT_PAD_BEFORE_FEATURE_NUM / REQD_FEATURE_SLICE_SIZE);
    const size_t out_x_offset = (out_x + OUTPUT_PAD_BEFORE_SIZE_X) * out_x_pitch;
    const size_t out_y_offset = (out_y + OUTPUT_PAD_BEFORE_SIZE_Y) * out_y_pitch;
    const size_t out_b_offset = b * out_b_pitch;
    const size_t out_fs_offset = (fs + out_pad_before_fs) * out_fs_pitch;


    const size_t output_offset = out_fs_offset + out_b_offset + out_y_offset + out_x_offset;

    const bool full_f = OUTPUT_FEATURE_NUM % REQD_FEATURE_SLICE_SIZE == 0 ||
                        fs * REQD_FEATURE_SLICE_SIZE + REQD_FEATURE_SLICE_SIZE <= OUTPUT_FEATURE_NUM;

    if (full_f)
    {
        UNIT_BLOCK_WRITE2(output, output_offset, TO_UNIT_BLOCK2(results));
    }
    else
    {
        unroll_for (uint ofi = 0; ofi < REQD_FEATURES_PER_WORK_ITEM; ++ofi)
        {
            if (fs * REQD_FEATURE_SLICE_SIZE + ofi * REQD_SUB_GROUP_SIZE + sglid < OUTPUT_FEATURE_NUM)
            {
                output[output_offset + ofi * REQD_SUB_GROUP_SIZE + sglid] = (UNIT_TYPE)results[ofi];
            }
        }
    }
}

#undef TO_UNIT_BLOCK2
#undef READ_BLOCK2_INPUT
#undef ACC_TYPE2
#undef FEATURE_SLICE_SIZE
#undef INIT_VAL
