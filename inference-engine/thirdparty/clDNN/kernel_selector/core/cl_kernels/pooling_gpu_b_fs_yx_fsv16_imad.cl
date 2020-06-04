// Copyright (c) 2020 Intel Corporation
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
#include "include/data_types.cl"

#define ALIGN_TO(val, multiple) (((val) + (multiple) - 1) / (multiple) * (multiple))

#define AS_TYPE(type, val) CAT(as_, type)(val)
#define IN_VEC16 MAKE_VECTOR_TYPE(INPUT0_TYPE, 16)
#define OUT_VEC16 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 16)
#define CONVERT_OUT CAT(convert_, OUTPUT_TYPE)
#define CONVERT_OUT_VEC16 CAT(convert_, OUT_VEC16)
#define FEATURE_SLICE_SIZE 16

#if MAX_POOLING
    #define INIT_VAL CHAR_MIN
#elif AVG_POOLING
    #define INIT_VAL 0
#else
#error
#endif


inline int FUNC(apply_pooling)(int tmp, int in)
{
#if MAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(pooling_gpu_b_fs_yx_fsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = (bf * FEATURE_SLICE_SIZE) % ALIGN_TO(INPUT0_FEATURE_NUM, FEATURE_SLICE_SIZE);
    const uint b    = (bf * FEATURE_SLICE_SIZE) / ALIGN_TO(INPUT0_FEATURE_NUM, FEATURE_SLICE_SIZE);

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    int result[FEATURE_SLICE_SIZE] = { INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL,
                                       INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL };

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elements = 0;
#endif

    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0);
    __attribute__((opencl_unroll_hint(POOL_SIZE_Y)))
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
        if(!zero_y)
        {
            __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                if(!zero)
                {
                    const uint input_idx = batch_and_feature_offset + input_offset_y*IN_Y_PITCH + input_offset_x*IN_X_PITCH;

                    int4 int_data = vload4(0, (__global int*)(input + input_idx));
                    IN_VEC16 ch16_data = AS_TYPE(IN_VEC16, int_data);
                    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
                    for(uint k = 0; k < FEATURE_SLICE_SIZE; k++)
                    {
                        result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
                    }

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
    uint input_idx = INPUT0_GET_INDEX(b, f, offset_y, offset_x);
    __attribute__((opencl_unroll_hint(POOL_SIZE_Y)))
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int4 int_data = vload4(0, (__global int*)(input + input_idx));
            IN_VEC16 ch16_data = AS_TYPE(IN_VEC16, int_data);
            __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
            for(uint k = 0; k < FEATURE_SLICE_SIZE; k++)
            {
                result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
            }

            input_idx += IN_X_PITCH;
        }
        input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elements = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
#if ENABLE_ROUND
    int16 pool_result;
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for(uint i = 0; i < FEATURE_SLICE_SIZE; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        pool_result[i] = convert_int(round(((float)result[i] / max(num_elements, (uint)1))));
    #else
        pool_result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
    }
#else
    float16 pool_result;
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for(uint i = 0; i < FEATURE_SLICE_SIZE; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        pool_result[i] = (float)result[i] / max(num_elements, (uint)1);
    #else
        pool_result[i] = (float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X);
    #endif
    }
#endif  // ENABLE_ROUND
#else  // AVG_POOLING
    int16 pool_result;
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for (uint i = 0; i < FEATURE_SLICE_SIZE; ++i) {
        pool_result[i] = result[i];
    }
#endif  // AVG_POOLING

OUT_VEC16 final_result = (OUTPUT_TYPE)(0);
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    //FUSED_OPS_LOAD_PER_SCALE
    FUSED_OPS_PRELOAD
#endif
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for (uint i = 0; i < FEATURE_SLICE_SIZE; ++i) {
#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC
#else
        FUSED_OPS
#endif
        final_result[i] = FUSED_OPS_RESULT;
#else
        final_result[i] = pool_result[i];
#endif
    }
    
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);

#if OUTPUT_TYPE_SIZE == 1
    vstore4(as_uint4(final_result), 0, ((__global uint*)(output + output_pos)));
#else
    *((__global OUT_VEC16*)(output + output_pos)) = final_result;
#endif
}

#undef ALIGN_TO
#undef AS_TYPE
#undef IN_VEC16
#undef OUT_VEC16
#undef CONVERT_OUT
#undef CONVERT_OUT_VEC16
#undef INIT_VAL
#undef FEATURE_SLICE_SIZE