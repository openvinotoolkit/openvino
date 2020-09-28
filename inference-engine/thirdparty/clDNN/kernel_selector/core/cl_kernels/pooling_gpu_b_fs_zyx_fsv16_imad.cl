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

#define ACTIVATION_VEC16 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 16)
#define TO_ACTIVATION_VEC16 CAT(convert_, ACTIVATION_VEC16)

#define FEATURE_SLICE_SIZE 16

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#else
#error
#endif

inline ACCUMULATOR_TYPE FUNC(apply_pooling)(ACCUMULATOR_TYPE tmp, ACCUMULATOR_TYPE in)
{
#if MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((intel_reqd_sub_group_size(FEATURE_SLICE_SIZE)))
KERNEL(pooling_gpu_b_fs_zyx_fsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_global_id(0);
#if OUTPUT_DIMS == 4
    const uint y   = (uint)get_global_id(1);
    const uint z    = 0;
#else
    const uint zy   = (uint)get_global_id(1);
    const uint y    = zy % OUTPUT_SIZE_Y;
    const uint z    = zy / OUTPUT_SIZE_Y;
#endif
    const uint bf   = (uint)get_global_id(2);
    const uint f    = (bf * FEATURE_SLICE_SIZE) % ALIGN_TO(INPUT0_FEATURE_NUM, FEATURE_SLICE_SIZE);
    const uint b    = (bf * FEATURE_SLICE_SIZE) / ALIGN_TO(INPUT0_FEATURE_NUM, FEATURE_SLICE_SIZE);

    const bool last_in_f_group = (f == FEATURE_SLICE_SIZE * ((INPUT0_FEATURE_NUM - 1) / FEATURE_SLICE_SIZE));

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int offset_z = (int)z*STRIDE_SIZE_Z - PADDING_SIZE_Z;

    ACCUMULATOR_TYPE result[FEATURE_SLICE_SIZE] = { INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL,
                                                    INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL };

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y ||
        offset_z + POOL_SIZE_Z < 0 || offset_z >= INPUT0_SIZE_Z)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elements = 0;
#endif

#if INPUT0_DIMS == 4
    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#else
    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#endif
    __attribute__((opencl_unroll_hint(POOL_SIZE_Z)))
    for(uint pz = 0; pz < POOL_SIZE_Z; pz++)
    {
        int input_offset_z = offset_z + pz;
        bool zero_z = input_offset_z >= INPUT0_SIZE_Z || input_offset_z < 0;
        if(!zero_z)
        {
            __attribute__((opencl_unroll_hint(POOL_SIZE_Y)))
            for(uint py = 0; py < POOL_SIZE_Y; py++)
            {
                int input_offset_y = offset_y + py;
                bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
                if(!zero_y)
                {
                    __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
                    for(uint px = 0; px < POOL_SIZE_X; px++)
                    {
                        int input_offset_x = offset_x + px;
                        bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                        if(!zero)
                        {
                            const uint input_idx = batch_and_feature_offset + input_offset_z*IN_Z_PITCH + input_offset_y*IN_Y_PITCH + input_offset_x*IN_X_PITCH;
                            IN_VEC16 ch16_data;
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                            if (!last_in_f_group) {
#endif
                                ch16_data = AS_TYPE(IN_VEC16, vload4(0, (__global int*)(input + input_idx)));
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                            } else {
                                __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE)))
                                for(uint k = 0; k < INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE; k++) {
                                    ch16_data[k] = input[input_idx + k];
                                }
                            }
#endif

#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                            if (!last_in_f_group) {
#endif
                                __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
                                for(uint k = 0; k < FEATURE_SLICE_SIZE; k++)
                                {
                                    result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
                                }
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                            } else {
                                __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE)))
                                for(uint k = 0; k < INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE; k++)
                                {
                                    result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
                                }
                            }
#endif

        #ifdef DYNAMIC_KERNEL_DIVIDER
                            num_elements++;
        #endif
                        }
                    }
                }
            }
        }
    }
#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int dend = min(offset_z + POOL_SIZE_Z, INPUT0_SIZE_Z + PADDING_SIZE_Z);
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
    const uint num_elements = (dend - offset_z) * (hend - offset_y) * (wend - offset_x);
#endif
#else // !CHECK_BOUNDRY
#if INPUT0_DIMS == 4
    uint input_idx = INPUT0_GET_INDEX(b, f, offset_y, offset_x);
#else
    uint input_idx = INPUT0_GET_INDEX(b, f, offset_z, offset_y, offset_x);
#endif
    __attribute__((opencl_unroll_hint(POOL_SIZE_Z)))
    for(uint pz = 0; pz < POOL_SIZE_Z; pz++)
    {
        __attribute__((opencl_unroll_hint(POOL_SIZE_Y)))
        for(uint py = 0; py < POOL_SIZE_Y; py++)
        {
            __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
            for(uint px = 0; px < POOL_SIZE_X; px++)
            {
                IN_VEC16 ch16_data;
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                if (!last_in_f_group) {
#endif
                    ch16_data = AS_TYPE(IN_VEC16, vload4(0, (__global int*)(input + input_idx)));
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                } else {
                    __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE)))
                    for(uint k = 0; k < INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE; k++) {
                        ch16_data[k] = input[input_idx + k];
                    }
                }
#endif

#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                if (!last_in_f_group) {
#endif
                    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
                    for(uint k = 0; k < FEATURE_SLICE_SIZE; k++)
                    {
                        result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
                    }
#if INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
                } else {
                    __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE)))
                    for(uint k = 0; k < INPUT0_FEATURE_NUM % FEATURE_SLICE_SIZE; k++)
                    {
                        result[k] = FUNC_CALL(apply_pooling)(result[k], ch16_data[k]);
                    }
                }
#endif
                input_idx += IN_X_PITCH;
            }
            input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
        }
        input_idx += (IN_Z_PITCH - POOL_SIZE_Y*IN_Y_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elements = POOL_SIZE_X*POOL_SIZE_Y*POOL_SIZE_Z;
#endif
#endif
    
    ACTIVATION_VEC16 pool_result;
#if defined AVG_POOLING
#if ENABLE_ROUND
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for(uint i = 0; i < FEATURE_SLICE_SIZE; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        pool_result[i] = convert_int(round(((float)result[i] / max(num_elements, (uint)1))));
    #else
        pool_result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
    }
#else
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for(uint i = 0; i < FEATURE_SLICE_SIZE; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        pool_result[i] = (float)result[i] / max(num_elements, (uint)1);
    #else
        pool_result[i] = (float)result[i] / (int)(POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X);
    #endif
    }
#endif  // ENABLE_ROUND
#else  // AVG_POOLING
    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for (uint i = 0; i < FEATURE_SLICE_SIZE; ++i) {
        pool_result[i] = result[i];
    }
#endif  // AVG_POOLING

    OUT_VEC16 final_result = (OUTPUT_TYPE)(0);
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif

    __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
    for (uint i = 0; i < FEATURE_SLICE_SIZE; ++i) {
#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC;
#else
        FUSED_OPS;
#endif
        final_result[i] = FUSED_OPS_RESULT;
#else
        final_result[i] = TO_OUTPUT_TYPE(ACTIVATION(pool_result[i], ACTIVATION_PARAMS));
#endif
    }

#if OUTPUT_DIMS == 4
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#else
    const uint output_pos = OUTPUT_GET_INDEX(b, f, z, y, x);
#endif

#if OUTPUT_TYPE_SIZE == 1
#if OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
    if (!last_in_f_group) {
#endif
        vstore4(as_uint4(final_result), 0, ((__global uint*)(output + output_pos)));
#if OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
    } else {
        __attribute__((opencl_unroll_hint(OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE)))
        for(uint k = 0; k < OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE; k++) {
            output[output_pos + k] = final_result[k];
        }
    }
#endif
#else
#if OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
    if (!last_in_f_group) {
#endif
        *((__global OUT_VEC16*)(output + output_pos)) = final_result;
#if OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE != 0
    } else {
        __attribute__((opencl_unroll_hint(OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE)))
        for(uint k = 0; k < OUTPUT_FEATURE_NUM % FEATURE_SLICE_SIZE; k++) {
            output[output_pos + k] = final_result[k];
        }
    }
#endif
#endif
}

#undef ALIGN_TO
#undef AS_TYPE
#undef IN_VEC16
#undef OUT_VEC16
#undef ACTIVATION_VEC16
#undef TO_ACTIVATION_VEC16
#undef INIT_VAL
#undef FEATURE_SLICE_SIZE
