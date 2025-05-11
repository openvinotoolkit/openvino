// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

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

KERNEL(pooling_gpu)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#ifdef SELECTED_INDICES_TYPE
, __global SELECTED_INDICES_TYPE* indices
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
#if OUTPUT_LAYOUT_BFYX  || OUTPUT_LAYOUT_BYXF || OUTPUT_LAYOUT_BFZYX ||\
    OUTPUT_LAYOUT_B_FS_ZYX_FSV16 || OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16 || OUTPUT_LAYOUT_B_FS_YX_FSV4
    const uint x    = (uint)get_global_id(0);
#if OUTPUT_DIMS == 5
    const uint y   = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z   = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
#else
    const uint y   = (uint)get_global_id(1);
    const uint z   = 0;
#endif
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;

    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }
#elif OUTPUT_LAYOUT_B_FS_YX_FSV32 || OUTPUT_LAYOUT_B_FS_ZYX_FSV32
    const uint fsv = get_global_id(0);
    const uint zyx = get_global_id(1);
    const uint fsb = get_global_id(2);

    const uint x = zyx % OUTPUT_SIZE_X;
#if OUTPUT_DIMS == 5
    const uint y = zyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const uint z = zyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y;
#else
    const uint y = zyx / OUTPUT_SIZE_X;
    const uint z = 0;
#endif
    const uint fs = fsb % ((OUTPUT_FEATURE_NUM + 32 - 1) / 32);
    const uint b = fsb / ((OUTPUT_FEATURE_NUM + 32 - 1) / 32);
    const uint f = fs * 32 + fsv;

    if (f >= OUTPUT_FEATURE_NUM) {
        return;
    }
#else
    const uint x    = (uint)get_global_id(1);
#if OUTPUT_DIMS == 5
    const uint y    = (uint)get_global_id(2) % OUTPUT_SIZE_Y;
    const uint z    = (uint)get_global_id(2) / OUTPUT_SIZE_Y;
#else
    const uint y    = (uint)get_global_id(2);
    const uint z    = 0;
#endif
    const uint bf   = (uint)get_global_id(0);
    const uint f    = bf / INPUT0_BATCH_NUM;
    const uint b    = bf % INPUT0_BATCH_NUM;
#endif

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int offset_z = (int)z*STRIDE_SIZE_Z - PADDING_SIZE_Z;

    ACCUMULATOR_TYPE result = INIT_VAL;

#ifdef SELECTED_INDICES_TYPE
    uint result_idx = 0;
#endif

#ifdef CHECK_BOUNDARY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y ||
        offset_z + POOL_SIZE_Z < 0 || offset_z >= INPUT0_SIZE_Z)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

#ifndef DILATION_SIZE_X
    #define DILATION_SIZE_X 1
#endif
#ifndef DILATION_SIZE_Y
    #define DILATION_SIZE_Y 1
#endif
#ifndef DILATION_SIZE_Z
    #define DILATION_SIZE_Z 1
#endif

#if OUTPUT_DIMS == 5
    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#else
    const uint batch_and_feature_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#endif

#if OUTPUT_DIMS == 5
    for(uint l = 0; l < POOL_SIZE_Z; l++)
    {
        int input_offset_z = offset_z + (l * DILATION_SIZE_Z);
        bool zero_z = input_offset_z >= INPUT0_SIZE_Z || input_offset_z < 0;
        if (!zero_z)
        {
#endif
            for(uint j = 0; j < POOL_SIZE_Y; j++)
            {
                int input_offset_y = offset_y + (j * DILATION_SIZE_Y);
                bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
                if(!zero_y)
                {
                    for(uint i = 0; i < POOL_SIZE_X; i++)
                    {
                        int input_offset_x = offset_x + (i * DILATION_SIZE_X);
                        bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                        if(!zero)
                        {
#if OUTPUT_DIMS == 5
    #if !INPUT0_SIMPLE
                            const uint input_idx = INPUT0_GET_INDEX(b, f, input_offset_z, input_offset_y, input_offset_x);
    #else
                            const uint input_idx = batch_and_feature_offset + input_offset_z*INPUT0_Z_PITCH + input_offset_y*INPUT0_Y_PITCH + input_offset_x*INPUT0_X_PITCH;
    #endif
#else
    #if !INPUT0_SIMPLE
                            const uint input_idx = INPUT0_GET_INDEX(b, f, input_offset_y, input_offset_x);
    #else
                            const uint input_idx = batch_and_feature_offset + input_offset_y*INPUT0_Y_PITCH + input_offset_x*INPUT0_X_PITCH;
    #endif
#endif
                            const ACCUMULATOR_TYPE casted_input = TO_ACCUMULATOR_TYPE(input[input_idx]);
                            #ifdef SELECTED_INDICES_TYPE
                                if (casted_input > result)
                                {
                                    result = casted_input;
                                    result_idx = input_idx;
                                }
                            #else
                                result = FUNC_CALL(apply_pooling)(result, casted_input);
                            #endif


#ifdef DYNAMIC_KERNEL_DIVIDER
                            num_elementes++;
#endif
                        }
                    }
                }
            }
#if OUTPUT_DIMS == 5
        }
    }
#endif

#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
#if OUTPUT_DIMS == 5
    const int zend = min(offset_z + POOL_SIZE_Z, INPUT0_SIZE_Z + PADDING_SIZE_Z);
    const uint num_elementes = (hend - offset_y) * (wend - offset_x) * (zend - offset_z);
#else
    const uint num_elementes = (hend - offset_y) * (wend - offset_x);
#endif

#endif  // DYNAMIC_WITH_PADDING_KERNEL_DIVIDER

#else  // CHECK_BOUNDARY

#if  OUTPUT_DIMS == 5  // 3D
    uint input_idx = INPUT0_GET_INDEX(b, f, offset_z, offset_y, offset_x);
#else
    uint input_idx = INPUT0_GET_INDEX(b, f, offset_y, offset_x);
#endif

#if OUTPUT_DIMS == 5
    for(uint l = 0; l < POOL_SIZE_Z; l++)
    {
#endif
        for(uint j = 0; j < POOL_SIZE_Y; j++)
        {
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
#if OUTPUT_DIMS == 5
    #if !INPUT0_SIMPLE
                uint input_idx = INPUT0_GET_INDEX(b, f, offset_z + l, offset_y + j, offset_x + i);
                result = FUNC_CALL(apply_pooling)(result, TO_ACCUMULATOR_TYPE(input[input_idx]));
    #else
                result = FUNC_CALL(apply_pooling)(result, TO_ACCUMULATOR_TYPE(input[input_idx]));
                input_idx += INPUT0_X_PITCH;
    #endif
#else
    #if !INPUT0_SIMPLE
                uint input_idx = INPUT0_GET_INDEX(b, f, offset_y + j, offset_x + i);
                result = FUNC_CALL(apply_pooling)(result, TO_ACCUMULATOR_TYPE(input[input_idx]));
    #else
                #ifdef SELECTED_INDICES_TYPE
                    const current_input_value = input[input_idx];
                    if (current_input_value > result)
                    {
                        result = current_input_value;
                        result_idx = input_idx;
                    }
                #else
                    result = FUNC_CALL(apply_pooling)(result, TO_ACCUMULATOR_TYPE(input[input_idx]));
                #endif
                input_idx += INPUT0_X_PITCH;
    #endif
#endif
            }
            input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
        }
#if  OUTPUT_DIMS == 5  // 3D
        input_idx += (INPUT0_Z_PITCH - POOL_SIZE_Y*INPUT0_Y_PITCH);
    }
#endif

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y*POOL_SIZE_Z;
#endif

#endif // CHECK_BOUNDARY

#if defined AVG_POOLING
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        result /= (ACCUMULATOR_TYPE)max(num_elementes, (uint)1);
    #else
        result /= (ACCUMULATOR_TYPE)(POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X);
    #endif
#endif  // defined AVG_POOLING

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE pool_result = TO_ACTIVATION_TYPE(result);

#if HAS_FUSED_OPS
      FUSED_OPS;
      final_result = FUSED_OPS_RESULT;
#else  // HAS_FUSED_OPS
      final_result = TO_OUTPUT_TYPE(ACTIVATION(pool_result, ACTIVATION_PARAMS));
#endif  // HAS_FUSED_OPS

#if OUTPUT_DIMS == 5
    const uint output_pos = OUTPUT_GET_INDEX(b, f, z, y, x);
#else
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#endif
    output[output_pos] = final_result;

#ifdef SELECTED_INDICES_TYPE
    #ifdef INDICES_UPPER_BOUND
        result_idx %= INDICES_UPPER_BOUND;
    #endif
    indices[output_pos] = TO_SELECTED_INDICES_TYPE(result_idx);
#endif
}

#undef INIT_VAL

#undef DILATION_SIZE_X
#undef DILATION_SIZE_Y
#undef DILATION_SIZE_Z
