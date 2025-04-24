// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef FORCE_SIMD_16
REQD_SUB_GROUP_SIZE(16)
#endif

KERNEL (lrn_gpu_across_channel_multiple_features)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if defined OUTPUT_LAYOUT_BFYX || defined OUTPUT_LAYOUT_B_FS_YX_FSV4
// PERF NOTE: SIMD IS OVER global_id(0) so in SIMD global_id(1) and global_id(2) does not change, so we can use group_id to have SIMD1 instructions
    const uint x            = get_global_id(0);
    const uint y            = get_group_id(1);
    const uint b_f          = get_group_id(2);
    const uint batch_id     = (b_f * OFM_PER_SIMD) / INPUT0_FEATURE_NUM;
    const uint feature_id   = (b_f % (INPUT0_FEATURE_NUM / OFM_PER_SIMD)) * OFM_PER_SIMD;

    if (x >= INPUT0_SIZE_X)
        return;
#elif defined OUTPUT_LAYOUT_YXFB
    const uint b_f          = get_global_id(0);
    const uint x            = get_group_id(1);
    const uint y            = get_group_id(2);
    const uint feature_id   = (b_f / INPUT0_BATCH_NUM) * OFM_PER_SIMD;
    const uint batch_id     = b_f % INPUT0_BATCH_NUM;
#endif

    uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

#if INPUT0_SIMPLE
    uint input_idx = input_id - PADDING*INPUT0_FEATURE_PITCH;
    input_idx =  MULTIPLY_OFFSET(INPUT0_TYPE, input_idx);
#endif

    int input_offset_f = feature_id - PADDING;

    INPUT0_TYPE vals[OFM_PER_SIMD];
    INPUT0_TYPE results[OFM_PER_SIMD] = { INPUT0_VAL_ZERO };

    // prefetch
    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;
    #if !INPUT0_SIMPLE
        uint input_idx = INPUT0_GET_INDEX(batch_id, feature_id - PADDING + j, y, x);
        input_idx =  MULTIPLY_OFFSET(INPUT0_TYPE, input_idx);
        vals[j] = zero ? INPUT0_VAL_ZERO : TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(INPUT0_TYPE, input, input_idx));
    #else
        vals[j] = zero ? INPUT0_VAL_ZERO : TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(INPUT0_TYPE, input, input_idx));
        input_idx += MULTIPLY_OFFSET(INPUT0_VAL_ZERO, INPUT0_FEATURE_PITCH);
    #endif
        input_offset_f++;
    }

    for (uint j = 0; j < LOCAL_SIZE-1; j++)
    {
        for(uint i = 0; i < OFM_PER_SIMD; i++)
        {
            results[i] = mad(vals[i], vals[i], results[i]);
        }
        for(uint i = 0; i < OFM_PER_SIMD-1; i++)
        {
            vals[i] = vals[i+1];
        }
    #if !INPUT0_SIMPLE
        uint input_idx = INPUT0_GET_INDEX(batch_id, input_offset_f, y, x);
        input_idx =  MULTIPLY_OFFSET(INPUT0_TYPE, input_idx);
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;
        vals[OFM_PER_SIMD-1] = zero ? INPUT0_VAL_ZERO : TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(INPUT0_TYPE, input, input_idx));
    #else
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;
        vals[OFM_PER_SIMD-1] = zero ? INPUT0_VAL_ZERO : TO_INPUT0_TYPE(ALPHA_VAL_FACTOR_DIV_BY_SIZE) * (*OFFSET_GLOBAL_PTR(INPUT0_TYPE, input, input_idx));
        input_idx += MULTIPLY_OFFSET(INPUT0_TYPE, INPUT0_FEATURE_PITCH);
    #endif
        input_offset_f++;
    }

    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
        results[j] = mad(vals[j], vals[j], results[j]);
        results[j] = mad(results[j], TO_INPUT0_TYPE(ALPHA_DIV_BY_SIZE), TO_INPUT0_TYPE(K));
        results[j] = native_powr(results[j], -TO_INPUT0_TYPE(BETA));
    }

    #if OUTPUT_SIMPLE
        uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    #endif

    INPUT0_TYPE lrn_result;

    for(uint j = 0; j < OFM_PER_SIMD; j++)
    {
    #if !OUTPUT_SIMPLE
        uint output_idx = OUTPUT_GET_INDEX(batch_id, feature_id + j, y, x);
        input_id = INPUT0_GET_INDEX(batch_id, feature_id + j, y, x);
    #endif
        lrn_result = results[j] * input[input_id];
        #if HAS_FUSED_OPS
            FUSED_OPS;
            output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
        #else
            output[output_idx] = ACTIVATION(lrn_result, ACTIVATION_PARAMS);
        #endif
    #if OUTPUT_SIMPLE
        output_idx += OUTPUT_FEATURE_PITCH;
        input_id += INPUT0_FEATURE_PITCH;
    #endif
    }
}
