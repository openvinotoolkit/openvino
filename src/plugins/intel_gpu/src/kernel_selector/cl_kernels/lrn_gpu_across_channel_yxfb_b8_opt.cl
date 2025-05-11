// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT_VECTOR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define OUTPUT_VECTOR_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (lrn_gpu_yxfb_b8)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{

    const uint batch_num_group  = (INPUT0_BATCH_NUM/SUB_GROUP_SIZE);
    const uint b_f              = get_global_id(0);
    const uint x                = (uint)get_global_id(1);
    const uint y                = (uint)get_global_id(2);
    const uint feature_id       = b_f / batch_num_group;
    const uint batch_id_group   = b_f % batch_num_group;
    const uint batch_id         = batch_id_group * SUB_GROUP_SIZE;

    const uint input_id = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;
    const uint input_id_group = input_id / SUB_GROUP_SIZE;

    int input_offset_f = feature_id - PADDING;

    const uint input_feature_pitch_group  = (INPUT0_FEATURE_PITCH/SUB_GROUP_SIZE);
    int input_idx_group = (int)input_id_group - PADDING*input_feature_pitch_group;

    INPUT_VECTOR_TYPE acc = 0;

    for (int i = 0; i < LOCAL_SIZE; i++)
    {
        bool zero = input_offset_f < 0 || input_offset_f >= INPUT0_FEATURE_NUM;

        if(!zero)
        {
            INPUT_VECTOR_TYPE value = vload8(input_idx_group, input);
            acc = mad(value, value, acc);
        }

        input_offset_f++;
        input_idx_group += input_feature_pitch_group;
    }
    acc = mad(acc, TO_INPUT0_TYPE(ALPHA_DIV_BY_SIZE), TO_INPUT0_TYPE(K));
    acc = native_powr(acc, -TO_INPUT0_TYPE(BETA));

    const uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
    const uint output_idx_group = output_idx / SUB_GROUP_SIZE;
    float8 _in = vload8(input_id_group, input);
    float8 lrn_result = ACTIVATION(acc * _in, ACTIVATION_PARAMS);

    #if HAS_FUSED_OPS
        FUSED_OPS;
        OUTPUT_VECTOR_TYPE res = FUSED_OPS_RESULT;
        vstore8(res, output_idx_group, output);
    #else
        vstore8(lrn_result, output_idx_group, output);
    #endif
}

#undef INPUT_VECTOR_TYPE
#undef OUTPUT_VECTOR_TYPE
