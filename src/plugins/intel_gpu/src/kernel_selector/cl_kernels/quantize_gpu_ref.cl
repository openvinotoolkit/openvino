// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
#endif
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
KERNEL(quantize_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* input_low,
    const __global INPUT2_TYPE* input_high,
    const __global INPUT3_TYPE* output_low,
    const __global INPUT4_TYPE* output_high,
          __global OUTPUT_TYPE* output)
{
    const int b = get_global_id(0);
    const int of = get_global_id(1);
#if OUTPUT_DIMS <= 4
    const int yx = get_global_id(2);
    const int x = yx % OUTPUT_SIZE_X;
    const int y = yx / OUTPUT_SIZE_X;
    const int z = 0;
#elif OUTPUT_DIMS == 5
    const int zyx = get_global_id(2);
    const int x = zyx % OUTPUT_SIZE_X;
    const int y = (zyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const int z = (zyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
#elif OUTPUT_DIMS == 6
    const int wzyx = get_global_id(2);
    const int x = wzyx % OUTPUT_SIZE_X;
    const int y = (wzyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const int z = ((wzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const int w = ((wzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z;
#elif OUTPUT_DIMS == 7
    const int uwzyx = get_global_id(2);
    const int x = uwzyx % OUTPUT_SIZE_X;
    const int y = (uwzyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const int z = ((uwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const int w = ((uwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const int u = ((uwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z / OUTPUT_SIZE_W;
#elif OUTPUT_DIMS == 8
    const int vuwzyx = get_global_id(2);
    const int x = vuwzyx % OUTPUT_SIZE_X;
    const int y = (vuwzyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const int z = ((vuwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z;
    const int w = ((vuwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const int u = ((vuwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z / OUTPUT_SIZE_W % OUTPUT_SIZE_U;
    const int v = ((vuwzyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y) / OUTPUT_SIZE_Z / OUTPUT_SIZE_W / OUTPUT_SIZE_U;
#endif

#if INPUT0_DIMS == 8
    const int input_offset = INPUT0_GET_INDEX(b, of, v, u, w, z, y, x);
#elif INPUT0_DIMS == 7
    const int input_offset = INPUT0_GET_INDEX(b, of, u, w, z, y, x);
#elif INPUT0_DIMS == 6
    const int input_offset = INPUT0_GET_INDEX(b, of, w, z, y, x);
#elif INPUT0_DIMS == 5
    const int input_offset = INPUT0_GET_INDEX(b, of, z, y, x);
#elif INPUT0_DIMS <= 4
    const int input_offset = INPUT0_GET_INDEX(b, of, y, x);
#endif

#if OUTPUT_DIMS == 8
    const int output_offset = OUTPUT_GET_INDEX(b, of, v, u, w, z, y, x);
#elif OUTPUT_DIMS == 7
    const int output_offset = OUTPUT_GET_INDEX(b, of, u, w, z, y, x);
#elif OUTPUT_DIMS == 6
    const int output_offset = OUTPUT_GET_INDEX(b, of, w, z, y, x);
#elif OUTPUT_DIMS == 5
    const int output_offset = OUTPUT_GET_INDEX(b, of, z, y, x);
#elif OUTPUT_DIMS <= 4
    const int output_offset = OUTPUT_GET_INDEX(b, of, y, x);
#endif

#if INPUT1_DIMS == 8
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#elif INPUT1_DIMS == 7
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT1_DIMS == 6
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT1_DIMS == 5
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT1_DIMS <= 4
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT2_DIMS == 8
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#elif INPUT2_DIMS == 7
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT2_DIMS == 6
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT2_DIMS == 5
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT2_DIMS <= 4
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT3_DIMS == 8
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#elif INPUT3_DIMS == 7
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT3_DIMS == 6
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT3_DIMS == 5
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT3_DIMS <= 4
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT4_DIMS == 8
    const int output_high_offset = INPUT4_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#elif INPUT4_DIMS == 7
    const int output_high_offset = INPUT4_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT4_DIMS == 6
    const int output_high_offset = INPUT4_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT4_DIMS == 5
    const int output_high_offset = INPUT4_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT4_DIMS <= 4
    const int output_high_offset = INPUT4_GET_INDEX_SAFE(b, of, y, x);
#endif

    INPUT0_TYPE val = input[input_offset];

#if OUTPUT_LAYOUT_B_FS_YX_FSV16
    if (of >= OUTPUT_FEATURE_NUM)
        return;
#else
    if (x >= OUTPUT_SIZE_X || y >= OUTPUT_SIZE_Y || z >= OUTPUT_SIZE_Z)
        return;
#endif

    INPUT0_TYPE input_low_val  = input_low[input_low_offset];
    INPUT0_TYPE input_high_val  = input_high[input_high_offset];
    INPUT0_TYPE output_low_val  = output_low[output_low_offset];
    INPUT0_TYPE output_high_val  = output_high[output_high_offset];


    if (val <= input_low_val)
    {
        output[output_offset] = TO_OUTPUT_TYPE(output_low_val);
    }
    else if (val > input_high_val)
    {
        output[output_offset] = TO_OUTPUT_TYPE(output_high_val);
    }
    else
    {
#if OUTPUT_IS_FP
       output[output_offset] = TO_OUTPUT_TYPE(round((val - input_low_val) / (input_high_val - input_low_val) * (LEVELS-1))
                             * (UNIT_VAL_ONE / (LEVELS-1) * (output_high_val - output_low_val)) + output_low_val);
#else
       // TODO: the outer round should be deleted once output range is correct
        output[output_offset] = TO_OUTPUT_TYPE(round(round((val - input_low_val) / (input_high_val - input_low_val) * (LEVELS-1))
                              * (UNIT_VAL_ONE / (LEVELS-1) * (output_high_val - output_low_val)) + output_low_val));
#endif
    }
}
