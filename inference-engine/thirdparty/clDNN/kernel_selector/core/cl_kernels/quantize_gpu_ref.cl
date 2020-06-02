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

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"

#ifdef SUB_GROUP_SIZE
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
KERNEL(quantize_ref)(const __global INPUT0_TYPE* input,
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
#endif

#if PACKED_BINARY_OUTPUT
    const int output_offset = OUTPUT_OFFSET
                            + b*OUTPUT_FEATURE_NUM_PACKED*OUTPUT_FEATURE_PITCH
                            + of*OUTPUT_FEATURE_PITCH
                            + y*OUTPUT_Y_PITCH
                            + x*OUTPUT_X_PITCH;

    const int threshold_offset = INPUT1_OFFSET
                               + (b % INPUT1_BATCH_NUM)*INPUT1_BATCH_PITCH
                               + (y % INPUT1_SIZE_Y)*INPUT1_Y_PITCH
                               + (x % INPUT1_SIZE_X)*INPUT1_X_PITCH;

    OUTPUT_TYPE res = 0x00000000;
#if SINGLE_OUT_VAL
    int high_bit = output_high[0] == UNIT_VAL_ONE ? 1 : 0;
    int low_bit  = output_low[0] == UNIT_VAL_ONE ? 1 : 0;
#endif
    int limit = min((int)OC_BLOCK_SIZE, (int)INPUT0_FEATURE_NUM);
    for (int f = 0; f < limit; f++)
    {
        UNIT_TYPE val = input[INPUT0_GET_INDEX(b, of*OC_BLOCK_SIZE + f, y, x)];
        UNIT_TYPE threshold  = input_low[threshold_offset + ((of*OC_BLOCK_SIZE + f) % INPUT1_FEATURE_NUM)*INPUT1_FEATURE_PITCH];
#if PER_CHANNEL_OUT_VAL
        int high_bit = output_high[of*OC_BLOCK_SIZE + f] == UNIT_VAL_ONE ? 1 : 0;
        int low_bit  = output_low[of*OC_BLOCK_SIZE + f] == UNIT_VAL_ONE ? 1 : 0;
#endif
        res |= (((val > threshold) ? high_bit : low_bit) << f);
    }

    if (x >= OUTPUT_SIZE_X || y >= OUTPUT_SIZE_Y)
        return;

    output[output_offset] = res;

#else

#if INPUT0_DIMS == 5
    const int input_offset = INPUT0_GET_INDEX(b, of, z, y, x);
#elif INPUT0_DIMS <= 4
    const int input_offset = INPUT0_GET_INDEX(b, of, y, x);
#endif

#if OUTPUT_DIMS == 5
    const int output_offset = OUTPUT_GET_INDEX(b, of, z, y, x);
#elif OUTPUT_DIMS <= 4
    const int output_offset = OUTPUT_GET_INDEX(b, of, y, x);
#endif

#if INPUT1_DIMS == 5
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT1_DIMS <= 4
    const int input_low_offset = INPUT1_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT2_DIMS == 5
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT2_DIMS <= 4
    const int input_high_offset = INPUT2_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT3_DIMS == 5
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT3_DIMS <= 4
    const int output_low_offset = INPUT3_GET_INDEX_SAFE(b, of, y, x);
#endif

#if INPUT4_DIMS == 5
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

#endif
}
