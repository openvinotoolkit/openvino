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

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(quantize_ref)(const __global INPUT0_TYPE* input,
                     const __global INPUT1_TYPE* input_low,
                     const __global INPUT2_TYPE* input_high,
                     const __global INPUT3_TYPE* output_low,
                     const __global INPUT4_TYPE* output_high,
                           __global OUTPUT_TYPE* output)
{
    const int b = get_global_id(0);
    const int of = get_global_id(1);
    const int yx = get_global_id(2);
    const int x = get_global_id(2) % OUTPUT_SIZE_X;
    const int y = get_global_id(2) / OUTPUT_SIZE_X;

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
    const int input_offset = INPUT0_OFFSET
                           + b*INPUT0_BATCH_PITCH
                           + of*INPUT0_FEATURE_PITCH
                           + y*INPUT0_Y_PITCH
                           + x*INPUT0_X_PITCH;
    const int output_offset = OUTPUT_OFFSET
                            + b*OUTPUT_BATCH_PITCH
                            + of*OUTPUT_FEATURE_PITCH
                            + y*OUTPUT_Y_PITCH
                            + x*OUTPUT_X_PITCH;
    const int input_low_offset = INPUT1_OFFSET
                               + (b % INPUT1_BATCH_NUM)*INPUT1_BATCH_PITCH
                               + (of % INPUT1_FEATURE_NUM)*INPUT1_FEATURE_PITCH
                               + (y % INPUT1_SIZE_Y)*INPUT1_Y_PITCH
                               + (x % INPUT1_SIZE_X)*INPUT1_X_PITCH;
    const int input_high_offset = INPUT2_OFFSET
                                + (b % INPUT2_BATCH_NUM)*INPUT2_BATCH_PITCH
                                + (of % INPUT2_FEATURE_NUM)*INPUT2_FEATURE_PITCH
                                + (y % INPUT2_SIZE_Y)*INPUT2_Y_PITCH
                                + (x % INPUT2_SIZE_X)*INPUT2_X_PITCH;
    const int output_low_offset = INPUT3_OFFSET
                                + (b % INPUT3_BATCH_NUM)*INPUT3_BATCH_PITCH
                                + (of % INPUT3_FEATURE_NUM)*INPUT3_FEATURE_PITCH
                                + (y % INPUT3_SIZE_Y)*INPUT3_Y_PITCH
                                + (x % INPUT3_SIZE_X)*INPUT3_X_PITCH;
    const int output_high_offset = INPUT4_OFFSET
                                 + (b % INPUT4_BATCH_NUM)*INPUT4_BATCH_PITCH
                                 + (of % INPUT4_FEATURE_NUM)*INPUT4_FEATURE_PITCH
                                 + (y % INPUT4_SIZE_Y)*INPUT4_Y_PITCH
                                 + (x % INPUT4_SIZE_X)*INPUT4_X_PITCH;

    INPUT0_TYPE val = input[input_offset];
    if (x >= OUTPUT_SIZE_X || y >= OUTPUT_SIZE_Y)
        return;

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
       output[output_offset] = TO_OUTPUT_TYPE(round((val - input_low_val) / (input_high_val - input_low_val) * (LEVELS-1)) /
                                              (LEVELS-1) * (output_high_val - output_low_val) + output_low_val);
    }

#endif
}
