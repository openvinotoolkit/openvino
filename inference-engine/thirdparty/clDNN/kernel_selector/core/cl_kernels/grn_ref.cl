// Copyright (C) 2020 Intel Corporation
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

KERNEL(grn_ref)(
    const __global INPUT0_TYPE*  input,
    __global OUTPUT_TYPE* output)
{
    const uint ob = get_global_id(0);
    const uint oy = get_global_id(1);
    const uint ox = get_global_id(2);

    int in_offset  = INPUT0_GET_INDEX(ob, 0, oy, ox);
    int out_offset = OUTPUT_GET_INDEX(ob, 0, oy, ox);

    ACCUMULATOR_TYPE variance = 0;
    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        int in_off = in_offset + f * INPUT0_FEATURE_PITCH;
        INPUT0_TYPE val = input[in_off];
        variance += val*val;
    }
    variance = sqrt(variance + BIAS);
    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        int in_off = in_offset + f * INPUT0_FEATURE_PITCH;
        int out_off = out_offset + f * OUTPUT_FEATURE_PITCH;
        output[out_off] = (OUTPUT_TYPE)(input[in_off] / variance);
    }
}
