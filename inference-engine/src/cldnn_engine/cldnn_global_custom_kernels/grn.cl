// Copyright (C) 2018-2020 Intel Corporation
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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void grn(const __global INPUT0_TYPE*  input,
                        __global OUTPUT0_TYPE* output)
{
    const int dims = sizeof(INPUT0_DIMS) / sizeof(INPUT0_DIMS[0]);

    const int F = INPUT0_DIMS[1];
    const int Y = INPUT0_DIMS[2];
    const int X = INPUT0_DIMS[3];

    const int IF = F + INPUT0_LOWER_PADDING[1] + INPUT0_UPPER_PADDING[1];
    const int IY = Y + INPUT0_LOWER_PADDING[2] + INPUT0_UPPER_PADDING[2];
    const int IX = X + INPUT0_LOWER_PADDING[3] + INPUT0_UPPER_PADDING[3];

    const int OF = OUTPUT0_DIMS[1] + OUTPUT0_LOWER_PADDING[1] + OUTPUT0_UPPER_PADDING[1];
    const int OY = OUTPUT0_DIMS[2] + OUTPUT0_LOWER_PADDING[2] + OUTPUT0_UPPER_PADDING[2];
    const int OX = OUTPUT0_DIMS[3] + OUTPUT0_LOWER_PADDING[3] + OUTPUT0_UPPER_PADDING[3];

    int in_padding = INPUT0_LOWER_PADDING[2]*OX + INPUT0_LOWER_PADDING[3];
    int out_padding = OUTPUT0_LOWER_PADDING[2]*OX + OUTPUT0_LOWER_PADDING[3];

    int in_offset = get_global_id(0)*IF*IY*IX + get_global_id(1)*IX + in_padding;
    int out_offset = get_global_id(0)*OF*OY*OX + get_global_id(1)*OX + out_padding;

    for (int x = 0; x < X; x++)
    {
        ACCUMULATOR_TYPE variance = 0;
        for (int f = 0; f < F; f++)
        {
            int in_off = in_offset + f*IY*IX + x;
            INPUT0_TYPE val = input[in_off];
            variance += val*val;
        }
        variance = sqrt(variance + bias_);
        for (int f = 0; f < F; f++)
        {
            int in_off = in_offset + f*IY*IX + x;
            int out_off = out_offset + f*OY*OX + x;
            output[out_off] = (OUTPUT0_TYPE)(input[in_off] / variance);
        }
    }
}
