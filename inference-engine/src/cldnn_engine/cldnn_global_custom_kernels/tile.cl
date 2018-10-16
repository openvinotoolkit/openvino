// Copyright (c) 2018 Intel Corporation
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

__kernel void tile(const __global INPUT0_TYPE*  input,
                        __global OUTPUT0_TYPE* output)
{
    const int dims = sizeof(INPUT0_DIMS) / sizeof(INPUT0_DIMS[0]);

    int outer_dim = 1;
    int inner_dim = 1;

    __global OUTPUT0_TYPE* pdst = output;
    __global INPUT0_TYPE*  psrc = input;

    for (int i = 0; i < axis_; i++)
    {
        outer_dim *= INPUT0_DIMS[i];
    }
    for (int i = axis_; i < dims; i++)
    {
        inner_dim *= INPUT0_DIMS[i];
    }

    for (int i = 0; i < outer_dim; i++)
    {
        for (int t = 0; t < tiles_; t++)
        {
            for (int j = 0; j < inner_dim; j++)
            {
                pdst[j] = psrc[j];
            }
            pdst += inner_dim;
        }
        psrc += inner_dim;
    }
}
