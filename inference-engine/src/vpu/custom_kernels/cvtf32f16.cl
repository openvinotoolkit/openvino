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

__kernel void cvtf32f16(const __global float* restrict inImage,
                              __global half*  restrict outImage,
                                       float   scale,
                                       float   bais)
{
    int idx = get_global_id(0)
            + get_global_id(1) * get_global_size(0)
            + get_global_id(2) * get_global_size(0) * get_global_size(1);

    outImage[idx] = convert_half(inImage[idx]*scale+bais);
}
