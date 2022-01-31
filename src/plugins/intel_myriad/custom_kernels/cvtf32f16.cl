// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
