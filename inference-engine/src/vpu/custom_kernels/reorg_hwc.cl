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

#define MIN(v1, v2) ((v1) < (v2) ? (v1) : (v2))

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void reorg(__global half* restrict src,
                    __global half* restrict out,
                    int h,
                    int w,
                    int stride)
{
    int j = MIN(get_global_id(0), h-1);

    int k = get_global_id(1);
    int c = get_global_size(1);

    int out_c = c / (stride * stride);
    int oc    = c * (stride * stride);
    int oh    = h / stride;
    int ow    = w / stride;

    int in_index = w * (j + h*k);

    int new_z = in_index / (oh*ow);
    int new_y = (in_index %(oh*ow)) / ow;
    int new_x = (in_index %(oh*ow)) % ow;
    int new_index = new_z + new_x * oc + new_y * oc * ow;

    in_index++;

    int c2 = k % out_c;
    int offset = k / out_c;
    int w2 = 0 * stride + offset % stride;
    int h2 = j * stride + offset / stride;
    int out_index = w2 + w * stride * (h2 + h * stride * c2);

    for (int i = 0; i < w; ++i, out_index+=stride, in_index++)
    {
        // repacking coordinates
        int k0 =  out_index / (h*w);
        int j0 = (out_index % (h*w)) / w;
        int i0 = (out_index % (h*w)) % w;
        int out_index_repack = k0 + c * i0 + c * w * j0;
        out[new_index] = src[out_index_repack];

        int new_z =  in_index / (oh*ow);
        int new_y = (in_index %(oh*ow)) / ow;
        int new_x = (in_index %(oh*ow)) % ow;
        new_index = new_z + new_x * oc + new_y * oc * ow;
    }
}
