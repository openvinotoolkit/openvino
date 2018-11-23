// Copyright (c) 2017 Intel Corporation
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

#if OUTPUT_LAYOUT_BFYX
    #define IW INPUT0_SIZES[0]
    #define IH INPUT0_SIZES[1]
    #define IC INPUT0_SIZES[2]
    #define B  INPUT0_SIZES[3]

#elif OUTPUT_LAYOUT_YXFB
    #define IW INPUT0_SIZES[3]
    #define IH INPUT0_SIZES[2]
    #define IC INPUT0_SIZES[1]
    #define B  INPUT0_SIZES[0]
#endif

#define ic_off (IC / (STRIDE * STRIDE))
#define ih_off (IH * STRIDE)
#define iw_off (IW * STRIDE)

KERNEL (reorg_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if OUTPUT_LAYOUT_BFYX
    int ic = get_global_id(2);
    int ih = get_global_id(1);
    int iw = get_global_id(0);
        for (int b = 0; b < B; b++) {
        int dstIndex = b*IC*IH*IW + ic*IH*IW + ih*IW + iw;

        int oc = ic % ic_off;
        int offset = ic / ic_off;

        int ow = iw * STRIDE + offset % STRIDE;
        int oh = ih * STRIDE + offset / STRIDE;

        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;

        output[dstIndex] = input[srcIndex];
    }
#elif OUTPUT_LAYOUT_YXFB
    int ic = get_global_id(0) / B;
    int ib = get_global_id(0) % B;
    int ih = get_global_id(2);
    int iw = get_global_id(1);
    for (int b = 0; b < B; b++) {
        int dstIndex = ib + ic*B + ih*IC*B + iw*IH*IC*B;

        int oc = ic % ic_off;
        int offset = ic / ic_off;

        int ow = iw * STRIDE + offset % STRIDE;
        int oh = ih * STRIDE + offset / STRIDE;

        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;

        output[dstIndex] = input[srcIndex];
    }
#endif
    

}
