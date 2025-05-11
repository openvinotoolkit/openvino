// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"


#if OUTPUT_LAYOUT_YXFB
    #define IW INPUT0_SIZES[3]
    #define IH INPUT0_SIZES[2]
    #define IC INPUT0_SIZES[1]
    #define B  INPUT0_SIZES[0]
#elif OUTPUT_LAYOUT_BYXF
    #define IW INPUT0_SIZES[1]
    #define IH INPUT0_SIZES[2]
    #define IC INPUT0_SIZES[0]
    #define B  INPUT0_SIZES[3]
#else
    #define IW INPUT0_SIZES[0]
    #define IH INPUT0_SIZES[1]
    #define IC INPUT0_SIZES[2]
    #define B  INPUT0_SIZES[3]
#endif

#define ic_off (IC / (STRIDE * STRIDE))
#define ih_off (IH * STRIDE)
#define iw_off (IW * STRIDE)

#if !defined(OUTPUT_LAYOUT_BFYX)
inline void FUNC(planar_to_bfyx)(const uint planar_index,
                                 const uint batch_num, const uint channel_num, const uint height, const uint width,
                                 uint* dst_b, uint* dst_f, uint* dst_y, uint* dst_x)
{
    const uint feature_size = height * width;
    const uint batch_size = channel_num * feature_size;

    *dst_b = planar_index / batch_size;
    const uint dst_fxy = planar_index % batch_size;
    *dst_f = dst_fxy / feature_size;
    const uint dst_xy = dst_fxy % feature_size;
    *dst_y = dst_xy / width;
    *dst_x = dst_xy % width;
}
#endif

KERNEL (reorg_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    int ic = get_global_id(2);
    int ih = get_global_id(1);
    int iw = get_global_id(0);

#if OUTPUT_LAYOUT_BFYX
    for (int b = 0; b < B; b++) {
        int dstIndex = b*IC*IH*IW + ic*IH*IW + ih*IW + iw;

        int oc = ic % ic_off;
        int offset = ic / ic_off;

        int ow = iw * STRIDE + offset % STRIDE;
        int oh = ih * STRIDE + offset / STRIDE;

        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;

        output[dstIndex] = input[srcIndex];
    }
#else
    const uint OC = IC * STRIDE * STRIDE;
    const uint OH = IH / STRIDE;
    const uint OW = IW / STRIDE;

    for (int b = 0; b < B; b++) {
        const uint dstPlanarIndex = b*IC*IH*IW + ic*IH*IW + ih*IW + iw;
        uint dstB, dstC, dstY, dstX;
        FUNC_CALL(planar_to_bfyx)(dstPlanarIndex, B, OC, OH, OW, &dstB, &dstC, &dstY, &dstX);
        const uint dstIndex = OUTPUT_GET_INDEX(dstB, dstC, dstY, dstX);

        const int oc = ic % ic_off;
        const int offset = ic / ic_off;

        const int ow = iw * STRIDE + offset % STRIDE;
        const int oh = ih * STRIDE + offset / STRIDE;

        const int srcPlanarIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;
        uint srcB, srcC, srcY, srcX;
        FUNC_CALL(planar_to_bfyx)(srcPlanarIndex, B, IC, IH, IW, &srcB, &srcC, &srcY, &srcX);
        const uint srcIndex = INPUT0_GET_INDEX(srcB, srcC, srcY, srcX);

        output[dstIndex] = input[srcIndex];
    }
#endif
}

#undef iw_off
#undef ih_off
#undef ic_off
#undef B
#undef IC
#undef IH
#undef IW
