/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/ocl/ocl_types.h"

#define ALG_AVG (ALG_AVG_NP || ALG_AVG_P)

KERNEL_ATTR
__kernel void gen9_global_pooling_bwd(__global DATA_T *diff_src,
        __global int *ws, __global DATA_T *diff_dst) {
    const int mb = GWS_GET_MB();
    const int c = GWS_GET_C();
    const int spatial = GWS_GET_SPATIAL();

    const bool is_in_padded_area = NEED_ZERO_PADDING && (mb >= MB || c >= C);
    const int dst_off = DST_OFF(mb, c, 0, 0, 0);
#if ALG_AVG
    // Read dst value only once
    const DATA_T dst_val = diff_dst[dst_off];
#endif // ALG_AVG
    int ws_val = ws[dst_off];
    for (int sp_idx = spatial;
            sp_idx < min(spatial + SPATIAL_CHUNK, SPATIAL_DIM); sp_idx++) {
        const int iw = sp_idx % IW;
        const int ih = ((sp_idx - iw) % (IH * IW)) / IW;
        const int id = (sp_idx - iw - ih * IW) / (IH * IW);
        DATA_T val_to_write;
        if (is_in_padded_area)
            val_to_write = DATA_ZERO;
        else {
#if ALG_MAX
            // Read dst value only in case it's going to be used
            const int current_input_idx = id * IH * IW + ih * IW + iw;
            if (current_input_idx == ws_val) {
                val_to_write = diff_dst[dst_off];
            } else {
                val_to_write = DATA_ZERO;
            }
#else // ALG_MAX
            float dst_val_f = DATA_TO_REF(dst_val) / SPATIAL_DIM;
            val_to_write = CONVERT_DATA_T(dst_val_f);
#endif // ALG_MAX
        }
        const int src_off = SRC_OFF(mb, c, id, ih, iw);
        diff_src[src_off] = val_to_write;
    }
}
