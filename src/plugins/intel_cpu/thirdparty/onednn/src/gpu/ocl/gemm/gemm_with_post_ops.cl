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
#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"
#if defined(DST_DT_BF16)
#define DST_TO_ACC(x) cvt_bf16_to_f32(x)
#else
#define DST_TO_ACC(x) (x)
#endif
#if defined(BIA_DT_BF16)
#define BIA_TO_ACC(x) cvt_bf16_to_f32(x)
#else
#define BIA_TO_ACC(x) (x)
#endif
#if defined(SRC_DT_BF16)
#define SRC_TO_ACC(x) cvt_bf16_to_f32(x)
#else
#define SRC_TO_ACC(x) (x)
#endif
#ifndef BIA_D2
#define BIA_D2 1
#endif
#ifndef BIA_D3
#define BIA_D3 1
#endif

#if BIA_NDIMS == 2
#define BIA_OFF(x0, x1, d, h, w) \
    (((x0 % BIA_D0) % BIA_B0) * BIA_SB0 + ((x0 % BIA_D0) / BIA_B0) * BIA_S0 \
            + ((x1 % BIA_D1) % BIA_B1) * BIA_SB1 \
            + ((x1 % BIA_D1) / BIA_B1) * BIA_S1)
#elif BIA_NDIMS == 1
#define BIA_OFF(x1, x0, d, h, w) (x0)
#endif
__kernel void gemm_post_ops(__global SRC_DATA_T *src, __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, __global SPAD_DATA_T *scratchpad,
        global float *scales, int scale_stride) {
    const uint mb = GWS_GET_MB();
    const uint oc = GWS_GET_OC();
    const uint mb2 = GWS_GET_MB2();
    const uint mb3 = GWS_GET_MB3();

    const size_t data_idx = DST_OFF(mb, oc, 0, mb3, mb2);

#if USE_TEMP_DST == 1
    ACC_DATA_T acc = SRC_TO_ACC(scratchpad[data_idx]);
#else
    ACC_DATA_T acc = SRC_TO_ACC(src[data_idx]);
#endif

#if WITH_BIAS == 1
    const size_t bia_idx = BIA_OFF(mb, oc, 0, 0, 0);
    acc += BIA_TO_ACC(bias[bia_idx]);
#endif

#if WITH_SCALES
    const float scale = scales[scale_stride * oc];
    acc *= scale;
#endif

    // Apply postops
    float sum_src;
#if WITH_SUM
    sum_src = DST_TO_ACC(dst[data_idx]);
#endif

    float accumulator = acc;
    if (NDIMS == 2)
        APPLY_POST_OPS_SERIAL_BINARY_2D(
                accumulator, float, sum_src, float, mb, 1, oc, 1);

    dst[data_idx] = TO_DST(accumulator);
}
