/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"
#ifdef DST_DT_F32
#define DST_TO_ACC(x) (x)
#else
#define DST_TO_ACC(x) TO_DEF_ACC_DATA_T(x)
#endif
#ifdef BIAS_DT_F32
#define BIAS_TO_ACC(x) (x)
#else
#define BIAS_TO_ACC(x) TO_DEF_ACC_DATA_T(x)
#endif
#ifdef SRC_DT_F32
#define SRC_TO_ACC(x) (x)
#else
#define SRC_TO_ACC(x) TO_DEF_ACC_DATA_T(x)
#endif

__kernel void gemm_post_ops_inner_product(__global SRC_DATA_T *src,
        __global BIAS_DATA_T *bias, __global DST_DATA_T *dst POST_OP_ARGS,
        __global SPAD_DATA_T *scratchpad, global float *scales) {
    const size_t mb = get_global_id(0) / OC;
    const size_t oc = get_global_id(0) % OC;

    const size_t data_idx = mb * OC + oc;
#if USE_TEMP_DST == 1
    ACC_DATA_T acc = SRC_TO_ACC(scratchpad[data_idx]);
#else
    ACC_DATA_T acc = SRC_TO_ACC(src[data_idx]);
#endif

#if WITH_BIAS == 1
    acc += BIAS_TO_ACC(bias[oc]);
#endif

#if WITH_SCALES
#if SCALES_COMMON
    const float scale = scales[0];
#elif SCALES_PER_OC
    const float scale = scales[oc];
#else
#error "Unsupported scale type"
#endif
    acc *= scale;
#endif

    // Apply postops
    float sum_src;
#if WITH_SUM
    sum_src = DST_TO_ACC(dst[data_idx]);
#endif

    float accumulator = acc;
    APPLY_POST_OPS_SERIAL_BINARY_2D(
            accumulator, float, sum_src, float, mb, 1, oc, 1);

    dst[data_idx] = TO_DST(accumulator);
}
