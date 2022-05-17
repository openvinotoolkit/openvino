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

#define IS_IN_PART(x) (dst_dims[CONCAT_AXIS] < CONCAT3(SRC, x, _END))

#define SET_DIMS(x, y) \
    { \
        part = y; \
        if (y > 0) { \
            src_dims[CONCAT_AXIS] \
                    = dst_dims[CONCAT_AXIS] - CONCAT3(SRC, x, _END); \
        } \
        src_off = OFF_MD(CONCAT2(SRC, y), src_dims[0], src_dims[1], \
                src_dims[2], src_dims[3], src_dims[4], src_dims[5]); \
        src = CONCAT2(src, y); \
    }

#define SRC_DATA_T SRC0_DATA_T
#define DD(i) CONCAt2(DST_D, i)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, dim5) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || dim4 >= DD(4) || dim5 >= DD(5)

KERNEL_ATTR
__kernel void gen9_concat(__global DST_DATA_T *dst,
        __global const SRC_DATA_T *src0, __global const SRC_DATA_T *src1,
        __global const SRC_DATA_T *src2, __global const SRC_DATA_T *src3,
        __global const SRC_DATA_T *src4, __global const SRC_DATA_T *src5,
        __global const SRC_DATA_T *src6, __global const SRC_DATA_T *src7,
        __global const SRC_DATA_T *src8, __global const SRC_DATA_T *src9,
        __global const SRC_DATA_T *src10, __global const SRC_DATA_T *src11,
        __global const SRC_DATA_T *src12, __global const SRC_DATA_T *src13,
        __global const SRC_DATA_T *src14, __global const SRC_DATA_T *src15) {
    int dst_dims[6], src_dims[6];
    src_dims[0] = dst_dims[0] = GWS_GET_D0();
    src_dims[1] = dst_dims[1] = GWS_GET_D1();
    src_dims[2] = dst_dims[2] = GWS_GET_D2();
    src_dims[3] = dst_dims[3] = GWS_GET_D3();
    src_dims[4] = dst_dims[4] = GWS_GET_D4();
    src_dims[5] = dst_dims[5] = GWS_GET_D5();

    const int iter_dim_end = dst_dims[ITER_DIM] + ITER_DIM_CHUNK;

    if (NEEDS_PADDING(dst_dims[0], dst_dims[1], dst_dims[2], dst_dims[3],
                dst_dims[4], dst_dims[5])) {
        for (; dst_dims[ITER_DIM] < iter_dim_end; dst_dims[ITER_DIM]++) {
            const int dst_off = OFF_MD(DST, dst_dims[0], dst_dims[1],
                    dst_dims[2], dst_dims[3], dst_dims[4], dst_dims[5]);
#if SUB_GROUP_SIZE > 1
            BLOCK_WRITE_DST(&dst[dst_off], TO_DST(DATA_ZERO));
#else // SUB_GROUP_SIZE > 1
            dst[dst_off] = TO_DST(DATA_ZERO);
#endif // SUB_GROUP_SIZE > 1
        }
        return;
    }
    for (; dst_dims[ITER_DIM] < min(DD(ITER_DIM), iter_dim_end);
            dst_dims[ITER_DIM]++, src_dims[ITER_DIM]++) {
        int part;
        int src_off;
        __global SRC_DATA_T *src;

        if (IS_IN_PART(0)) SET_DIMS(0, 0)
#if NUM_INPUTS >= 2
        else if (IS_IN_PART(1))
            SET_DIMS(0, 1)
#endif
#if NUM_INPUTS >= 3
        else if (IS_IN_PART(2))
            SET_DIMS(1, 2)
#endif
#if NUM_INPUTS >= 4
        else if (IS_IN_PART(3))
            SET_DIMS(2, 3)
#endif
#if NUM_INPUTS >= 5
        else if (IS_IN_PART(4))
            SET_DIMS(3, 4)
#endif
#if NUM_INPUTS >= 6
        else if (IS_IN_PART(5))
            SET_DIMS(4, 5)
#endif
#if NUM_INPUTS >= 7
        else if (IS_IN_PART(6))
            SET_DIMS(5, 6)
#endif
#if NUM_INPUTS >= 8
        else if (IS_IN_PART(7))
            SET_DIMS(6, 7)
#endif
#if NUM_INPUTS >= 9
        else if (IS_IN_PART(8))
            SET_DIMS(7, 8)
#endif
#if NUM_INPUTS >= 10
        else if (IS_IN_PART(9))
            SET_DIMS(8, 9)
#endif
#if NUM_INPUTS >= 11
        else if (IS_IN_PART(10))
            SET_DIMS(9, 10)
#endif
#if NUM_INPUTS >= 12
        else if (IS_IN_PART(11))
            SET_DIMS(10, 11)
#endif
#if NUM_INPUTS >= 13
        else if (IS_IN_PART(12))
            SET_DIMS(11, 12)
#endif
#if NUM_INPUTS >= 14
        else if (IS_IN_PART(13))
            SET_DIMS(12, 13)
#endif
#if NUM_INPUTS >= 15
        else if (IS_IN_PART(14))
            SET_DIMS(13, 14)
#endif
#if NUM_INPUTS >= 16
        else if (IS_IN_PART(15))
            SET_DIMS(14, 15)
#endif

        const int dst_off = OFF_MD(DST, dst_dims[0], dst_dims[1], dst_dims[2],
                dst_dims[3], dst_dims[4], dst_dims[5]);

#if SUB_GROUP_SIZE > 1
#if DT_BF16 == 1
        float src_val = DATA_TO_REF(AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
#else // DT_BF16 == 1
        SRC_DATA_T src_val = AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off]));
#endif // DT_BF16 == 1
        BLOCK_WRITE_DST(&dst[dst_off], TO_DST(src_val));
#else // SUB_GROUP_SIZE > 1
#if DT_BF16 == 1
        float src_val = DATA_TO_REF(src[src_off]);
#else // DT_BF16 == 1
        SRC_DATA_T src_val = src[src_off];
#endif // DT_BF16 == 1
        dst[dst_off] = TO_DST(src_val);
#endif // SUB_GROUP_SIZE > 1
    }
    for (; dst_dims[ITER_DIM] < iter_dim_end; dst_dims[3]++) {
        const int dst_off = OFF_MD(DST, dst_dims[0], dst_dims[1], dst_dims[2],
                dst_dims[3], dst_dims[4], dst_dims[5]);
#if SUB_GROUP_SIZE > 1
        BLOCK_WRITE_DST(&dst[dst_off], TO_DST(DATA_ZERO));
#else // SUB_GROUP_SIZE > 1
        dst[dst_off] = TO_DST(DATA_ZERO);
#endif // SUB_GROUP_SIZE > 1
    }
}
