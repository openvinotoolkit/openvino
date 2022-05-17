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

#define DT_UNDEF 1
#include "gpu/ocl/ocl_types.h"

#include "gpu/ocl/ocl_math_utils.h"

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(SRC, (x0), (x1), (x2), (x3), (x4), (x5))
#define DST_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(DST, (x0), (x1), (x2), (x3), (x4), (x5))

int rev_transposed(int a) {
    return ((a % TRANSPOSE_COL) * TRANSPOSE_ROW + a / TRANSPOSE_COL);
}

__kernel void ref_shuffle(__global DATA_T *src, __global DATA_T *dst) {

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

    int d[5];
    d[0] = GWS_GET_D0();
    d[1] = GWS_GET_D1();
    d[2] = GWS_GET_D2();
    d[3] = GWS_GET_D3();
    d[4] = GWS_GET_D4();
    d[5] = GWS_GET_D5();

    const ulong src_off = SRC_OFF(d[0], d[1], d[2], d[3], d[4], d[5]);

    d[AXIS] = rev_transposed(d[AXIS]);
    const ulong dst_off = DST_OFF(d[0], d[1], d[2], d[3], d[4], d[5]);

    dst[dst_off] = src[src_off];
}
