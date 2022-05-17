/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(val)))

#define VECT_SIZE 8
#define NUM_BUF (SOFTMAX_AXIS_SIZE / SUB_GROUP_SIZE / VECT_SIZE)

#if IS_FWD

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_fwd(__global DATA_T *src, __global DATA_T *dst) {
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    float8 d[NUM_BUF];

    float max_ = -FLT_MAX;
    float denom_ = 0.f;

    src += data_off;

    for (int k = 0; k < NUM_BUF; ++k) {
        d[k] = LOAD_DATA_8x16(&src[k * VECT_SIZE * SUB_GROUP_SIZE]);
        for (int i = 0; i < VECT_SIZE; ++i) {
            max_ = max(d[k][i], max_);
        }
    }

    max_ = sub_group_reduce_max(max_);

    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += exp(d[k][i] - max_);
#else
        d[k] = exp(d[k] - max_);
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += d[k][i];
#endif
    }

    denom_ = sub_group_reduce_add(denom_);

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        STORE_DATA_8x16(&dst[k * VECT_SIZE * SUB_GROUP_SIZE], d[k]);
    }
}

#endif
