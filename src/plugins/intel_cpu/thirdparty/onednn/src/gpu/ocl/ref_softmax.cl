/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#if IS_FWD
#define DD(i) CONCAt2(DST_D, i)
#elif IS_BWD
#define DD(i) CONCAt2(DIFF_SRC_D, i)
#else
#error unsupported data parameter
#endif

#define OFF(dim, idx) \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    softmax_dim >= DD(0) || dim0 >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 1
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || softmax_dim >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 2
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || softmax_dim >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 3
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || softmax_dim >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 4
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) \
            + OFF(softmax_dim, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || softmax_dim >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 5
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) + OFF(dim4, 4) \
            + OFF(softmax_dim, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || dim4 >= DD(4) || softmax_dim >= DD(5)
#else
#error unsupported softmax dimension
#endif

#if IS_FWD
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))

__kernel void
ref_softmax_fwd_generic(__global DATA_T *src, __global DATA_T *dst) {

    const int dim[] = {
            (get_global_id(0) / GROUP_SIZE) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            (get_global_id(0) / GROUP_SIZE) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    const int local_id = get_local_id(0);

    // SOFTMAX_AXIS is the size of axis around which softmax operation is
    // performed

    // begin and end indices calculated according to thread's id
    const int begin = local_id * (SOFTMAX_AXIS / GROUP_SIZE);
    const int end = (local_id == GROUP_SIZE - 1)
            ? SOFTMAX_AXIS
            : (local_id + 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#if SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE) \
        > SOFTMAX_AXIS / GROUP_SIZE
    const int buf_size
            = SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#else
    const int buf_size = SOFTMAX_AXIS / GROUP_SIZE;
#endif

    DEF_ACC_DATA_T d[buf_size];
    DEF_ACC_DATA_T max_ = -FLT_MAX;
    DEF_ACC_DATA_T denom_ = DATA_ZERO;

    // finding max value for each sub_group
    if (!(NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], begin))) {
        for (int i = begin; i < end && i < DD(SOFTMAX_AXIS_IDX); ++i) {
            size_t data_off
                    = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
            d[i - begin] = TO_DEF_ACC_DATA_T(src[data_off]);
            max_ = max(max_, d[i - begin]);
        }
    }

    // reduce using work_group_reduce if no. of subgroups > 1, for e.g.
    // if group_size is 32, there will be 2 sub-groups (size of each sub-group
    // is 16 which is an optimal value)
#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif
    // updating dst tensor and accumulating denom for last step
    if (!(NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], begin))) {
        for (int i = begin; i < end && i < DD(SOFTMAX_AXIS_IDX); ++i) {
#if LOGSOFTMAX
            denom_ += exp(d[i - begin] - max_);
#else
            d[i - begin] = exp(d[i - begin] - max_);
            denom_ += d[i - begin];
#endif
        }
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    denom_ = sub_group_reduce_add(denom_);
#else
    denom_ = work_group_reduce_add(denom_);
#endif

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    for (int i = begin; i < end; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);

        if (NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], i)) {
            dst[data_off] = DATA_ZERO;
        } else {
#if LOGSOFTMAX
            dst[data_off] = TO_DATA_T(d[i - begin] - max_ - denom_);
#else
            dst[data_off] = TO_DATA_T(d[i - begin] * denom_);
#endif
        }
    }
}

#endif

#if IS_BWD
__kernel void ref_softmax_bwd_generic(__global DATA_T *dst,
        __global DATA_T *diff_src, __global DATA_T *diff_dst) {
    const int dim[] = {
            get_global_id(0) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            get_global_id(0) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    DEF_ACC_DATA_T sbr = 0.f;
    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);

        if (NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], i)) {
            continue;
        }

        DEF_ACC_DATA_T g_temp = TO_DEF_ACC_DATA_T(diff_dst[idx]);
#if LOGSOFTMAX
        sbr += g_temp;
#else
        DEF_ACC_DATA_T y_temp = TO_DEF_ACC_DATA_T(dst[idx]);
        sbr += g_temp * y_temp;
#endif
    }

    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);

        if (NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], i)) {
            diff_src[idx] = DATA_ZERO;
            continue;
        }
#if LOGSOFTMAX
        diff_src[idx] = TO_DATA_T(TO_DEF_ACC_DATA_T(diff_dst[idx])
                - exp(TO_DEF_ACC_DATA_T(dst[idx])) * sbr);
#else
        DEF_ACC_DATA_T inner_data = TO_DEF_ACC_DATA_T(diff_dst[idx]) - sbr;
        diff_src[idx] = TO_DATA_T(TO_DEF_ACC_DATA_T(dst[idx]) * inner_data);
#endif
    }
}
#endif
