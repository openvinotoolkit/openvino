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

#include "gpu/ocl/ocl_eltwise.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define SIMD GWS_SGS_DEFAULT

KERNEL_ATTR
__kernel void gen9_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, float alpha, float beta) {
    const uint grsize = get_local_size(0);
    const uint grid = get_group_id(0);
    const uint sgid = get_sub_group_id();
    const uint lid = get_sub_group_local_id();

    const uint gid = get_global_id(0);

    ptrdiff_t offset
            = (grid * grsize + sgid * get_max_sub_group_size()) * VECT_DT_N;

    // grsize is a multiple of 16, SIMD is 16 -> offset mod 16 = 0
    // -> read_pos correctly aligned for block reads
    // -> write_pos correctly aligned for block writes
    __global BLOCK_DATA_T *read_pos = (__global BLOCK_DATA_T *)src + offset;
    __global BLOCK_DATA_T *write_pos = (__global BLOCK_DATA_T *)dst + offset;

    VECT_DATA_T val;
    const uint nel_per_read = SIMD * VECT_DT_N;

    // READ
    if (offset + nel_per_read < NELEMS) {
        val = AS_VECT_DATA_T(VECT_BLOCK_READ(read_pos));

    } else {
        // read data in the same access pattern block_reads would
        uint pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < NELEMS; ++i) {
            val[i] = src[pos];
            pos += SIMD;
        }
    }

    // COMPUTE
    for (int i = 0; i < VECT_DT_N; ++i) {
        val[i] = CONVERT_DATA_T(
                fwd_eltwise(DATA_TO_REF(val[i]), alpha, beta, 1.0f));
    }

    // WRITE
    if (offset + nel_per_read < NELEMS) {
        VECT_BLOCK_WRITE(write_pos, AS_VECT_BLOCK_DATA_T(val));

    } else {
        uint pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < NELEMS; ++i) {
            dst[pos] = val[i];
            pos += SIMD;
        }
    }
}

KERNEL_ATTR
__kernel void gen9_eltwise_bwd(__global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, float alpha, float beta) {
    const uint grsize = get_local_size(0);
    const uint grid = get_group_id(0);
    const uint sgid = get_sub_group_id();
    const uint lid = get_sub_group_local_id();

    ptrdiff_t offset = (grid * grsize + sgid * SIMD) * VECT_DT_N;
    //TODO: It should be implemented two distinct offsets
    //The one for src and the second for diff_src

    // grsize is a multiple of 16, SIMD is 16 -> offset mod 16 = 0
    // -> read_pos correctly aligned for block reads
    // -> write_pos correctly aligned for block writes
    __global BLOCK_DATA_T *src_pos = (__global BLOCK_DATA_T *)src + offset;
    __global BLOCK_DATA_T *diff_pos
            = (__global BLOCK_DATA_T *)diff_dst + offset;
    __global BLOCK_DATA_T *write_pos
            = (__global BLOCK_DATA_T *)diff_src + offset;

    VECT_DATA_T val_dd;
    VECT_DATA_T val_src;
    const uint nel_per_read = SIMD * VECT_DT_N;

    // READ
    if (offset + nel_per_read < NELEMS) {
        val_src = AS_VECT_DATA_T(VECT_BLOCK_READ(src_pos));
        val_dd = AS_VECT_DATA_T(VECT_BLOCK_READ(diff_pos));

    } else {
        // read data in the same access pattern block_reads would
        uint pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < NELEMS; ++i) {
            val_dd[i] = diff_dst[pos];
            val_src[i] = src[pos];
            pos += SIMD;
        }
    }

    // COMPUTE
    for (int i = 0; i < VECT_DT_N; ++i) {
        val_dd[i] = CONVERT_DATA_T(bwd_eltwise(
                DATA_TO_REF(val_dd[i]), DATA_TO_REF(val_src[i]), alpha, beta));
    }

    // WRITE
    if (offset + nel_per_read < NELEMS) {
        VECT_BLOCK_WRITE(write_pos, AS_VECT_BLOCK_DATA_T(val_dd));

    } else {
        // write data in the same access pattern block_writes would
        uint pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < NELEMS; ++i) {
            diff_src[pos] = val_dd[i];
            pos += SIMD;
        }
    }
}
