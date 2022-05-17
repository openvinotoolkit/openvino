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
#if DST_DT_S8
#define DST_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#include "gpu/ocl/ocl_types.h"

float8 get_values(__global SRC_DATA_T *src, ptrdiff_t offset) {
    float8 val;
    const uint max_sub_group_size = get_max_sub_group_size();
    __global BLOCK_DATA_T *read_pos = (__global BLOCK_DATA_T *)src + offset;

    if (offset + VECT_DT_N * max_sub_group_size < N_ELEMS) {
        val = CONVERT_FLOAT8_T(AS_DATA8_T(BLOCK_READ8(read_pos)));
    } else {
        const uint sub_group_local_id = get_sub_group_local_id();
        uint pos = offset + sub_group_local_id;
        for (uint i = 0; pos < N_ELEMS && i < VECT_DT_N; i++) {
            val[i] = CONVERT_FLOAT_T(src[pos]);
            pos += max_sub_group_size;
        }
    }
    return val;
}

__kernel void gen9_sum(__global SRC_DATA_T *input0, __global SRC_DATA_T *input1,
        __global SRC_DATA_T *input2, __global SRC_DATA_T *input3,
        __global SRC_DATA_T *input4, __global SRC_DATA_T *input5,
        __global SRC_DATA_T *input6, __global SRC_DATA_T *input7,
        __global SRC_DATA_T *input8, __global SRC_DATA_T *input9,
        __global SRC_DATA_T *input10, __global SRC_DATA_T *input11,
        __global SRC_DATA_T *input12, __global SRC_DATA_T *input13,
        __global SRC_DATA_T *input14, __global SRC_DATA_T *input15,
        __global DST_DATA_T *output, __global float *scales) {

    const uint group_id = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint sub_group_id = get_sub_group_id();
    const uint max_sub_group_size = get_max_sub_group_size();
    const uint sub_group_local_id = get_sub_group_local_id();

    ptrdiff_t offset
            = (group_id * group_size + sub_group_id * max_sub_group_size)
            * VECT_DT_N;

    __global BLOCK_DATA_T *write_pos = (__global BLOCK_DATA_T *)output + offset;

    int id = 0;
    float8 sum = 0;
    if (id < N_INPUTS) sum += get_values(input0, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input1, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input2, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input3, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input4, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input5, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input6, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input7, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input8, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input9, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input10, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input11, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input12, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input13, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input14, offset) * scales[id++];
    if (id < N_INPUTS) sum += get_values(input15, offset) * scales[id++];

    if (offset + VECT_DT_N * max_sub_group_size < N_ELEMS) {
        DST_BLOCK_WRITE8(write_pos, TO_DST8(sum));
    } else {
        uint pos = offset + sub_group_local_id;
        for (uint i = 0; pos < N_ELEMS && i < VECT_DT_N; i++) {
            output[pos] = TO_DST(sum[i]);
            pos += max_sub_group_size;
        }
    }
}
