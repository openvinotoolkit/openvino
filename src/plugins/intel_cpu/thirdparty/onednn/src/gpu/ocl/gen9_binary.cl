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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#undef DST_OFF
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

#if SRC0_DT_S8
#define SRC0_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC_DT_S8

#if SRC1_DT_S8
#define SRC1_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC_DT_S8

#if SRC0_DT_U8
#define SRC0_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC0_DT_U8

#if SRC1_DT_U8
#define SRC1_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC1_DT_U8

#if SRC0_DT_F16
#define SRC0_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC0_DT_F16

#if SRC1_DT_F16
#define SRC1_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC1_DT_F16

#if SRC0_DT_S32
#define SRC0_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC0_DT_S32

#if SRC1_DT_S32
#define SRC1_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC1_DT_S32

#if SRC0_DT_F32
#define SRC0_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC0_DT_F32

#if SRC1_DT_F32
#define SRC1_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC1_DT_F32

#if SRC0_DT_BF16
#define SRC0_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC0_DT_BF16

#if SRC1_DT_BF16
#define SRC1_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC1_DT_BF16

#if DST_DT_S8
#define DST_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_uc4((__global uchar *)(dst), as_uchar4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_uc4((__global uchar *)(dst), as_uchar4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_us4((__global ushort *)(dst), as_ushort4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write4((__global uint *)(dst), as_uint4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write4((__global uint *)(dst), as_uint4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_us4((__global ushort *)(dst), as_ushort4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if NVECT == 1
#define VECT_INT_TO_FLOAT convert_float
#else
#define VECT_INT_TO_FLOAT CONCAT2(convert_float, NVECT)
#endif

#if !IS_NCX_LAYOUT
#if !PLAIN_TO_ABCD4AXB
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        float src0_scale, float src1_scale) {

    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();

    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src0 += src0_off;
    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;

    int src1_off = SRC1_OFF(dims0[0] * (!BCAST_DIM0), dims0[1] * (!BCAST_DIM1),
            dims0[2] * (!BCAST_DIM2), dims0[3] * (!BCAST_DIM3),
            dims0[4] * (!BCAST_DIM4), dims0[5] * (!BCAST_DIM5));
    src1 += src1_off;
#if NVECT == 1
    float d = 0;
    float dst_data;
    float tmp_src0 = CONVERT_FLOAT_T(SRC0_BLOCK_READ(&src0[0]));
#elif NVECT == 2
    float2 d = 0;
    float2 dst_data;
    float2 tmp_src0 = CONVERT_FLOAT2_T(SRC0_BLOCK_READ2(&src0[0]));
#elif NVECT == 4
    float4 d = 0;
    float4 dst_data;
    float4 tmp_src0 = CONVERT_FLOAT4_T(SRC0_BLOCK_READ4(&src0[0]));
#elif NVECT == 8
    float8 d = 0;
    float8 dst_data;
    float8 tmp_src0 = CONVERT_FLOAT8_T(SRC0_BLOCK_READ8(&src0[0]));
#endif

#if BCAST_DIM1
    float tmp_src1 = CONVERT_FLOAT_T(src1[0]);
#else
#if BCAST_AT_INNERMOST_DIM == 1 || NVECT == 1
    float tmp_src1 = CONVERT_FLOAT_T(SRC1_BLOCK_READ(&src1[0]));
#elif NVECT == 2
    float2 tmp_src1 = CONVERT_FLOAT2_T(SRC1_BLOCK_READ2(&src1[0]));
#elif NVECT == 4
    float4 tmp_src1 = CONVERT_FLOAT4_T(SRC1_BLOCK_READ4(&src1[0]));
#elif NVECT == 8
    float8 tmp_src1 = CONVERT_FLOAT8_T(SRC1_BLOCK_READ8(&src1[0]));
#endif
#endif

#if WITH_SRC0_SCALE
    tmp_src0 = tmp_src0 * src0_scale;
#endif
#if WITH_SRC1_SCALE
    tmp_src1 = tmp_src1 * src1_scale;
#endif

#if IS_ADD
    d = tmp_src0 + tmp_src1;
#elif IS_MUL
    d = tmp_src0 * tmp_src1;
#elif IS_MAX
    d = max(tmp_src0, tmp_src1);
#elif IS_MIN
    d = min(tmp_src0, tmp_src1);
#elif IS_DIV
    d = tmp_src0 / tmp_src1;
#elif IS_SUB
    d = tmp_src0 - tmp_src1;
#elif IS_GE
    d = VECT_INT_TO_FLOAT(tmp_src0 >= tmp_src1);
#elif IS_GT
    d = VECT_INT_TO_FLOAT(tmp_src0 > tmp_src1);
#elif IS_LE
    d = VECT_INT_TO_FLOAT(tmp_src0 <= tmp_src1);
#elif IS_LT
    d = VECT_INT_TO_FLOAT(tmp_src0 < tmp_src1);
#elif IS_EQ
    d = VECT_INT_TO_FLOAT(tmp_src0 == tmp_src1);
#elif IS_NE
    d = VECT_INT_TO_FLOAT(tmp_src0 != tmp_src1);
#endif

#if WITH_SUM
#if NVECT == 1
    dst_data = CONVERT_FLOAT_T(DST_BLOCK_READ(&dst[0]));
#elif NVECT == 2
    dst_data = CONVERT_FLOAT2_T(DST_BLOCK_READ2(&dst[0]));
#elif NVECT == 4
    dst_data = CONVERT_FLOAT4_T(DST_BLOCK_READ4(&dst[0]));
#elif NVECT == 8
    dst_data = CONVERT_FLOAT8_T(DST_BLOCK_READ8(&dst[0]));
#endif
#endif

    const int po_mb = dims0[0];
    const int po_oc = dims0[1] + get_sub_group_local_id();
#if NVECT == 1
    APPLY_POST_OPS_SERIAL(d, float, dst_data, float, po_mb, 1, po_oc, 1,
            dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
#else
    for (int vidx = 0; vidx < NVECT; ++vidx) {
        float d_i = d[vidx];
        float dst_i = dst_data[vidx];
        APPLY_POST_OPS_SERIAL(d_i, float, dst_i, float, po_mb, 1, po_oc, 1,
                dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
        d[vidx] = d_i;
        ++dims0[NDIMS - 1];
    }
#endif

#if NVECT == 1
    DST_BLOCK_WRITE(&dst[0], TO_DST(d));
#elif NVECT == 2
    DST_BLOCK_WRITE2(&dst[0], TO_DST2(d));
#elif NVECT == 4
    DST_BLOCK_WRITE4(&dst[0], TO_DST4(d));
#elif NVECT == 8
    DST_BLOCK_WRITE8(&dst[0], TO_DST8(d));
#endif
}

#else // !PLAIN_TO_ABCD4AXB
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        float src0_scale, float src1_scale) {

    src0 += SRC0_OFFSET0;
    src1 += SRC1_OFFSET0;
    dst += DST_OFFSET0;

    int sglid = get_sub_group_local_id();

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D3();
    const int d5 = GWS_GET_D3();

    const int d0_block = GWS_GET_D0_BLOCK();
    const int d1_block = GWS_GET_D1_BLOCK();
    const int d01_block = d0_block * d1_block;

    SRC0_DATA_T tmp_buf0[d01_block] = {0};
    SRC1_DATA_T tmp_buf1[d01_block] = {0};
    DST_DATA_T res_buf[d01_block] = {0};

    const int d0_inner_block = min(d0_block, SRC0_D0);
    const int d1_inner_block = min(d1_block, SRC0_D1);
    for (int d0_inner = 0; d0_inner < d0_inner_block; d0_inner++) {
        for (int d1_inner = 0; d1_inner < d1_inner_block; d1_inner++) {
            if (SRC0_D0 % d0_inner_block != 0 && d0 + d0_inner >= SRC0_D0)
                continue;
            if (SRC0_D1 % d1_inner_block != 0 && d1 + d1_inner >= SRC0_D1)
                continue;
            int src0_off;
            int src1_off;
            if (SRC0_S3_0 == 1) {
                // abcd layout.
                src0_off = SRC0_OFF(d0 + d0_inner, d1 + d1_inner, d2, d3, 0, 0);
                tmp_buf0[d0_inner * d1_block + d1_inner]
                        = SRC0_BLOCK_READ(&src0[src0_off]);
                src1_off = SRC1_OFF((d0 + d0_inner) * (!BCAST_DIM0),
                        (d1 + d1_inner) * (!BCAST_DIM1), d2 * (!BCAST_DIM2),
                        d3 * (!BCAST_DIM3), 0, 0);
            } else {
                // acdb layout.
                src0_off = SRC0_OFF(
                        d0 + d0_inner, d1 + d1_inner, d2, d3 + sglid, 0, 0);
                tmp_buf0[d0_inner * d1_block + d1_inner] = src0[src0_off];
                src1_off = SRC1_OFF((d0 + d0_inner) * (!BCAST_DIM0),
                        (d1 + d1_inner) * (!BCAST_DIM1), d2 * (!BCAST_DIM2),
                        (d3 + sglid) * (!BCAST_DIM3), 0, 0);
            }
#if BCAST_AT_INNERMOST_DIM == 1
            tmp_buf1[d0_inner * d1_block + d1_inner] = src1[src1_off];
#else
            tmp_buf1[d0_inner * d1_block + d1_inner]
                    = SRC1_BLOCK_READ(&src1[src1_off]);
#endif //BCAST_AT_INNERMOST_DIM
        }
    }

    int i = 0;
    for (int d0_i = 0; d0_i < d0_block; d0_i++) {
        for (int d1_i = 0; d1_i < d1_block; d1_i++) {

            float tmp_src0 = CONVERT_FLOAT_T(tmp_buf0[i]);
            float tmp_src1 = CONVERT_FLOAT_T(tmp_buf1[i]);
            float res;
            float dst_data;

#if WITH_SRC0_SCALE
            tmp_src0 = tmp_src0 * src0_scale;
#endif
#if WITH_SRC1_SCALE
            tmp_src1 = tmp_src1 * src1_scale;
#endif

#if IS_ADD
            res = tmp_src0 + tmp_src1;
#elif IS_MUL
            res = tmp_src0 * tmp_src1;
#elif IS_MAX
            res = max(tmp_src0, tmp_src1);
#elif IS_MIN
            res = min(tmp_src0, tmp_src1);
#elif IS_DIV
            res = tmp_src0 / tmp_src1;
#elif IS_SUB
            res = tmp_src0 - tmp_src1;
#elif IS_GE
            res = tmp_src0 >= tmp_src1;
#elif IS_GT
            res = tmp_src0 > tmp_src1;
#elif IS_LE
            res = tmp_src0 <= tmp_src1;
#elif IS_LT
            res = tmp_src0 < tmp_src1;
#elif IS_EQ
            res = tmp_src0 == tmp_src1;
#elif IS_NE
            res = tmp_src0 != tmp_src1;
#endif

            APPLY_POST_OPS_SERIAL(res, float, dst_data, float, d0 + d0_i, 1,
                    d1 + d1_i, 1, d2, 1, d3 + sglid, 1, d4, 1, d5, 1);

            res_buf[i] = TO_DST(res);
            ++i;
        }
    }

    DST_DATA_T res_all[d01_block][SUB_GROUP_SIZE];
    for (int i = 0; i < d01_block; i++)
        for (int j = 0; j < SUB_GROUP_SIZE; j++)
            res_all[i][j] = intel_sub_group_shuffle(res_buf[i], j);
    for (int d = 0; d < SUB_GROUP_SIZE; d += 8) {
        DST_DATA8_T res_tmp;
        for (int i = 0; i < 8; i++)
            res_tmp[i] = res_all[sglid][d + i];
        int dst_off = DST_OFF(d0, d1, d2, d3 + d, 0, 0);

        DST_BLOCK_WRITE8(&dst[dst_off], res_tmp);
    }
}

#endif // !PLAIN_TO_ABCD4AXB

#else // #if !IS_NCX_LAYOUT

KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        float src0_scale, float src1_scale) {

    int dims0[6] = {0};

    unsigned mid_dim = GWS_GET_MIXED_DIM();
    dims0[5] = mid_dim % DST_D5;
    mid_dim /= DST_D5;
    dims0[4] = mid_dim % DST_D4;
    mid_dim /= DST_D4;
    dims0[3] = mid_dim % DST_D3;
    mid_dim /= DST_D3;
    dims0[2] = mid_dim % DST_D2;
    mid_dim /= DST_D2;
    dims0[1] = mid_dim % DST_D1;
    mid_dim /= DST_D1;
    dims0[0] = mid_dim;

    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src0 += src0_off;

    int src1_off = SRC1_OFF(dims0[0] * (!BCAST_DIM0), dims0[1] * (!BCAST_DIM1),
            dims0[2] * (!BCAST_DIM2), dims0[3] * (!BCAST_DIM3),
            dims0[4] * (!BCAST_DIM4), dims0[5] * (!BCAST_DIM5));
    src1 += src1_off;

    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;

#if WITH_SRC0_SCALE
#define src0_scale_val src0_scale
#else
#define src0_scale_val 1
#endif
#if WITH_SRC1_SCALE
#define src1_scale_val src1_scale
#else
#define src1_scale_val 1
#endif

#define READ_DATA(size, name, source_ptr, dest_ptr, scale) \
    { \
        unsigned offset = 0; \
        unroll_for(unsigned j8 = 0; j8 < size / 8; ++j8) { \
            *((float8 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT8_T(CONCAT2(name, _BLOCK_READ8)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 8; \
        } \
        if ((size % 8) / 4) { \
            *((float4 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT4_T(CONCAT2(name, _BLOCK_READ4)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 4; \
        } \
        if ((size % 4) / 2) { \
            *((float2 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT2_T(CONCAT2(name, _BLOCK_READ2)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 2; \
        } \
        if ((size % 2)) { \
            *((float *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT_T(CONCAT2(name, _BLOCK_READ)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
        } \
    }

    float tmp_src0[NVECT];
    READ_DATA(NVECT, SRC0, (&src0[0]), (&tmp_src0[0]), src0_scale_val);

#if BCAST_AT_INNERMOST_DIM
    float tmp_src1[1];
    tmp_src1[0] = src1_scale_val * CONVERT_FLOAT_T(src1[0]);
#define SRC1_IDX_MASK 0
#else
    float tmp_src1[NVECT];
    READ_DATA(NVECT, SRC1, (&src1[0]), (&tmp_src1[0]), src1_scale_val);
#define SRC1_IDX_MASK 1
#endif

    float tmp[NVECT];
    unroll_for(unsigned idx = 0; idx < NVECT; ++idx) {
#if IS_ADD
        tmp[idx] = tmp_src0[idx] + tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_MUL
        tmp[idx] = tmp_src0[idx] * tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_MAX
        tmp[idx] = max(tmp_src0[idx], tmp_src1[idx * SRC1_IDX_MASK]);
#elif IS_MIN
        tmp[idx] = min(tmp_src0[idx], tmp_src1[idx * SRC1_IDX_MASK]);
#elif IS_DIV
        tmp[idx] = tmp_src0[idx] / tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_SUB
        tmp[idx] = tmp_src0[idx] - tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_GE
        tmp[idx] = tmp_src0[idx] >= tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_GT
        tmp[idx] = tmp_src0[idx] > tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_LE
        tmp[idx] = tmp_src0[idx] <= tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_LT
        tmp[idx] = tmp_src0[idx] < tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_EQ
        tmp[idx] = tmp_src0[idx] == tmp_src1[idx * SRC1_IDX_MASK];
#elif IS_NE
        tmp[idx] = tmp_src0[idx] != tmp_src1[idx * SRC1_IDX_MASK];
#endif
    }

    float dst_data[NVECT];
#if WITH_SUM
    READ_DATA(NVECT, DST, (&dst[0]), (&dst_data[0]), 1);
#endif
    dims0[NDIMS - 1] += get_sub_group_local_id();
    unroll_for(unsigned idx = 0; idx < NVECT; ++idx) {
        float d_i = tmp[idx];
        float dst_i = dst_data[idx];
        APPLY_POST_OPS_SERIAL(d_i, float, dst_i, float, dims0[0], 1, dims0[1],
                1, dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
        tmp[idx] = d_i;
        dims0[NDIMS - 1] += 16;
    }

#define WRITE_DATA(size, name, source_ptr, dest_ptr) \
    { \
        unsigned offset = 0; \
        unroll_for(unsigned j8 = 0; j8 < size / 8; ++j8) { \
            CONCAT2(name, _BLOCK_WRITE8) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST8(*((float8 *)(source_ptr + offset)))); \
            offset += 8; \
        } \
        if ((size % 8) / 4) { \
            CONCAT2(name, _BLOCK_WRITE4) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST4(*((float4 *)(source_ptr + offset)))); \
            offset += 4; \
        } \
        if ((size % 4) / 2) { \
            CONCAT2(name, _BLOCK_WRITE2) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST2(*((float2 *)(source_ptr + offset)))); \
            offset += 2; \
        } \
        if ((size % 2)) { \
            CONCAT2(name, _BLOCK_WRITE) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST(*((float *)(source_ptr + offset)))); \
        } \
    }
    WRITE_DATA(NVECT, DST, (&tmp[0]), (&dst[0]));
}

#endif // #if !IS_NCX_LAYOUT
