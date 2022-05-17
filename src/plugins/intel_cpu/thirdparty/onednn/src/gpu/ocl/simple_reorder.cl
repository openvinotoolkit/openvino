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

// Temporary W/A for bf16 problems in HW and compiler
#undef cl_future_bf16_cvt

#define DT_UNDEF 1
#include "gpu/ocl/ocl_types.h"

#include "gpu/ocl/ocl_math_utils.h"

#if SRC_DT_F16 || DST_DT_F16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(SRC, (x0), (x1), (x2), (x3), (x4), (x5))
#define DST_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(DST, (x0), (x1), (x2), (x3), (x4), (x5))

#define SRC_OFF_G(gr, x0, x1, x2, x3, x4) \
    OFF_MD(SRC, gr, (x0), (x1), (x2), (x3), (x4))
#define DST_OFF_G(gr, x0, x1, x2, x3, x4) \
    OFF_MD(DST, gr, (x0), (x1), (x2), (x3), (x4))

#if SRC_DT_S8
#define SRC_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_S8

#if SRC_DT_U8
#define SRC_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if SRC_DT_F16
#define SRC_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if SRC_DT_S32
#define SRC_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // SRC_DT_S32

#if SRC_DT_F32
#define SRC_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // SRC_DT_F32

#if SRC_DT_BF16
#define SRC_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if DST_DT_S8
#define DST_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if SRC_DT_BF16 && DST_DT_BF16
#define SRC_TO_DST(x) (x)
#define SRC_TO_DST8(x) (x)
#else
#define SRC_TO_DST(x) TO_DST(SRC_TO_REF(x))
#define SRC_TO_DST8(x) TO_DST8(SRC_TO_REF8(x))
#endif

#if WITH_SUM_A
#define REORDER(_dst, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _s = _a * _x; \
        _dst = TO_DST(_s); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _s = _a * _x; \
        _dst = TO_DST8(_s); \
    } while (0)

#elif WITH_SUM_AB
#define REORDER(_dst, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _y = DST_TO_REF(_dst); \
        const float _s = _a * _x + _b * _y; \
        _dst = TO_DST(_s); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _y = convert_float8(DST_TO_REF8(_dst)); \
        const float8 _s = _a * _x + _b * _y; \
        _dst = TO_DST8(_s); \
    } while (0)

#elif SCALE_QUANT
#define REORDER(_out, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _s = _a * _x + _b; \
        _out = TO_DST(_s); \
    } while (0)
#define REORDER8(_out, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _s = _a * _x + _b; \
        _out = TO_DST8(_s); \
    } while (0)

#else // WITH_SUM_AB == 0
#define REORDER(_dst, _src, _a, _b) \
    do { \
        _dst = SRC_TO_DST(_src); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        _dst = SRC_TO_DST8(_src); \
    } while (0)

#endif // WITH_SUM_AB

#if SCALE_QUANT

#define MASK_D(_d) ((SCALE_MASK >> _d) & 1)

#define SCALE_D0 (MASK_D(0) ? SRC_D0 : 1)
#define SCALE_D1 (MASK_D(1) ? SRC_D1 : 1)
#define SCALE_D2 (MASK_D(2) ? SRC_D2 : 1)
#define SCALE_D3 (MASK_D(3) ? SRC_D3 : 1)
#define SCALE_D4 (MASK_D(4) ? SRC_D4 : 1)
#define SCALE_D5 (MASK_D(5) ? SRC_D5 : 1)

#define SCALE_S0 (SCALE_D1 * SCALE_D2 * SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S1 (SCALE_D2 * SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S2 (SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S3 (SCALE_D4 * SCALE_D5)
#define SCALE_S4 (SCALE_D5)
#define SCALE_S5 (1)

#define SCALE_OFF(x0, x1, x2, x3, x4, x5) \
    ((x0)*SCALE_S0 * MASK_D(0) + (x1)*SCALE_S1 * MASK_D(1) \
            + (x2)*SCALE_S2 * MASK_D(2) + (x3)*SCALE_S3 * MASK_D(3) \
            + (x4)*SCALE_S4 * MASK_D(4) + (x5)*SCALE_S5 * MASK_D(5))

#endif // SCALE_QUANT

KERNEL_ATTR
__kernel void simple_reorder(__global SRC_DATA_T *restrict src,
        __global DST_DATA_T *restrict dst, float alpha, float beta,
        __global float *restrict scales) {

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

#if REF_REORDER
    const int d0 = GWS_GET_D0();
    const int d1_blk_start = GWS_GET_D1();
    const int d2_blk_start = GWS_GET_D2();
    const int d3_blk_start = GWS_GET_D3();
    const int d4_blk_start = GWS_GET_D4();
    const int d5_blk_start = GWS_GET_D5();

    const int d1_blk_end = d1_blk_start + GWS_GET_D1_BLOCK();
    const int d2_blk_end = d2_blk_start + GWS_GET_D2_BLOCK();
    const int d3_blk_end = d3_blk_start + GWS_GET_D3_BLOCK();
    const int d4_blk_end = d4_blk_start + GWS_GET_D4_BLOCK();
    const int d5_blk_end = d5_blk_start + GWS_GET_D5_BLOCK();

    for (int d1 = d1_blk_start; d1 < d1_blk_end; ++d1) {
        for (int d2 = d2_blk_start; d2 < d2_blk_end; ++d2) {
            for (int d3 = d3_blk_start; d3 < d3_blk_end; ++d3) {
                for (int d4 = d4_blk_start; d4 < d4_blk_end; ++d4) {
                    for (int d5 = d5_blk_start; d5 < d5_blk_end; ++d5) {
                        const int src_off = SRC_OFF(d0, d1, d2, d3, d4, d5);
                        const int dst_off = DST_OFF(d0, d1, d2, d3, d4, d5);
#if PAD_FILL_ZERO == 1
                        int pad_d0 = d0 >= SRC_D0;
                        int pad_d1 = NDIMS > 1 && d1 >= SRC_D1;
                        int pad_d2 = NDIMS > 2 && d2 >= SRC_D2;
                        int pad_d3 = NDIMS > 3 && d3 >= SRC_D3;
                        int pad_d4 = NDIMS > 4 && d4 >= SRC_D4;
                        int pad_d5 = NDIMS > 5 && d5 >= SRC_D5;
                        if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4
                                || pad_d5) {
                            dst[dst_off] = 0;
                            continue;
                        }
#endif
#if SCALE_QUANT
                        alpha = scales[SCALE_OFF(d0, d1, d2, d3, d4, d5)];
#endif
                        REORDER(dst[dst_off], src[src_off], alpha, beta);
                    }
                }
            }
        }
    }

#elif ALT_OFFSETS
    // This implementation uses two main features:
    // 1. Extremely simple offset calculations. It works on 2D-4D tensors with
    // plain format. Coordinates of up to 3 dimensions are passed unmodified
    // through get_global_id(). 4th dimension, if there is one, is passed as
    // block size.
    // 2. Uses work group size at least 8. If that would mean using nonuniform
    // work groups, number of work items will be padded to keep them uniform.
    const int d0 = get_global_id(0);
    const int d1 = get_global_id(1);
    const int d2 = get_global_id(2);
#ifdef LIMIT_MAX_D0
    // padding for uniform work groups, don't write there
    if (d0 >= LIMIT_MAX_D0) { return; }
#endif
    const int src_base = S0 * d0 + S1 * d1 + S2 * d2;
    const int dst_base = D0 * d0 + D1 * d1 + D2 * d2;

    for (int db = 0; db < BLK; ++db) {
        const int src_off = src_base + db * SB;
        const int dst_off = dst_base + db * DB;
        REORDER(dst[dst_off], src[src_off], alpha, beta);
    }

#elif PLAIN_xFxE_TO_ABCDEF
    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();

#if WITH_SUM_AB
#define SUM_OUTPUT 1
#else
#define SUM_OUTPUT 0
#endif

    const unsigned sglid = get_sub_group_local_id();

#define REORDER_BLOCK(block_size, src_memory, src_swap, src_offset) \
    { \
        unroll_for(unsigned sidx = 0; sidx < block_size; ++sidx) { \
            const unsigned src_off \
                    = SRC_OFF(d0, d1, d2, d3, d4, sidx + src_offset); \
            src_memory[sidx] = SRC_BLOCK_READ(&src[src_off]); \
        } \
        unroll_for(int j = 0; j < SUB_GROUP_SIZE; j++) \
                unroll_for(int i = 0; i < block_size; i++) { \
            unsigned x = (i + j * block_size) / SUB_GROUP_SIZE; \
            unsigned y = (i + j * block_size) % SUB_GROUP_SIZE; \
            unsigned sg_src = (i + j * block_size) / block_size; \
            src_swap[x][y] = intel_sub_group_shuffle(src_mem[i], sg_src); \
        } \
\
        DST_DATA_T dst_tmp; \
        unsigned dst_off; \
        if (block_size < 16) dst_off = DST_OFF(d0, d1, d2, d3, d4, 0); \
\
        unroll_for(unsigned sidx = 0; sidx < block_size; ++sidx) { \
            if (block_size >= 16) \
                dst_off = DST_OFF(d0, d1, d2, d3, d4 + sidx, src_offset); \
            if (SUM_OUTPUT) dst_tmp = DST_BLOCK_READ(&dst[dst_off]); \
            REORDER(dst_tmp, src_swap[sidx][sglid], alpha, beta); \
            DST_BLOCK_WRITE(&dst[dst_off], dst_tmp); \
            if (block_size < 16) dst_off += SUB_GROUP_SIZE; \
        } \
    }

#if DST_D5 > 16
    unsigned block_size = 16;
#else
    unsigned block_size = DST_D5;
#endif

    SRC_DATA_T src_mem[16];
    SRC_DATA_T src_all[16][SUB_GROUP_SIZE];

    REORDER_BLOCK(block_size, src_mem, src_all, d5);

#elif TRANSPOSE_NXN || LOCAL_NXN
    // Fast reorder in which a subgroup of N work items loads N disjoint sets
    // of N sequential addresses, transposes the NxN matrix and writes it back
    // as N sets of N addresses. N = 8 or 16. Sets are strided in src by the
    // dimansion that'll be last in dst and strided in dst by dimension that
    // was last in src.
    // One version uses intel_sub_group_read/write functions to perform
    // the transposition. It needs no auxiliary mem accesses, but uses a lot
    // of private registers.
    // The other version uses local memory to performs transposition. It
    // would be slower but uses few private registers, so it's actually faster
    // on Gen9 architecture where private data would otherwise spill over to
    // global mem.
    //
    // LocalNxN kernel shares local memory across work items. In general that
    // doesn't work without barriers. Here we take caution to only access
    // data that was produced by work items in the same *sub*group. Those
    // are guaranteed to execute simultaneously, so no barriers are needed.
#define BATCH_SIZE SUB_GROUP_SIZE
    int sgId = get_sub_group_local_id();
    int sg_off = get_sub_group_id() * SUB_GROUP_SIZE * BATCH_SIZE;

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();

    const int d0_block = GWS_GET_D0_BLOCK();
    const int d1_block = GWS_GET_D1_BLOCK();
    const int d2_block = GWS_GET_D2_BLOCK();
    const int d3_block = GWS_GET_D3_BLOCK();
    const int d4_block = GWS_GET_D4_BLOCK();
    const int d5_block = GWS_GET_D5_BLOCK();

#ifdef TRANSPOSE_NXN
    SRC_DATA_T src_buf[SUB_GROUP_SIZE];
    SRC_DATA_T dst_buf[SUB_GROUP_SIZE];
#else
    __local SRC_DATA_T tmp[SG_PER_WG * BATCH_SIZE];
#endif
    SRC_DATA_T send_buf;

#if PAD_FILL_ZERO == 1
    const int pad_d0 = d0 >= SRC_D0;
    const int pad_d1 = NDIMS > 1 && d1 >= SRC_D1;
    const int pad_d2 = NDIMS > 2 && d2 >= SRC_D2;
    const int pad_d3 = NDIMS > 3 && d3 >= SRC_D3;
    const int pad_d4 = NDIMS > 4 && d4 >= SRC_D4;
    const int pad_d5 = NDIMS > 5 && d5 >= SRC_D5;
    const int pad = pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5;
#else
    const int pad = 0;
#endif
    if (!pad) {
        for_(int d0i = 0; d0i < d0_block; d0i++)
        for_(int d1i = 0; d1i < d1_block; d1i++)
        for_(int d2i = 0; d2i < d2_block; d2i++)
        for_(int d3i = 0; d3i < d3_block; d3i++)
        for_(int d4i = 0; d4i < d4_block; d4i++)
        for (int d5i = 0; d5i < d5_block; d5i++) {
            const int iter = d0i + d1i + d2i + d3i + d4i + d5i;
            const int src_off = SRC_OFF(
                    d0 + d0i, d1 + d1i, d2 + d2i, d3 + d3i, d4 + d4i, d5 + d5i);
#ifdef TRANSPOSE_NXN
            src_buf[iter] = SRC_BLOCK_READ(&src[src_off]);
#else
            tmp[sg_off + iter * SUB_GROUP_SIZE + sgId]
                    = SRC_BLOCK_READ(&src[src_off]);
#endif
        }
#ifdef TRANSPOSE_NXN
        // Share and transpose. Each work item keeps 1 own value and
        // gets (N-1) values from other work items
        dst_buf[sgId] = src_buf[sgId];
        for (int i = 1; i < SUB_GROUP_SIZE; i++) {
            send_buf = src_buf[(i + sgId) % BATCH_SIZE];
            dst_buf[(BATCH_SIZE + sgId - i) % BATCH_SIZE]
                    = intel_sub_group_shuffle(
                            send_buf, (BATCH_SIZE + sgId - i) % BATCH_SIZE);
        }
#endif
    }
    for_(int d0i = 0; d0i < d0_block; d0i++)
    for_(int d1i = 0; d1i < d1_block; d1i++)
    for_(int d2i = 0; d2i < d2_block; d2i++)
    for_(int d3i = 0; d3i < d3_block; d3i++)
    for_(int d4i = 0; d4i < d4_block; d4i++)
    for (int d5i = 0; d5i < d5_block; d5i++) {
        const int iter = d0i + d1i + d2i + d3i + d4i + d5i;
#if DST_BLOCK_DIM == 0
        const int dst_off = DST_OFF(d0 + iter, d1, d2, d3, d4, d5);
#elif DST_BLOCK_DIM == 1
        const int dst_off = DST_OFF(d0, d1 + iter, d2, d3, d4, d5);
#elif DST_BLOCK_DIM == 2
        const int dst_off = DST_OFF(d0, d1, d2 + iter, d3, d4, d5);
#elif DST_BLOCK_DIM == 3
        const int dst_off = DST_OFF(d0, d1, d2, d3 + iter, d4, d5);
#elif DST_BLOCK_DIM == 4
        const int dst_off = DST_OFF(d0, d1, d2, d3, d4 + iter, d5);
#elif DST_BLOCK_DIM == 5
        const int dst_off = DST_OFF(d0, d1, d2, d3, d4, d5 + iter);
#endif
        DST_DATA_T dst_tmp;
        if (!pad) {
#if WITH_SUM_AB
            dst_tmp = DST_BLOCK_READ(&dst[dst_off]);
#endif
#if SCALE_QUANT
            alpha = scales[SCALE_OFF(d0, d1, d2, d3, d4, d5)];
#endif
#ifdef TRANSPOSE_NXN
            REORDER(dst_tmp, dst_buf[iter], alpha, beta);
#else
            send_buf = tmp[sg_off + sgId * SUB_GROUP_SIZE + iter];
            REORDER(dst_tmp, send_buf, alpha, beta);
#endif
        } else {
            dst_tmp = 0;
        }
        DST_BLOCK_WRITE(&dst[dst_off], dst_tmp);
    }

#elif REORDER_NCHW

#define BIGGER_THAN_16 (SRC_D1 >= 16)

    int sgId = get_sub_group_local_id();

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();

    const int d1_block = GWS_GET_D1_BLOCK();

    SRC_DATA_T src_buf[SUB_GROUP_SIZE];
    SRC_DATA_T dst_buf[SUB_GROUP_SIZE];
#if BIGGER_THAN_16
    SRC_DATA_T send_buf;
#else
    SRC_DATA_T exch_buf[d1_block][SUB_GROUP_SIZE];
#endif

#if BIGGER_THAN_16
#define STRIDE_S SRC_D1
#else
#define STRIDE_S 16
#endif
#define STRIDE_D (SRC_D2 * SRC_D3)

    for (int i = 0; i < d1_block; i++) {
        int src_off = SRC_OFF(d0, d1, d2, d3, 0, 0) + STRIDE_S * i;
        src_buf[i] = SRC_BLOCK_READ(&src[src_off]);
    }
#if BIGGER_THAN_16
    for (int i = 0; i < SUB_GROUP_SIZE; i++) {
        send_buf = src_buf[(i + sgId) % 16];
        dst_buf[(16 + sgId - i) % 16]
                = intel_sub_group_shuffle(send_buf, (16 + sgId - i) % 16);
    }
#else
    for (int i = 0; i < d1_block; i++) {
        for (int sg = 0; sg < SUB_GROUP_SIZE; sg++) {
            exch_buf[i][sg] = intel_sub_group_shuffle(src_buf[i], sg);
        }
    }
    for (int i = 0; i < d1_block; i++) {
        int ofs = i + sgId * d1_block;
        dst_buf[i] = exch_buf[ofs / SUB_GROUP_SIZE][ofs % SUB_GROUP_SIZE];
    }
#endif
    for (int i = 0; i < d1_block; i++) {
        int dst_off = DST_OFF(d0, d1, d2, d3, 0, 0) + STRIDE_D * i;
        DST_DATA_T dst_tmp;
#if WITH_SUM_AB
        dst_tmp = DST_BLOCK_READ(&dst[dst_off]);
#endif
        REORDER(dst_tmp, dst_buf[i], alpha, beta);
        DST_BLOCK_WRITE(&dst[dst_off], dst_tmp);
    }

#elif PLAIN_TO_ABCD84A42B
    const int sglid = get_sub_group_local_id();
    const int sg = get_sub_group_id();

#define SRC_D3_ALIGNED (SRC_D3 % SUB_GROUP_SIZE == 0)
#define IS_SRC_ABCD_LAYOUT (SRC_S3_0 == 1)
#define IS_SRC_ACDB_LAYOUT (SRC_S3_0 != 1)

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();

    const int d0_block = GWS_GET_D0_BLOCK();
    const int d1_block = GWS_GET_D1_BLOCK();
    const int d01_block = d0_block * d1_block;

    __local SRC_DATA_T loc_buf[SG_PER_WG][d01_block][SUB_GROUP_SIZE];
    const int d0_inner_block = min(d0_block, SRC_D0);
    const int d1_inner_block = min(d1_block, SRC_D1);
    for (int d0_inner = 0; d0_inner < d0_inner_block; d0_inner++) {
        for (int d1_inner = 0; d1_inner < d1_inner_block; d1_inner++) {

            if (SRC_D0 % d0_inner_block != 0 && d0 + d0_inner >= SRC_D0) {
                loc_buf[sg][d0_inner * d1_block + d1_inner][sglid] = 0;
                continue;
            }
            if (SRC_D1 % d1_inner_block != 0 && d1 + d1_inner >= SRC_D1) {
                loc_buf[sg][d0_inner * d1_block + d1_inner][sglid] = 0;
                continue;
            }
            if (IS_SRC_ABCD_LAYOUT
                    && (SRC_D3_ALIGNED || d3 + SUB_GROUP_SIZE < SRC_D3)) {
                const int src_off
                        = SRC_OFF(d0 + d0_inner, d1 + d1_inner, d2, d3, 0, 0);
                loc_buf[sg][d0_inner * d1_block + d1_inner][sglid]
                        = src[src_off + sglid];
            } else {
                if (SRC_D3_ALIGNED || d3 + sglid < SRC_D3) {
                    const int src_off = SRC_OFF(
                            d0 + d0_inner, d1 + d1_inner, d2, d3 + sglid, 0, 0);
                    loc_buf[sg][d0_inner * d1_block + d1_inner][sglid]
                            = src[src_off];
                }
            }
        }
    }
    if (SRC_D3_ALIGNED || d3 + SUB_GROUP_SIZE < SRC_D3) {
        for (int d = 0; d < SUB_GROUP_SIZE; d += 8) {
            SRC_DATA8_T src_tmp;
            for (int i = 0; i < 8; i++) {
                src_tmp[i] = loc_buf[sg][sglid][d + i];
            }
            const int dst_off = DST_OFF(d0, d1, d2, d3 + d, 0, 0);

            DST_DATA8_T dst_tmp;
#if WITH_SUM_AB
            for (int n = 0; n < 8; n++) {
                dst_tmp[n] = dst[dst_off + sglid + n * SUB_GROUP_SIZE];
            }
#endif // WITH_SUM_AB
            REORDER8(dst_tmp, src_tmp, alpha, beta);
            for (int n = 0; n < 8; n++) {
                dst[dst_off + sglid + n * SUB_GROUP_SIZE] = dst_tmp[n];
            }
        }
    } else {
        for (int d = 0; d < SUB_GROUP_SIZE; d++) {
            if (d3 + d < SRC_D3) {
                const SRC_DATA_T src_tmp = loc_buf[sg][sglid][d];
                const int dst_off = DST_OFF(d0, d1, d2, d3 + d, 0, 0);
                DST_DATA_T dst_tmp;
#if WITH_SUM_AB
                dst_tmp[n] = dst[dst_off + sglid];
#endif // WITH_SUM_AB
                REORDER(dst_tmp, src_tmp, alpha, beta);
                dst[dst_off + sglid] = dst_tmp;
            }
        }
    }

#elif PLAIN_TO_AB_XX_8AYB
    // Reorders 2D plain format to a blocked one, where last two
    // blocks are 8a4b or 8a2b. Supports formats with more block layers.
    //
    // Uses subgroup size 16
    // Each subgroup will read 8 sets of 16 values. Sets are not
    // adjacent in src, they are strided by 0th dim
    // All those 8*16 values will be shared between work items in subgroup
    // Each WI selects a set of 8 values out of 8*16 to write back
    // Each subgroup will write 8 sets of 16 values. Sets are adjacent in dst.
    //
    // TODO: make it generic across number of dimensions, for now only works with 2D
    // TODO: reduce shuffles from 8*16 to 28(?) - even though it doesn't improve perf
    // TODO: the two dst_buf<-tmp_buf formulas should be unified
    int sgId = get_sub_group_local_id();

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();

    const int d0b = GWS_GET_D0_BLOCK();
    const int d1b = GWS_GET_D1_BLOCK();

    SRC_DATA_T src_buf[d0b];
    DST_DATA_T dst_buf[d0b];

    for (int d0i = 0; d0i < d0b; ++d0i) {
        const int src_off = SRC_OFF(d0 + d0i, d1, 0, 0, 0, 0);
        src_buf[d0i] = SRC_BLOCK_READ(&src[src_off]);
    }

    SRC_DATA_T tmp_buf[d0b][SUB_GROUP_SIZE];
    for (int i = 0; i < d0b; i++) {
        for (int sg = 0; sg < SUB_GROUP_SIZE; sg++) {
            tmp_buf[i][sg] = intel_sub_group_shuffle(src_buf[i], sg);
        }
    }
#if BLK_L == 4
    for (int d0i = 0; d0i < d0b; ++d0i) {
        dst_buf[d0i] = tmp_buf[(d0i % 2 * BLK_L) + sgId / BLK_L]
                              [(d0i / 2) * BLK_L + sgId % BLK_L];
    }
#else // BLK_L == 2
    for (int d0i = 0; d0i < d0b; ++d0i) {
        dst_buf[d0i] = tmp_buf[sgId / BLK_L][d0i * BLK_L + sgId % BLK_L];
    }
#endif
    for (int d0i = 0; d0i < d0b; ++d0i) {
        const int dst_off = DST_OFF(d0, d1, 0, 0, 0, 0) + SUB_GROUP_SIZE * d0i;

        DST_DATA_T dst_tmp;
#if WITH_SUM_AB
        dst_tmp = DST_BLOCK_READ(&dst[dst_off]);
#endif
        REORDER(dst_tmp, dst_buf[d0i], alpha, beta);
        DST_BLOCK_WRITE(&dst[dst_off], dst_tmp);
    }

#elif VECTORIZE_LAST_DIM

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();

    const int d0_block = GWS_GET_D0_BLOCK();
    const int d1_block = GWS_GET_D1_BLOCK();
    const int d2_block = GWS_GET_D2_BLOCK();
    const int d3_block = GWS_GET_D3_BLOCK();
    const int d4_block = GWS_GET_D4_BLOCK();

    for_(int d0i = 0; d0i < d0_block; d0i++)
    for_(int d1i = 0; d1i < d1_block; d1i++)
    for_(int d2i = 0; d2i < d2_block; d2i++)
    for_(int d3i = 0; d3i < d3_block; d3i++)
    for (int d4i = 0; d4i < d4_block; d4i++) {

        int src_off
                = SRC_OFF(d0 + d0i, d1 + d1i, d2 + d2i, d3 + d3i, d4 + d4i, d5);
        SRC_DATA_T src_tmp = SRC_BLOCK_READ(&src[src_off]);

        int dst_off
                = DST_OFF(d0 + d0i, d1 + d1i, d2 + d2i, d3 + d3i, d4 + d4i, d5);
        DST_DATA_T dst_tmp;
#if WITH_SUM_AB
        dst_tmp = DST_BLOCK_READ(&dst[dst_off]);
#endif
        REORDER(dst_tmp, src_tmp, alpha, beta);
        DST_BLOCK_WRITE(&dst[dst_off], dst_tmp);
    }

#elif PAD_INNERMOST
    const int sgId = get_sub_group_local_id();
    int d[6];
    int blk[6];
    int b[6] = {0, 0, 0, 0, 0, 0};

    d[0] = GWS_GET_D0();
    d[1] = GWS_GET_D1();
    d[2] = GWS_GET_D2();
    d[3] = GWS_GET_D3();
    d[4] = GWS_GET_D4();
    d[5] = GWS_GET_D5();
    blk[0] = GWS_GET_D0_BLOCK();
    blk[1] = GWS_GET_D1_BLOCK();
    blk[2] = GWS_GET_D2_BLOCK();
    blk[3] = GWS_GET_D3_BLOCK();
    blk[4] = GWS_GET_D4_BLOCK();
    blk[5] = GWS_GET_D5_BLOCK();

    __local SRC_DATA_T cache[SG_PER_WG * GROUP * GROUP * VECT_SIZE];

    // offset to local memory for given subgroup
    const int sg_off = get_sub_group_id() * VECT_SIZE * (GROUP * GROUP);

    for (int i = 0; i < blk[SRC_LOOP_DIM]; i++) {
        b[SRC_LOOP_DIM] = i;
        const int src_off = SRC_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);
        unroll_for(int j = 0; j < GROUP; j++) {
            const int coff = sg_off + VECT_SIZE * GROUP * i + VECT_SIZE * j;
            const int soff = src_off + INNERMOST_SIZE * j;
#if NON_INNERMOST_PADDING == 1
            const int pad_d0 = d[0] + b[0] >= SRC_D0;
            const int pad_d1 = NDIMS > 1 && d[1] + b[1] >= SRC_D1;
            const int pad_d2 = NDIMS > 2 && d[2] + b[2] >= SRC_D2;
            const int pad_d3 = NDIMS > 3 && d[3] + b[3] >= SRC_D3;
            const int pad_d4 = NDIMS > 4 && d[4] + b[4] >= SRC_D4;
            const int pad_d5 = NDIMS > 5 && d[5] + b[5] >= SRC_D5;
            if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
                cache[coff + sgId] = 0;
                continue;
            }
#endif
            if (sgId < INNERMOST_SIZE) {
                cache[coff + sgId] = src[soff + sgId];
            } else {
                cache[coff + sgId] = 0;
            }
        }
    }
    b[SRC_LOOP_DIM] = 0;
    for (int i = 0; i < blk[DST_LOOP_DIM]; i++) {
        b[DST_LOOP_DIM] = i;
        const int dst_off = DST_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);
#if SCALE_QUANT
        alpha = scales[SCALE_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5])];
#endif
        unroll_for(int j = 0; j < GROUP; j++) {
            const int coff = sg_off + VECT_SIZE * GROUP * j + VECT_SIZE * i;
            const int doff = dst_off + VECT_SIZE * j;
            DST_DATA_T dst_tmp;
#if WITH_SUM_AB
            dst_tmp = dst[doff + sgId];
#endif
            REORDER(dst_tmp, cache[coff + sgId], alpha, beta);
            dst[doff + sgId] = dst_tmp;
        }
    }

#elif VECTORIZE_GROUPS

    int d[6];
    int blk[6];
    int b[6] = {0, 0, 0, 0, 0, 0};

    d[0] = GWS_GET_D0();
    d[1] = GWS_GET_D1();
    d[2] = GWS_GET_D2();
    d[3] = GWS_GET_D3();
    d[4] = GWS_GET_D4();
    d[5] = GWS_GET_D5();
    blk[0] = GWS_GET_D0_BLOCK();
    blk[1] = GWS_GET_D1_BLOCK();
    blk[2] = GWS_GET_D2_BLOCK();
    blk[3] = GWS_GET_D3_BLOCK();
    blk[4] = GWS_GET_D4_BLOCK();
    blk[5] = GWS_GET_D5_BLOCK();

    SRC_DATA_T cache[GROUP * GROUP];
    // there will be blocks on 2 dimensions
    // order of loops is important: one dim gets adjacent data, the other does not
    // that order will be inverted in dst's loops
    // There will be GROUP reads of SIMD{VECT} items, each with
    // sizeof(SRC_DATA_T) bytes, and they all point to contiguous mem area.
    // GROUP and VECT must be selected in such way that resulting mem
    // accesses could be combined into full cache line accesses by mem controller
    int coeff = (VECT_DIM == SRC_LOOP_DIM ? get_sub_group_size() : 1);
    for (int i = 0; i < blk[SRC_LOOP_DIM]; i++) {
        b[SRC_LOOP_DIM] = coeff * i;
        const int src_off = SRC_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);
        unroll_for(int j = 0; j < GROUP; j++) {
            int cidx = GROUP * i + j;
            int sidx = src_off + get_sub_group_size() * j;
#if PAD_FILL_ZERO == 1
            int pad_d0 = d[0] + b[0] >= SRC_D0;
            int pad_d1 = NDIMS > 1 && d[1] + b[1] >= SRC_D1;
            int pad_d2 = NDIMS > 2 && d[2] + b[2] >= SRC_D2;
            int pad_d3 = NDIMS > 3 && d[3] + b[3] >= SRC_D3;
            int pad_d4 = NDIMS > 4 && d[4] + b[4] >= SRC_D4;
            int pad_d5 = NDIMS > 5 && d[5] + b[5] >= SRC_D5;
            if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
                cache[cidx] = 0;
                continue;
            }
#endif
            cache[cidx] = SRC_BLOCK_READ(&src[sidx]);
        }
    }
    b[SRC_LOOP_DIM] = 0;
    coeff = (VECT_DIM == DST_LOOP_DIM ? get_sub_group_size() : 1);
    for (int i = 0; i < blk[DST_LOOP_DIM]; i++) {
        b[DST_LOOP_DIM] = coeff * i;
        const int dst_off = DST_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);

        int pad_d0 = d[0] + b[0] >= DST_PD0;
        int pad_d1 = NDIMS > 1 && d[1] + b[1] >= DST_PD1;
        int pad_d2 = NDIMS > 2 && d[2] + b[2] >= DST_PD2;
        int pad_d3 = NDIMS > 3 && d[3] + b[3] >= DST_PD3;
        int pad_d4 = NDIMS > 4 && d[4] + b[4] >= DST_PD4;
        int pad_d5 = NDIMS > 5 && d[5] + b[5] >= DST_PD5;
        if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
            continue;
        }

        unroll_for(int j = 0; j < GROUP; j++) {
            int cidx = i + j * GROUP;
            int didx = dst_off + get_sub_group_size() * j;
            DST_DATA_T dst_tmp;
#if WITH_SUM_AB
            dst_tmp = DST_BLOCK_READ(&dst[didx]);
#endif
#if SCALE_QUANT
            alpha = scales[SCALE_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                    d[3] + b[3], d[4] + b[4], d[5] + b[5])];
#endif

            REORDER(dst_tmp, cache[cidx], alpha, beta);
            DST_BLOCK_WRITE(&dst[didx], dst_tmp);
        }
    }

#elif USE_DENSE_VECT
    const int d0_blk_start = GWS_GET_D0();
    const int d0_blk_end = d0_blk_start + (GWS_GET_D0_BLOCK() * 16);
    for (int d0 = d0_blk_start; d0 < d0_blk_end; d0 += 128) {
        SRC_DATA8_T src_tmp = SRC_BLOCK_READ8(&src[d0]);
        DST_DATA8_T dst_tmp;
#if WITH_SUM_AB
        dst_tmp = DST_BLOCK_READ8(&dst[d0]);
#endif
        REORDER8(dst_tmp, src_tmp, alpha, beta);
        DST_BLOCK_WRITE8(&dst[d0], dst_tmp);
    }

// xb -> xab and xb -> xba, where size of dst's innermost dim is less than 16
#elif XAB_XBA
    const int sgId = get_sub_group_local_id();
    const int sgNr = get_sub_group_id();
    const int wgId
            = get_group_id(0) + 10 * get_group_id(1) + 100 * get_group_id(2);

    int d[6];
    int blk[6];
    int b[6] = {0, 0, 0, 0, 0, 0};

    d[0] = GWS_GET_D0();
    d[1] = GWS_GET_D1();
    d[2] = GWS_GET_D2();
    d[3] = GWS_GET_D3();
    d[4] = GWS_GET_D4();
    d[5] = GWS_GET_D5();
    blk[0] = GWS_GET_D0_BLOCK();
    blk[1] = GWS_GET_D1_BLOCK();
    blk[2] = GWS_GET_D2_BLOCK();
    blk[3] = GWS_GET_D3_BLOCK();
    blk[4] = GWS_GET_D4_BLOCK();
    blk[5] = GWS_GET_D5_BLOCK();

    __local SRC_DATA_T tmp[SG_PER_WG * SUB_GROUP_SIZE * BLOCK_SIZE];
    SRC_DATA_T data;
    const int sgLs = SUB_GROUP_SIZE * BLOCK_SIZE;
    const int wg_off = sgNr * sgLs;
    for (int i = 0; i < blk[SRC_BLK_DIM]; i++) {
        b[SRC_BLK_DIM] = SRC_OFF_COEFF * i;
        const int src_off = SRC_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);
#if PAD_FILL_ZERO == 1
        const int pad_d0 = d[0] + b[0] >= SRC_D0;
        const int pad_d1 = NDIMS > 1 && d[1] + b[1] >= SRC_D1;
        const int pad_d2 = NDIMS > 2 && d[2] + b[2] >= SRC_D2;
        const int pad_d3 = NDIMS > 3 && d[3] + b[3] >= SRC_D3;
        const int pad_d4 = NDIMS > 4 && d[4] + b[4] >= SRC_D4;
        const int pad_d5 = NDIMS > 5 && d[5] + b[5] >= SRC_D5;
        if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
            tmp[wg_off + SUB_GROUP_SIZE * i + sgId] = 0;
            continue;
        }
#endif
        data = src[src_off + sgId];
        tmp[wg_off + SUB_GROUP_SIZE * i + sgId] = data;
    }
    b[SRC_BLK_DIM] = 0;

    for (int i = 0; i < blk[SRC_BLK_DIM]; i++) {
        b[DST_BLK_DIM] = DST_OFF_COEFF * i;
        const int dst_off = DST_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5]);
#if XB_TO_XAB
        SRC_DATA_T data = tmp[wg_off + (sgId % (SUB_GROUP_SIZE / BLOCK_SIZE))
                + (sgId / (SUB_GROUP_SIZE / BLOCK_SIZE)) * SUB_GROUP_SIZE
                + (SUB_GROUP_SIZE / BLOCK_SIZE) * i];
#else // XB_TO_XBA
        SRC_DATA_T data = tmp[wg_off + (sgId % BLOCK_SIZE) * SUB_GROUP_SIZE
                + sgId / BLOCK_SIZE + (SUB_GROUP_SIZE / BLOCK_SIZE) * i];
#endif

#if SCALE_QUANT
        alpha = scales[SCALE_OFF(d[0] + b[0], d[1] + b[1], d[2] + b[2],
                d[3] + b[3], d[4] + b[4], d[5] + b[5])];
#endif
        DST_DATA_T dst_tmp;
#if WITH_SUM_AB
        dst_tmp = dst[dst_off + sgId];
#endif
        REORDER(dst_tmp, data, alpha, beta);
        dst[dst_off + sgId] = dst_tmp;
    }

//###################################################################################
#else // unroll_* kernels start here

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();
    const int local_id = get_sub_group_local_id();

// unroll_16a16b
#if SRC_16A16B || DST_16A16B || SRC_16B16A || DST_16B16A
    src += SRC_OFF(d0, d1, d2, d3, d4, d5);
    dst += DST_OFF(d0, d1, d2, d3, d4, d5);

    SRC_DATA8_T in0, in1;
#if SRC_16A16B || SRC_16B16A
    in0 = SRC_BLOCK_READ8(&src[0]);
    in1 = SRC_BLOCK_READ8(&src[8 * 16]);
#else
    for (int i = 0; i < 8; i++) {
#if DST_16B16A
        in0[i] = src[SRC_OFF(local_id, i, 0, 0, 0, 0)];
        in1[i] = src[SRC_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        in0[i] = src[SRC_OFF(i, local_id, 0, 0, 0, 0)];
        in1[i] = src[SRC_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // DST_16B16A
    }
#endif // SRC_16A16B || SRC_16B16A

    DST_DATA8_T dst0, dst1;
#if (SRC_16A16B || SRC_16B16A) && (DST_16A16B || DST_16B16A)
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if SRC_16B16A
        dst0[i] = dst[DST_OFF(local_id, i + 0, 0, 0, 0, 0)];
        dst1[i] = dst[DST_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        dst0[i] = dst[DST_OFF(i + 0, local_id, 0, 0, 0, 0)];
        dst1[i] = dst[DST_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // SRC_16B16A
    }
#endif // WITH_SUM_AB
#elif DST_16A16B || DST_16B16A
#if WITH_SUM_AB
    dst0 = DST_BLOCK_READ8(&dst[0]);
    dst1 = DST_BLOCK_READ8(&dst[8 * 16]);
#endif // WITH_SUM_AB
#else
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if SRC_16B16A
        dst0[i] = dst[DST_OFF(local_id, i + 0, 0, 0, 0, 0)];
        dst1[i] = dst[DST_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        dst0[i] = dst[DST_OFF(i + 0, local_id, 0, 0, 0, 0)];
        dst1[i] = dst[DST_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // SRC_16B16A
    }
#endif // WITH_SUM_AB
#endif // (SRC_16A16B || SRC_16B16A) && (DST_16A16B || DST_16B16A)

    REORDER8(dst0, in0, alpha, beta);
    REORDER8(dst1, in1, alpha, beta);

#if (SRC_16A16B || SRC_16B16A) && (DST_16A16B || DST_16B16A)
    for (int i = 0; i < 8; i++) {
#if SRC_16B16A
        dst[DST_OFF(local_id, i + 0, 0, 0, 0, 0)] = dst0[i];
        dst[DST_OFF(local_id, i + 8, 0, 0, 0, 0)] = dst1[i];
#else
        dst[DST_OFF(i + 0, local_id, 0, 0, 0, 0)] = dst0[i];
        dst[DST_OFF(i + 8, local_id, 0, 0, 0, 0)] = dst1[i];
#endif // SRC_16B16A
    }
#elif DST_16A16B || DST_16B16A
    DST_BLOCK_WRITE8(&dst[0], dst0);
    DST_BLOCK_WRITE8(&dst[8 * 16], dst1);
#else
    for (int i = 0; i < 8; i++) {
#if SRC_16B16A
        dst[DST_OFF(local_id, i + 0, 0, 0, 0, 0)] = dst0[i];
        dst[DST_OFF(local_id, i + 8, 0, 0, 0, 0)] = dst1[i];
#else
        dst[DST_OFF(i + 0, local_id, 0, 0, 0, 0)] = dst0[i];
        dst[DST_OFF(i + 8, local_id, 0, 0, 0, 0)] = dst1[i];
#endif // SRC_16B16A
    }
#endif // (SRC_16A16B || SRC_16B16A) && (DST_16A16B || DST_16B16A)

// unroll_16b
#elif SRC_16B || DST_16B
    SRC_DATA_T src_tmp;
#if SRC_16B
    src += SRC_OFF(d0, d1, d2, d3, d4, d5);
    src_tmp = SRC_BLOCK_READ(&src[0]);
#else
    src += SRC_OFF(d0, d1 + local_id, d2, d3, d4, d5);
    src_tmp = src[0];
#endif // SRC_16B

    DST_DATA_T dst_tmp;
#if DST_16B
    dst += DST_OFF(d0, d1, d2, d3, d4, d5);
#if WITH_SUM_AB
    dst_tmp = DST_BLOCK_READ(&dst[0]);
#endif // WITH_SUM_AB
#else
    dst += DST_OFF(d0, d1 + local_id, d2, d3, d4, d5);
#if WITH_SUM_AB
    dst_tmp = dst[0];
#endif // WITH_SUM_AB
#endif // DST_16B

    REORDER(dst_tmp, src_tmp, alpha, beta);

#if DST_16B
    DST_BLOCK_WRITE(&dst[0], dst_tmp);
#else
    dst[0] = dst_tmp;
#endif // DST_16B

// unroll_16b16c
#elif SRC_16B16C || DST_16B16C || SRC_16C16B || DST_16C16B
    const int g = d0;

    SRC_DATA8_T in0, in1;
#if SRC_16B16C || SRC_16C16B
    src += SRC_OFF_G(g, d1, d2, d3, d4, d5);
    in0 = SRC_BLOCK_READ8(&src[0]);
    in1 = SRC_BLOCK_READ8(&src[8 * 16]);
#else
    for (int i = 0; i < 8; i++) {
#if DST_16C16B
        in0[i] = src[SRC_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        in1[i] = src[SRC_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        in0[i] = src[SRC_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        in1[i] = src[SRC_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // DST_16C16B
    }
#endif // SRC_16B16C || SRC_16C16B

    DST_DATA8_T dst0, dst1;

#if (SRC_16B16C || SRC_16C16B) && (DST_16B16C || DST_16C16B)
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if SRC_16C16B
        dst0[i] = dst[DST_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        dst1[i] = dst[DST_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        dst0[i] = dst[DST_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        dst1[i] = dst[DST_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // SRC_16C16B
    }
#endif // WITH_SUM_AB
#elif DST_16B16C || DST_16C16B
    dst += DST_OFF_G(g, d1, d2, d3, d4, d5);
#if WITH_SUM_AB
    dst0 = DST_BLOCK_READ8(&dst[0]);
    dst1 = DST_BLOCK_READ8(&dst[8 * 16]);
#endif // WITH_SUM_AB
#else
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if SRC_16C16B
        dst0[i] = dst[DST_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        dst1[i] = dst[DST_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        dst0[i] = dst[DST_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        dst1[i] = dst[DST_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // SRC_16C16B
    }
#endif // WITH_SUM_AB
#endif // (SRC_16B16C || SRC_16C16B) && (DST_16B16C || DST_16C16B)

    REORDER8(dst0, in0, alpha, beta);
    REORDER8(dst1, in1, alpha, beta);

#if (SRC_16B16C || SRC_16C16B) && (DST_16B16C || DST_16C16B)
    for (int i = 0; i < 8; i++) {
#if SRC_16C16B
        dst[DST_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)] = dst0[i];
        dst[DST_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)] = dst1[i];
#else
        dst[DST_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)] = dst0[i];
        dst[DST_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)] = dst1[i];
#endif // SRC_16C16B
    }
#elif DST_16B16C || DST_16C16B
    DST_BLOCK_WRITE8(&dst[0], dst0);
    DST_BLOCK_WRITE8(&dst[8 * 16], dst1);
#else
    for (int i = 0; i < 8; i++) {
#if SRC_16C16B
        dst[DST_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)] = dst0[i];
        dst[DST_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)] = dst1[i];
#else
        dst[DST_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)] = dst0[i];
        dst[DST_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)] = dst1[i];
#endif // SRC_16C16B
    }
#endif // (SRC_16B16C || SRC_16C16B) && (DST_16B16C || DST_16C16B)
#endif // SRC_16B16C || DST_16B16C || SRC_16C16B || DST_16C16B

#endif // REF_REORDER, PLAIN_xFxE_TO_ABCDEF, TRANSPOSE_16X16 etc.
}
