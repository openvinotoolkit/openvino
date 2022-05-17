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

#if OC % OC_BLOCK != 0
#define OC_NBLOCKS_TAIL ((OC - (OC & ~(OC_BLOCK - 1)) + 3) / 4)
#else
#define OC_NBLOCKS_TAIL 8
#endif

#if IW_BLOCK == 4
#define BLOCK 4
#define ACC_DATA_BLOCK int4
#define A_DATA_BLOCK_T MMAD_DATA4_T
#define WRITE_LOCAL block_write4
#define READ_BLOCK intel_sub_group_block_read4
DECLARE_MMAD_EMU(mmad_tail, idot4, OC_NBLOCKS_TAIL, 4, A_DATA_BLOCK_T, int8,
        ACC_DATA_BLOCK)
#define MMAD_FULL mmad8x4
#define MMAD_TAIL mmad_tail

#elif IW_BLOCK == 8
#define BLOCK 8
#define ACC_DATA_BLOCK int8
#define A_DATA_BLOCK_T MMAD_DATA8_T
#define WRITE_LOCAL block_write8
#define READ_BLOCK intel_sub_group_block_read8

DECLARE_MMAD_EMU(mmad_tail, idot4, OC_NBLOCKS_TAIL, 8, A_DATA_BLOCK_T, int8,
        ACC_DATA_BLOCK)
#define MMAD_FULL mmad8x8
#define MMAD_TAIL mmad_tail
#else
#error "Wrong IW_BLOCK"
#endif

#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#if IC % IC_BLOCK == 0
#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#else
#define BLOCK_READ_BIA(data, idx) \
    data = (float4)0; \
    int i; \
    for (i = idx; i < idx + IC_BLOCK && i < IC - (IC % SUB_GROUP_SIZE); \
            i += SUB_GROUP_SIZE) { \
        data[(i - idx) / SUB_GROUP_SIZE] = as_float( \
                intel_sub_group_block_read((__global uint *)&bias[i])); \
    } \
    if ((get_sub_group_local_id() < IC % SUB_GROUP_SIZE) \
            && (i == IC - IC % SUB_GROUP_SIZE)) { \
        data[(i - idx) / SUB_GROUP_SIZE] \
                = as_float(bias[i + get_sub_group_local_id()]); \
    }

#endif // BLOCK_READ_BIA -> IC % IC_BLOCK != 0

#define HAS_PAD_W (PW > 0 || OW * SW - PW + (KW - 1) * (1 + DW) >= IW)

#if IS_NHWC
inline void write_ic_block4(__global SRC_DATA_T *src, int off, uchar4 value) {
    const int local_id = get_sub_group_local_id();
#if IC % IC_BLOCK != 0
    int tail = IC - off;
    if (tail < IC_BLOCK) {
        if (local_id < tail) src[8 * 0 + local_id] = value.s0;
        if (local_id < tail - 8 * 1) src[1 * 8 + local_id] = value.s1;
        if (local_id < tail - 8 * 2) src[2 * 8 + local_id] = value.s2;
        if (local_id < tail - 8 * 3) src[3 * 8 + local_id] = value.s3;
        return;
    }
#endif
#if IC % 4 != 0
    src[0 * 8 + local_id] = value.s0;
    src[1 * 8 + local_id] = value.s1;
    src[2 * 8 + local_id] = value.s2;
    src[3 * 8 + local_id] = value.s3;
    return;
#else
    intel_sub_group_block_write_uc4((__global uchar *)src, value);
    return;
#endif
}

inline void write_local(__local uint *dst_iw_slm_copy, int oc_nchunk,
        __global DATA_T *dst_copy) {
    const int local_id = get_sub_group_local_id();
#if OC % OC_BLOCK != 0
    int oc_block_tail = OC % OC_BLOCK;
    int oc_bound_tail = oc_block_tail % 4;
    int max_i = (local_id * 4 < (oc_block_tail - oc_bound_tail)
                        || oc_nchunk < (OC_NCHUNK - 1))
            ? 4
            : (local_id * 4 == (oc_block_tail - oc_bound_tail) ? oc_bound_tail
                                                               : 0);
    uchar4 tmp = 0;
    for (int i = 0; i < max_i; ++i) {
        tmp[i] = dst_copy[local_id * 4 + i];
    }
    dst_iw_slm_copy[local_id] = as_uint(tmp);
    return;
#endif
    block_write(dst_iw_slm_copy,
            intel_sub_group_block_read((const __global uint *)(dst_copy)));
    return;
}
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
__kernel void
conv_bwd_data_x8s8x8(const __global SRC_DATA_T *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst) {

    const int group_ic = get_group_id(0) * IC_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int sub_group_lid = get_sub_group_local_id();

    const int ic = (sub_group_id % IC_GROUP);
    const int sp = (sub_group_id / IC_GROUP);
    const int g = (group_ic + ic) / IC_NCHUNK;
    const int group_oc = OC_NCHUNK * g;
    const int gid = group_sp / (IW_NCHUNK * IH);
    const int gihw = group_sp % (IW_NCHUNK * IH);
    const int gih = gihw / IW_NCHUNK;
    const int giw = IW_BLOCK * (gihw % IW_NCHUNK);

    const int id = gid;
    const int iw = giw + IW_BLOCK * sp; // IW_BLOCK * (sp)-> iw_group
    const int ih = gih;

#if IS_NHWC
    dst += group_mb * MB_BLOCK * OD * OH * OW * G * OC;
    dst += OC_BLOCK * group_oc;

    src += group_mb * MB_BLOCK * ID * IH * IW * G * IC;
    src += (IW * IH * id + IW * ih + iw) * G * IC;
    src += (group_ic + ic) * IC_BLOCK;
#else
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * (group_ic + ic);
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * group_oc;
#endif
    wei += OC_BLOCK * KD * KH * KW * IC_BLOCK * (group_ic + ic) * OC_NCHUNK;

    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;

    __local uint dst_slm[DST_SLM_SIZE];
    __local uint *dst_iw_slm = dst_slm + (OC_BLOCK / 4) * sp * IW_BLOCK;

    __global DATA_T *dst_tmp = dst;
    for (int oc_chunk = 0; oc_chunk < OC_NCHUNK; oc_chunk++) {
        A_DATA_BLOCK_T D0;
        for (int kd = 0; kd < KD; kd++) {
            int od = id - kd * (1 + DD) + PD;
            if (od < 0 || od % SD != 0) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            od /= SD;
            if (od >= OD) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                int oh = (ih - kh * (1 + DH) + PH);
                if (oh < 0 || oh % SH != 0) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                oh /= SH;
                if (oh >= OH) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
#if IS_NHWC
                __global DATA_T *dst_cur
                        = dst + (G * OC * (OW * OH * od + OW * oh));
#else
                __global DATA_T *dst_cur = dst
                        + (OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh));
#endif
                barrier(CLK_LOCAL_MEM_FENCE);
#if !HAS_PAD_W && SW == 1 && KW == 1 && !IS_NHWC && IW % IW_BLOCK == 0
                // Copy block to SLM
                int ow_min = (iw - (KW - 1) * (1 + DW));
                int ow_max = (iw + IW_BLOCK - 1);
                __attribute__((opencl_unroll_hint)) for (int i = ow_min;
                                                         i <= ow_max;
                                                         i += IW_BLOCK) {
                    if (i < 0 || i >= OW) {
                        block_write(dst_iw_slm + (i - ow_min) * 8, 0);
                        continue;
                    }

                    WRITE_LOCAL(dst_iw_slm + (i - ow_min) * 8,
                            READ_BLOCK((const __global uint
                                            *)(&dst_cur[i * OC_BLOCK])));
                }
#else

                int ow_min = (iw - (KW - 1) * (1 + DW) + PW);
                int ow_max = (iw + IW_BLOCK - 1 + PW);
                __attribute__((opencl_unroll_hint)) for (int i = ow_min;
                                                         i <= ow_max; i++) {
                    if (i < 0 || i % SW != 0) {
                        block_write(dst_iw_slm + (i - ow_min) * 8, 0);
                        continue;
                    }
                    int index = i / SW;
                    if (index >= OW) {
                        block_write(dst_iw_slm + (i - ow_min) * 8, 0);
                        continue;
                    }
#if IS_NHWC
                    write_local(dst_iw_slm + (i - ow_min) * 8, oc_chunk,
                            dst_cur + (index * G * OC));
#else
                    block_write(dst_iw_slm + (i - ow_min) * 8,
                            block_read((const __global uint
                                            *)(&dst_cur[index * OC_BLOCK])));
#endif
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int kw = 0; kw < KW; kw++) {
                    unroll_for(int i = 0; i < IW_BLOCK; i++) {
                        int ow_index = (iw + i - kw * (1 + DW) + PW) - ow_min;
                        D0[i] = block_read(dst_iw_slm + (ow_index * 8));
                    }
                    int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;
#if OC % OC_BLOCK != 0
                    if (oc_chunk == OC_NCHUNK - 1) {
                        unroll_for(int i = 0; i < OC_NBLOCKS_TAIL; ++i)
                                BLOCK_READ_WHT_1x32(W0[i], (i + 0) * OC_BLOCK);
                        if (IC > 8)
                            unroll_for(int i = 0; i < OC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W1[i], (i + 8) * OC_BLOCK);
                        if (IC > 16)
                            unroll_for(int i = 0; i < OC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W2[i], (i + 16) * OC_BLOCK);
                        if (IC > 24)
                            unroll_for(int i = 0; i < OC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W3[i], (i + 24) * OC_BLOCK);

                        C00 = MMAD_TAIL(D0, W0, C00);
                        if (IC > 8) C01 = MMAD_TAIL(D0, W1, C01);
                        if (IC > 16) C02 = MMAD_TAIL(D0, W2, C02);
                        if (IC > 24) C03 = MMAD_TAIL(D0, W3, C03);
                    } else
#endif // OC % OC_BLOCK != 0
                    {
                        BLOCK_READ_WHT_8x32(W0, 0);
                        if (IC > 8) BLOCK_READ_WHT_8x32(W1, 8 * OC_BLOCK);
                        if (IC > 16) BLOCK_READ_WHT_8x32(W2, 16 * OC_BLOCK);
                        if (IC > 24) BLOCK_READ_WHT_8x32(W3, 24 * OC_BLOCK);
                        C00 = MMAD_FULL(D0, W0, C00);
                        if (IC > 8) C01 = MMAD_FULL(D0, W1, C01);
                        if (IC > 16) C02 = MMAD_FULL(D0, W2, C02);
                        if (IC > 24) C03 = MMAD_FULL(D0, W3, C03);
                    }
                    wei += IC_BLOCK * OC_BLOCK;
                } // KW
            } // KH
        } // KD
#if IS_NHWC
        dst += OC_BLOCK;
#else
        dst += OC_BLOCK * MB_BLOCK * OH * OW * OD;
#endif
    }

#define PACK(idx) \
    BIAS_SUM_RELU(D00[0], tmp[0], C00[idx], bia[0]); \
    BIAS_SUM_RELU(D00[1], tmp[1], C01[idx], bia[1]); \
    BIAS_SUM_RELU(D00[2], tmp[2], C02[idx], bia[2]); \
    BIAS_SUM_RELU(D00[3], tmp[3], C03[idx], bia[3]); \
    src_pack[idx] = as_uint(D00);

#if WITH_BIAS
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA) \
    TMP = (float)ACC + BIA; \
    RES = TO_SRC(TMP);
#else
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA) RES = TO_SRC((float)ACC);
#endif // WITH_BIAS

    uchar4 D00;
    uint8 src_pack;
    float4 bia, tmp;
    BLOCK_READ_BIA(bia, (group_ic + ic) * IC_BLOCK);
#if IS_NHWC
#if IW_TAIL
    if (iw + IW_BLOCK > IW) {
        for (int i = 0; i < IW_TAIL; ++i) {
            PACK(i);
        }
        __attribute__((opencl_unroll_hint(IW_TAIL))) for (int i = 0;
                                                          i < IW_TAIL; i++) {
            write_ic_block4(src + i * G * IC, (group_ic + ic) * IC_BLOCK,
                    as_uchar4(src_pack[i]));
        }
    } else {
#endif // IW_TAIL
#if IW_BLOCK == 4 || IW_BLOCK == 8
        for (int i = 0; i < IW_BLOCK; ++i) {
            PACK(i);
        }
        write_ic_block4(src + G * IC * 0, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[0]));
        write_ic_block4(src + G * IC * 1, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[1]));
        write_ic_block4(src + G * IC * 2, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[2]));
        write_ic_block4(src + G * IC * 3, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[3]));

#endif // IW_BLOCK == 4 || IW_BLOCK == 8
#if IW_BLOCK == 8
        write_ic_block4(src + G * IC * 4, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[4]));
        write_ic_block4(src + G * IC * 5, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[5]));
        write_ic_block4(src + G * IC * 6, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[6]));
        write_ic_block4(src + G * IC * 7, (group_ic + ic) * IC_BLOCK,
                as_uchar4(src_pack[7]));
#endif // IW_BLOCK == 8
#if IW_TAIL
    }
#endif // IW_TAIL
#else // IS_NHWC
#if IW_TAIL
    if (iw + IW_BLOCK > IW) {
        for (int i = 0; i < IW_TAIL; ++i) {
            PACK(i);
        }
        __attribute__((opencl_unroll_hint(IW_TAIL))) for (int i = 0;
                                                          i < IW_TAIL; i++) {
            intel_sub_group_block_write_uc4(
                    (__global uchar *)&src[i * IC_BLOCK],
                    as_uchar4(src_pack[i]));
        }
    } else {
#endif // IW_TAIL
#if IW_BLOCK == 4 || IW_BLOCK == 8
        for (int i = 0; i < IW_BLOCK; ++i) {
            PACK(i);
        }
        intel_sub_group_block_write_uc16((__global uchar *)&src[0 * IC_BLOCK],
                as_uchar16(src_pack.s0123));
#endif // IW_BLOCK == 4 || IW_BLOCK == 8
#if IW_BLOCK == 8
        intel_sub_group_block_write_uc16((__global uchar *)&src[4 * IC_BLOCK],
                as_uchar16(src_pack.s4567));
#endif // IW_BLOCK == 8
#if IW_TAIL
    }
#endif // IW_TAIL
#endif //IS_NHWC
}
