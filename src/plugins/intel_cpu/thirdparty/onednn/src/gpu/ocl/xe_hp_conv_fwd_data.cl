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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define AS_SRC_DATA_BLOCK_T AS_MMAD_DATA8_T

#if SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WEI(data, idx) \
    data = as_int8(block_read8((__local uint *)&wei_tmp[idx]));
#else
#define WEI wei
#define BLOCK_READ_WEI(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = AS_SRC_DATA_BLOCK_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));

#if BIA_DT_F32
#define BLOCK_READ_BIA(data, idx) \
    if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK \
            || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK <= OC_PADDED) { \
        data = as_float4( \
                intel_sub_group_block_read4((__global uint *)&bias[idx])); \
    } else { \
        float2 tmp = as_float2( \
                intel_sub_group_block_read2((__global uint *)&bias[idx])); \
        data = (float4)(tmp.s0, tmp.s1, 0.0f, 0.0f); \
    }
#elif BIA_DT_F16
#define BLOCK_READ_BIA(data, idx) \
    if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK \
            || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK <= OC_PADDED) { \
        data = convert_float4(as_half4(intel_sub_group_block_read_us4( \
                (__global ushort *)&bias[idx]))); \
    } else { \
        float2 tmp = convert_float2(as_half2(intel_sub_group_block_read_us2( \
                (__global ushort *)&bias[idx]))); \
        data = (float4)(tmp.s0, tmp.s1, 0.0f, 0.0f); \
    }
#elif BIA_DT_BF16
#define BLOCK_READ_BIA(data, idx) \
    if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK \
            || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK <= OC_PADDED) { \
        data = cvt_bf16_to_f32(intel_sub_group_block_read_us4( \
                (__global ushort *)&bias[idx])); \
    } else { \
        float2 tmp = cvt_bf16_to_f32(intel_sub_group_block_read_us2( \
                (__global ushort *)&bias[idx])); \
        data = (float4)(tmp.s0, tmp.s1, 0.0f, 0.0f); \
    }
#endif

#if DT_F16
#define MMAD8X8 mmad8x8_f16
#elif DT_BF16
#define MMAD8X8 mmad8x8_bf16
#else
#define MMAD8X8 mmad8x8
#endif

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#if SCALES_PER_OC
#define SCALE scales
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
xe_hp_conv_fwd(const __global SRC_DATA_T *src, const __global WEI_DATA_T *wei,
        const __global BIA_DATA_T *bias, __global DST_DATA_T *dst POST_OP_ARGS,
        float scale, const __global float *scales_per_oc) {
    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int ocl_local_id = get_local_id(0);
    const int subg_local_id = get_sub_group_local_id();

    const int sub_group_id = get_sub_group_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);

    const int g = (group_oc + oc) * (OC_CALC_BLOCK / OC_BLOCK) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;

    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = gohw % OW_PADDED;

    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;

    const int local_oh = sp / OW_PADDED;
    const int local_ow = sp % OW_PADDED;
    const int local_ih = local_oh * SH;
    const int local_iw = local_ow * SW;

    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh + local_oh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih + local_ih - PH;

#if !SLM_WEI
    if (ow >= OW) return;
#endif // SLM_WEI

    dst += OC_CALC_BLOCK * MB_BLOCK * OD * OH * OW * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    wei += WEI_BLOCK * KD * KH * KW * (group_oc + oc) * IC_NCHUNK;

    MMAD_ACC_DATA8_T C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    MMAD_ACC_DATA8_T C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    MMAD_ACC_DATA8_T C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    MMAD_ACC_DATA8_T C30 = 0, C31 = 0, C32 = 0, C33 = 0;

#if SLM_WEI
    __local WEI_DATA_T wei_loc[KW * OC_GROUP * WEI_BLOCK];
    __local WEI_DATA_T *wei_loc_base = wei_loc + KW * WEI_BLOCK * oc;
#endif // SLM_WEI

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {
        __attribute__((opencl_unroll_hint(1))) // attr:no-format
        for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += WEI_BLOCK * KH * KW;
                continue;
            }
            __attribute__((opencl_unroll_hint)) // attr:no-format
            for (int kh = 0; kh < KH; kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += WEI_BLOCK * KW;
                    continue;
                }

#if SLM_WEI
                barrier(CLK_LOCAL_MEM_FENCE);
                const __global WEI_DATA_T *wei_copy_from
                        = wei + sp * KW * WEI_BLOCK / LWS_1;
                __local WEI_DATA_T *wei_copy_to
                        = wei_loc_base + sp * KW * WEI_BLOCK / LWS_1;
                for (int bl = 0; bl < KW; bl++) {
                    block_write4(
                            (__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK],
                            intel_sub_group_block_read4(
                                    (__global uint *)&wei_copy_from[bl * 4
                                            * IC_BLOCK]));
                }
                __local WEI_DATA_T *wei_tmp = wei_loc_base;
                barrier(CLK_LOCAL_MEM_FENCE);
#endif // SLM_WEI

                __attribute__((opencl_unroll_hint)) // attr:no-format
                for (int kw = 0; kw < KW; kw++) {
                    SRC_DATA_BLOCK_T S0 = 0, S1 = 0, S2 = 0, S3 = 0;
                    int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;
                    if (kw * (1 + DW) + iw >= 0 && kw * (1 + DW) + iw < IW) {
                        BLOCK_READ_SRC(S0, 0);
#if MB > 8
                        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
#if MB > 16
                        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
#if MB > 24
                        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8

                        BLOCK_READ_WEI(W0, 0);
                        BLOCK_READ_WEI(W1, 8 * IC_BLOCK);
                        BLOCK_READ_WEI(W2, 16 * IC_BLOCK);
                        BLOCK_READ_WEI(W3, 24 * IC_BLOCK);
                    }
                    C00 = MMAD8X8(S0, W0, C00);
                    C01 = MMAD8X8(S0, W1, C01);
                    C02 = MMAD8X8(S0, W2, C02);
                    C03 = MMAD8X8(S0, W3, C03);
#if MB > 8
                    C10 = MMAD8X8(S1, W0, C10);
                    C11 = MMAD8X8(S1, W1, C11);
                    C12 = MMAD8X8(S1, W2, C12);
                    C13 = MMAD8X8(S1, W3, C13);
#if MB > 16
                    C20 = MMAD8X8(S2, W0, C20);
                    C21 = MMAD8X8(S2, W1, C21);
                    C22 = MMAD8X8(S2, W2, C22);
                    C23 = MMAD8X8(S2, W3, C23);
#if MB > 24
                    C30 = MMAD8X8(S3, W0, C30);
                    C31 = MMAD8X8(S3, W1, C31);
                    C32 = MMAD8X8(S3, W2, C32);
                    C33 = MMAD8X8(S3, W3, C33);
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    WEI += WEI_BLOCK;
                }
#if SLM_WEI
                wei += WEI_BLOCK * KW;
#endif // SLM_WEI
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, (group_oc + oc) * OC_CALC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_oc + oc) * OC_CALC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#define APPLY_POST_OPS_COMMON(accumulator, dest, dest_dt, mb_shift) \
    const int po_mb = (group_mb * MB_BLOCK + mb_shift * 8 + n_i) % MB; \
    const int po_oc = (group_oc * OC_CALC_BLOCK) % (OC * G); \
    APPLY_POST_OPS_TRY_BURST(accumulator, float, dest, dest_dt, po_mb, 1, \
            po_oc, 4 * SUB_GROUP_SIZE, subg_local_id);

#if DT_F16 || DT_BF16
#define DO_POST_OP(mb_shift, d_pack0, d_pack1) \
    do { \
        float4 tmpsum_val; \
        tmpsum_val.s01 = DST_TO_REF2(AS_DST_DATA2_T(d_pack0)); \
        tmpsum_val.s23 = DST_TO_REF2(AS_DST_DATA2_T(d_pack1)); \
        APPLY_POST_OPS_COMMON(tmp, tmpsum_val, float, mb_shift); \
    } while (0)
#else // DT_F16 || DT_BF16
#define DO_POST_OP(mb_shift, d_pack0, d_pack1) \
    do { \
        SUM_DATA4_T d = AS_SUM_DATA4_T(d_pack0); \
        APPLY_POST_OPS_COMMON(tmp, d, SUM_DATA_T, mb_shift); \
    } while (0)
#endif // DT_F16 || DT_BF16

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#if DT_F16 || DT_BF16
#define CONVERT_PACK(idx) \
    do { \
        DST_DATA2_T tmp_cvt0 \
                = (DST_DATA2_T)(REF_TO_DST(tmp.s0), REF_TO_DST(tmp.s1)); \
        dst_pack0[idx] = AS_DST_PACK(tmp_cvt0); \
        DST_DATA2_T tmp_cvt1 \
                = (DST_DATA2_T)(REF_TO_DST(tmp.s2), REF_TO_DST(tmp.s3)); \
        dst_pack1[idx] = AS_DST_PACK(tmp_cvt1); \
    } while (0)

#define WRITE_DST() \
    do { \
        BLOCK_WRITE_DST(dst, dst_pack0.s0123, 0) \
        BLOCK_WRITE_DST(dst, dst_pack0.s4567, 4 * OC_BLOCK) \
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK \
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK \
                        <= OC_PADDED) { \
            __global DST_DATA_T *dst1 \
                    = dst + OC_BLOCK * MB_BLOCK * OD * OH * OW; \
            BLOCK_WRITE_DST(dst1, dst_pack1.s0123, 0) \
            BLOCK_WRITE_DST(dst1, dst_pack1.s4567, 4 * OC_BLOCK) \
        } \
    } while (0)
#else
#define CONVERT_PACK(idx) \
    do { \
        DST_DATA4_T tmp_cvt = (DST_DATA4_T)(TO_DST(tmp.s0), TO_DST(tmp.s1), \
                TO_DST(tmp.s2), TO_DST(tmp.s3)); \
        dst_pack0[idx] = as_uint(tmp_cvt); \
    } while (0)

#define WRITE_DST() \
    do { \
        BLOCK_WRITE_DST(dst, dst_pack0.s0123, 0) \
        BLOCK_WRITE_DST(dst, dst_pack0.s4567, 4 * OC_BLOCK) \
    } while (0)
#endif

#define STORE_DST(mb_shift, C0, C1, C2, C3, D0, D1) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            const int po_mb = (group_mb * MB_BLOCK + mb_shift * 8 + n_i); \
            if (MB % MB_BLOCK == 0 || po_mb < MB) { \
                QUANTIZE_ADD_BIAS(); \
                DO_POST_OP(mb_shift, D0[n_i], D1[n_i]); \
            } \
            CONVERT_PACK(n_i); \
        } \
        WRITE_DST(); \
    } while (0)

    if (ow < OW) {
        float4 tmp;

#if DST_DT_F32
        ulong8 dst_pack0, dst_pack1;
        ulong8 D00, D01, D02, D03;
        ulong8 D10, D11, D12, D13;
#define AS_DST_PACK as_ulong
#define BLOCK_WRITE_DST(p, data, idx) \
    intel_sub_group_block_write_ui8((__global uint *)&p[idx], as_uint8(data));
#define BLOCK_READ_DST(idx) \
    as_ulong4(intel_sub_group_block_read_ui8((__global uint *)&dst[idx]));
#elif DST_DT_BF16 || DST_DT_F16
        uint8 dst_pack0, dst_pack1;
        uint8 D00, D01, D02, D03;
        uint8 D10, D11, D12, D13;
#define AS_DST_PACK as_uint
#define BLOCK_WRITE_DST(p, data, idx) \
    intel_sub_group_block_write_us8( \
            (__global ushort *)&p[idx], as_ushort8(data));
#define BLOCK_READ_DST(idx) \
    as_uint4(intel_sub_group_block_read_us8((__global ushort *)&dst[idx]));
#else
        uint8 dst_pack0;
        uint8 D00, D01, D02, D03;
#define BLOCK_WRITE_DST(p, data, idx) \
    intel_sub_group_block_write_uc16( \
            (__global uchar *)&p[idx], as_uchar16(data));
#define BLOCK_READ_DST(idx) \
    as_uint4(intel_sub_group_block_read_uc16((__global uchar *)&dst[idx]));
#endif

#if WITH_SUM
        D00.s0123 = BLOCK_READ_DST(0);
        D00.s4567 = BLOCK_READ_DST(4 * OC_BLOCK);
#if DT_F16 || DT_BF16
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D10.s0123 = BLOCK_READ_DST(OC_BLOCK * MB_BLOCK * OD * OH * OW);
            D10.s4567 = BLOCK_READ_DST(
                    4 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
        }
#endif
#if MB > 8
        D01.s0123 = BLOCK_READ_DST(8 * OC_BLOCK);
        D01.s4567 = BLOCK_READ_DST(12 * OC_BLOCK);
#if DT_F16 || DT_BF16
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D11.s0123 = BLOCK_READ_DST(
                    8 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
            D11.s4567 = BLOCK_READ_DST(
                    12 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
        }
#endif
#if MB > 16
        D02.s0123 = BLOCK_READ_DST(16 * OC_BLOCK);
        D02.s4567 = BLOCK_READ_DST(20 * OC_BLOCK);
#if DT_F16 || DT_BF16
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D12.s0123 = BLOCK_READ_DST(
                    16 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
            D12.s4567 = BLOCK_READ_DST(
                    20 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
        }
#endif
#if MB > 24
        D03.s0123 = BLOCK_READ_DST(24 * OC_BLOCK);
        D03.s4567 = BLOCK_READ_DST(28 * OC_BLOCK);
#if DT_F16 || DT_BF16
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D13.s0123 = BLOCK_READ_DST(
                    24 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
            D13.s4567 = BLOCK_READ_DST(
                    28 * OC_BLOCK + OC_BLOCK * MB_BLOCK * OD * OH * OW);
        }
#endif
#endif // MB > 24
#endif // MB > 16
#endif // MB > 8
#endif

        STORE_DST(0, C00, C01, C02, C03, D00, D10);
        dst += 8 * OC_BLOCK;
        STORE_DST(1, C10, C11, C12, C13, D01, D11);
        dst += 8 * OC_BLOCK;
        STORE_DST(2, C20, C21, C22, C23, D02, D12);
        dst += 8 * OC_BLOCK;
        STORE_DST(3, C30, C31, C32, C33, D03, D13);
    }
}
