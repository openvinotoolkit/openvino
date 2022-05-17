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

#if IS_DW != 1
#error "Kernel supports depth-wise convolutions only"
#endif

#ifdef DST_DT_S8
#if VER_32MB16C
#define DST_MB_BLOCK MB_BLOCK
#else // VER_32MB16C
#define DST_MB_BLOCK (MB_BLOCK * 2)
#endif // VER_32MB16C
#define DST_OC_BLOCK (OC_BLOCK * 2)
#endif // DST_DT_S8

#define APPLY_POST_OPS_COMMON(nelems, accumulator, dest_data, mb_shift) \
    { \
        const int po_mb = mb_shift + mb; \
        const int po_oc = g; \
        int po_mb_count; \
        if (VER_16MB16C == 1) { \
            po_mb_count = nelems; \
        } else { \
            po_mb_count = 1; \
        } \
        APPLY_POST_OPS_TRY_BURST(accumulator, DATA_T, dest_data, DATA_T, \
                po_mb, po_mb_count, po_oc, SUB_GROUP_SIZE, \
                get_sub_group_local_id()); \
    }

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_conv_dw_fwd(const __global DATA_T *src, const __global DATA_T *wei,
        const __global DATA_T *bias, __global DST_DATA_T *dst POST_OP_ARGS) {

    MAYBE_SKIP_NON_UNIFORM_WG();

#if VER_8OW16C
    const int osp = get_global_id(1);
    const int od = osp / (OWB * OH);
    const int ohw = osp % (OWB * OH);
    const int ow = (ohw % OWB) * OW_BLOCK;
    const int oh = ohw / OWB;
    const int g
            = (get_group_id(0) * (LWS_0 / SUB_GROUP_SIZE) + get_sub_group_id())
            * OC_BLOCK;
    const int mb = get_global_id(2) * MB_BLOCK;

    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;
#ifdef DST_DT_S8 // 32c dst
    const int G_32block = G % 32 ? (32 + G - (G % 32)) : G;
    dst += mb * G_32block * OD * OH * OW
            + (g / 32 * 32) * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * (DST_OC_BLOCK)
            + (g % 32);
#else
    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK;
#endif
    src += mb
                    * ((G_WO_PADDING / IC_BLOCK)
                            + (G_WO_PADDING % IC_BLOCK > 0 ? 1 : 0))
                    * IC_BLOCK * ID * IH * IW
            + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK;
    wei += g * KD * KH * KW;

    DATA_T S00[OW_BLOCK] = {DATA_ZERO};
    if (WITH_BIAS) {
        const int bg_off = g + get_sub_group_local_id();
        DATA_T b = (G_WO_PADDING % OC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off]
                : DATA_ZERO;
        unroll_for(int k = 0; k < OW_BLOCK; k++) { S00[k] = b; }
    }

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; kd++)
        for (int kh = 0; kh < KH; kh++) {
            if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) continue;
            if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH) continue;

            const __global DATA_T *src1 = src
                    + (kd * (1 + DD) * IH + kh * (1 + DH)) * IW * MB_BLOCK
                            * IC_BLOCK;
            DATA_T tempA[SW * OW_BLOCK + KW * (1 + DW)] = {0};
            __attribute__((opencl_unroll_hint(
                    SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
            for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                if ((i + iw) >= 0 && (i + iw) < IW) {
                    tempA[i] = AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T
                                    *)(&src1[i * IC_BLOCK])));
                }
            }
            for (int kw = 0; kw < KW; kw++) {
                const __global DATA_T *wei1
                        = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
#else
    const int kw = 0;
    const __global DATA_T *wei1 = wei;
    const __global DATA_T *src1 = src;
#endif
                DATA_T B0 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)(wei1)));
                DATA_T A0;

                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int k = 0; k < OW_BLOCK; k++) {
                    if (G != G_WO_PADDING && g >= G_WO_PADDING) {
                        S00[k] = DATA_ZERO;
                        continue;
                    }
#if KH != 1 || KW != 1 || KD != 1
                    A0 = tempA[k * SW + kw * (1 + DW)];
#else
        if (iw + kw * (1 + DW) + k * SW < 0
                || iw + kw * (1 + DW) + k * SW >= IW)
            A0 = DATA_ZERO;
        else
            A0 = AS_DATA_T(BLOCK_READ(
                    (const __global BLOCK_DATA_T *)(&src1[k * SW * IC_BLOCK])));
#endif
                    S00[k] = fma(A0, (DATA_T)B0, S00[k]);
                }
#if KH != 1 || KW != 1 || KD != 1
            }
        }
#endif

    DATA_T D00[OW_BLOCK] = {0};
#if WITH_SUM
#ifdef DST_DT_S8
    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < OW_BLOCK; k++) {
        D00[k] = CONVERT_DATA_T(BLOCK_READ_DST(
                (const __global DST_DATA_T *)&dst[k * DST_OC_BLOCK]));
    }
#else
    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < OW_BLOCK; k++) {
        D00[k] = AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&dst[k * OC_BLOCK]));
    }
#endif
#endif

    APPLY_POST_OPS_COMMON(OW_BLOCK, S00, D00, 0);

    if (OW % OW_BLOCK == 0 || ow + OW_BLOCK <= OW) {
        __attribute__((opencl_unroll_hint)) // attr:no-format
        for (int k = 0; k < OW_BLOCK; k++) {
#ifdef DST_DT_S8
            BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[k * DST_OC_BLOCK],
                    CONVERT_DST_DATA_T(S00[k]));
#else
            BLOCK_WRITE((__global BLOCK_DATA_T *)&dst[k * OC_BLOCK],
                    AS_UINT_T(S00[k]));
#endif
        }
    } else {
        __attribute__((opencl_unroll_hint)) // attr:no-format
        for (int k = 0; k < OW % OW_BLOCK; k++) {
#ifdef DST_DT_S8
            BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[k * DST_OC_BLOCK],
                    CONVERT_DST_DATA_T(S00[k]));
#else
            BLOCK_WRITE((__global BLOCK_DATA_T *)&dst[k * OC_BLOCK],
                    AS_UINT_T(S00[k]));
#endif
        }
    }
#endif

#if VER_16MB16C || VER_32MB16C
    const int osp = get_global_id(1);
    const int od = osp / (OWB * OH);
    const int ohw = osp % (OWB * OH);
    const int ow = (ohw % OWB) * OW_BLOCK;
    const int oh = ohw / OWB;
    const int g
            = (get_group_id(0) * (LWS_0 / SUB_GROUP_SIZE) + get_sub_group_id())
            * OC_BLOCK;
    const int mb = get_global_id(2) * MB_BLOCK;

    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;

#ifdef DST_DT_S8 //32n32c dst
    const int G_32block = G % 32 ? (32 + G - (G % 32)) : G;
    dst += (mb / DST_MB_BLOCK) * G_32block * OD * OH * OW * DST_MB_BLOCK
            + (mb % DST_MB_BLOCK) * DST_OC_BLOCK
            + (g / DST_OC_BLOCK) * OD * OH * OW * DST_MB_BLOCK * DST_OC_BLOCK
            + (od * OH * OW + oh * OW + ow) * DST_MB_BLOCK * DST_OC_BLOCK
            + (g % DST_OC_BLOCK);
#else
    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK;
#endif
    src += mb
                    * ((G_WO_PADDING / IC_BLOCK)
                            + (G_WO_PADDING % IC_BLOCK > 0 ? 1 : 0))
                    * IC_BLOCK * ID * IH * IW
            + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK;
    wei += g * KD * KH * KW;

    DATA8_T S00 = DATA_ZERO;
    DATA8_T S01 = DATA_ZERO;
#if VER_32MB16C
    DATA8_T S02 = DATA_ZERO;
    DATA8_T S03 = DATA_ZERO;
#endif

    if (WITH_BIAS) {
        const int bg_off = g + get_sub_group_local_id();
        DATA_T b = (G_WO_PADDING % OC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off]
                : DATA_ZERO;
        unroll_for(int k = 0; k < 8; k++) {
            S00[k] = b;
            S01[k] = b;
#if VER_32MB16C
            S02[k] = b;
            S03[k] = b;
#endif
        }
    }

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; kd++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID)
                    continue;
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                    continue;
                if (iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW)
                    continue;

                const __global DATA_T *wei1
                        = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
                const __global DATA_T *src1 = src
                        + (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW
                                  + kw * (1 + DW))
                                * MB_BLOCK * IC_BLOCK;
#else
    const __global DATA_T *wei1 = wei;
    const __global DATA_T *src1 = src;
#endif
                if (G != G_WO_PADDING && g >= G_WO_PADDING) {
                    S00 = DATA_ZERO;
                    S01 = DATA_ZERO;
#if VER_32MB16C
                    S02 = DATA_ZERO;
                    S03 = DATA_ZERO;
#endif
                    continue;
                }
                DATA8_T A0 = AS_DATA8_T(
                        BLOCK_READ8((const __global BLOCK_DATA_T *)(src1)));
                DATA8_T A1 = AS_DATA8_T(BLOCK_READ8(
                        (const __global BLOCK_DATA_T *)&src1[8 * IC_BLOCK]));
#if VER_32MB16C
                DATA8_T A2 = AS_DATA8_T(BLOCK_READ8(
                        (const __global BLOCK_DATA_T *)&src1[16 * IC_BLOCK]));
                DATA8_T A3 = AS_DATA8_T(BLOCK_READ8(
                        (const __global BLOCK_DATA_T *)&src1[24 * IC_BLOCK]));
#endif
                DATA_T B0 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)(wei1)));

                S00 = fma(A0, (DATA8_T)B0, S00);
                S01 = fma(A1, (DATA8_T)B0, S01);
#if VER_32MB16C
                S02 = fma(A2, (DATA8_T)B0, S02);
                S03 = fma(A3, (DATA8_T)B0, S03);
#endif
#if KH != 1 || KW != 1 || KD != 1
            }
#endif

    DATA8_T D00;
    DATA8_T D01;
#if VER_32MB16C
    DATA8_T D02;
    DATA8_T D03;
#endif
#if WITH_SUM
#ifdef DST_DT_S8
    for (int i = 0; i < 8; ++i) {
        D00[i] = CONVERT_DATA_T(
                BLOCK_READ_DST((__global DST_DATA_T *)&dst[i * 32]));
        D01[i] = CONVERT_DATA_T(
                BLOCK_READ_DST((__global DST_DATA_T *)&dst[(i * 32) + 256]));
#if VER_32MB16C
        D02[i] = CONVERT_DATA_T(
                BLOCK_READ_DST((__global DST_DATA_T *)&dst[i * 32] + 512));
        D03[i] = CONVERT_DATA_T(
                BLOCK_READ_DST((__global DST_DATA_T *)&dst[(i * 32) + 768]));
#endif
    }
#else
    D00 = AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)dst));
    D01 = AS_DATA8_T(
            BLOCK_READ8((const __global BLOCK_DATA_T *)&dst[8 * OC_BLOCK]));
#if VER_32MB16C
    D02 = AS_DATA8_T(
            BLOCK_READ8((const __global BLOCK_DATA_T *)&dst[16 * OC_BLOCK]));
    D03 = AS_DATA8_T(
            BLOCK_READ8((const __global BLOCK_DATA_T *)&dst[24 * OC_BLOCK]));
#endif
#endif
#endif

    APPLY_POST_OPS_COMMON(8, S00, D00, 0);
    APPLY_POST_OPS_COMMON(8, S01, D01, 8);
#if VER_32MB16C
    APPLY_POST_OPS_COMMON(8, S02, D02, 16);
    APPLY_POST_OPS_COMMON(8, S03, D03, 24);
#endif

#ifdef DST_DT_S8
    for (int i = 0; i < 8; ++i) {
        BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[i * DST_OC_BLOCK],
                CONVERT_DST_DATA_T(S00[i]));
        BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[(i + 8) * DST_OC_BLOCK],
                CONVERT_DST_DATA_T(S01[i]));
#if VER_32MB16C
        BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[(i + 16) * DST_OC_BLOCK],
                CONVERT_DST_DATA_T(S02[i]));
        BLOCK_WRITE_DST((__global DST_DATA_T *)&dst[(i + 24) * DST_OC_BLOCK],
                CONVERT_DST_DATA_T(S03[i]));
#endif
    }
#else
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[0], AS_UINT8_T(S00));
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[8 * OC_BLOCK], AS_UINT8_T(S01));
#if VER_32MB16C
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[16 * OC_BLOCK], AS_UINT8_T(S02));
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[24 * OC_BLOCK], AS_UINT8_T(S03));
#endif
#endif

#endif
    return;
}
