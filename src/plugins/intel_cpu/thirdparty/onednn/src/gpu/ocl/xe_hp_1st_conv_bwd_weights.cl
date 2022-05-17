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
#include "gpu/ocl/ocl_types.h"

#define DT_UNDEF

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_types.h"

#if OD > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

#define BLOCK_READ_SRC(ptr, dst, lim, off) \
    short8 temp = cvt_f32_to_bf16(1); \
    temp[0] = ptr[SW * SUB_GROUP_SIZE + SW * sglid - PW]; \
    temp[1] = ptr[SW * SUB_GROUP_SIZE + SW * sglid + 1 - PW]; \
    slm_s[SW * sglid] = slm_s[SW * sglid + SUB_GROUP_SIZE * SW]; \
    if (SW == 2) \
        slm_s[SW * sglid + 1] = slm_s[SW * sglid + SUB_GROUP_SIZE * SW + 1]; \
    if (SW * ow + SW * SUB_GROUP_SIZE + SW * sglid >= IW + PW) { \
        temp[0] = cvt_f32_to_bf16(0); \
    } \
    if (SW == 2 \
            && SW * ow + SW * SUB_GROUP_SIZE + SW * sglid + 1 >= IW + PW) { \
        temp[1] = cvt_f32_to_bf16(0); \
    } \
    slm_s[SW * SUB_GROUP_SIZE + SW * sglid] = temp[0]; \
    slm_s[SW * SUB_GROUP_SIZE + SW * sglid + 1] = temp[1]; \
    temp[0] = slm_s[SW * sglid]; \
    temp[1] = slm_s[SW * sglid + 1]; \
    temp[2] = slm_s[SW * sglid + SW * SUB_GROUP_SIZE]; \
    temp[3] = slm_s[SW * sglid + SW * SUB_GROUP_SIZE + 1]; \
    for (int ctr = 0; ctr < lim / 2; ctr++) { \
\
        if (SW == 2) { \
            dst[off + 4 * ctr] \
                    = intel_sub_group_shuffle_down(temp[0], temp[2], ctr); \
            dst[off + 4 * ctr + 2] \
                    = intel_sub_group_shuffle_down(temp[1], temp[3], ctr); \
        } else if (SW == 1) { \
            dst[off + 4 * ctr] \
                    = intel_sub_group_shuffle_down(temp[0], temp[2], 2 * ctr); \
            dst[off + 4 * ctr + 2] = intel_sub_group_shuffle_down( \
                    temp[0], temp[2], 2 * ctr + 1); \
        } \
    }

#define BLOCK_READ_DST8(ptr, dst, off, lim) \
    dst.s01234567 = as_short8(intel_sub_group_block_read_us8(ptr + off));

#define BLOCK_READ_SLM8(ptr, dst, off) \
    if (ow + OW_BLOCK <= OW) { \
        dst.s02468ace = as_short8(intel_sub_group_block_read_us8(ptr)); \
    } else { \
        for (int i = 0; i < OW % OW_BLOCK; i++) \
            dst[2 * i] = as_short( \
                    intel_sub_group_block_read_us(ptr + SUB_GROUP_SIZE * i)); \
    }

#if BWD_WEIGHTS == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
xe_hp_1st_conv_bwd_weights(__global SRC_DATA_T *src,
        volatile __global float *diff_wei, volatile __global float *diff_bias,
        __global DST_DATA_T *diff_dst) {

    MAYBE_SKIP_NON_UNIFORM_WG();

#if VER_8OW16C == 1
#define HAS_PAD_W (PW > 0 || OW * SW - PW + (KW - 1) * (1 + DW) >= IW)
    const int sglid = get_sub_group_local_id();
    const int glid_1 = get_global_id(1);
    const int lid_0 = get_local_id(0);
    const int glid_2 = get_global_id(2);

    const int ksp = glid_1 * LWS_0 + SUB_GROUP_SIZE * (lid_0 / SUB_GROUP_SIZE);

#if CASE_3D
    const int kd = ksp / (KWB * KH * IC);
    const int khw = ksp % (KWB * KH * IC);
#else
    const int khw = ksp;
    const int kd = 0;
#endif

    const int kh = khw / (KWB * IC);
    const int kw = (khw % (KWB * IC)) % KWB;

    const int chunk = glid_2 % NCHUNK;
    const int icb_ocb = glid_2 / NCHUNK;
    const int icb = icb_ocb % (IC / ICB);
    const int ocb = icb_ocb / (IC / ICB);

    const int g_ic_oc = get_global_id(0);
    const int g = g_ic_oc / (OC * (IC / IC_BLOCK));
    const int io = g_ic_oc % (OC * (IC / IC_BLOCK));
    const int oc = (io % OCB) / (OC_BLOCK) + ocb * (OCB / OC_BLOCK);
    const int ic = (khw % (KWB * IC)) / KWB;

    const int sp_chunk = chunk % OSP_CHUNK;
    const int mb_chunk = chunk / OSP_CHUNK;

    const int ow_nb = (OW + OWB - 1) / OWB;
    const int oh_nb = (OH + OHB - 1) / OHB;

    const int od_beg = ((sp_chunk / ow_nb) / oh_nb) * ODB;
    const int oh_beg = ((sp_chunk / ow_nb) % oh_nb) * OHB;
    const int ow_beg = (sp_chunk % ow_nb) * OWB;

    const int mb = mb_chunk * MB_CHUNK_SIZE;
    const int mb_end = min((mb_chunk + 1) * MB_CHUNK_SIZE, MB);

    const bool do_bias = glid_1 == 0;

    const int OW_LOOP_BLOCK = 8;

    src += mb * IC * G * ID * IH * IW + g * IC * ID * IH * IW * MB_BLOCK;

    const int lchan = ((lid_0 / SUB_GROUP_SIZE) % 2);
    diff_dst += oc * OD * OH * OW * OC_BLOCK * MB_BLOCK
            + g * OC * OD * OH * OW * MB_BLOCK;

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + sglid;
    float bias_loc = 0.0f;
    float bias_loc2 = 0.0f;
#endif
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
    float8 blockC10 = 0.0f;
    float8 blockC11 = 0.0f;

    __local ushort slm_oc[16 * OC_BLOCK];
    const int padded_iw = SW * 2 * OW_BLOCK
            + 24; // 24 is zero filled tail should have minimum size of SW*8
    //slm src size = row size * 4 subgroups per work group
    __local ushort slm_src[4 * padded_iw];
    __local ushort *slm_s = &slm_src[(lid_0 / SUB_GROUP_SIZE) * padded_iw];

    for (int omb = mb; omb < mb_end; omb++) {
        const __global DST_DATA_T *diff_dst1_
                = diff_dst + omb * OC * G * OD * OH * OW;

        for (int od = od_beg; od < min(od_beg + ODB, OD); od++)
            for (int oh = oh_beg; oh < min(oh_beg + OHB, OH); oh++) {
                const __global DST_DATA_T *diff_dst1 = diff_dst1_
                        + od * OH * OW * OC_BLOCK + oh * OW * OC_BLOCK;
                bool skip = false;

                if (oh * SH + kh * (1 + DH) < PH
                        || oh * SH + kh * (1 + DH) >= IH + PH
#if CASE_3D
                        || od * SD + kd * (1 + DD) < PD
                        || od * SD + kd * (1 + DD) >= ID + PD
#endif
                ) {
                    skip = true;
                }

                int id = od * SD - PD + kd * (1 + DD);
                int ih = oh * SH - PH + kh * (1 + DH);
                __global SRC_DATA_T *src1 = src + (id)*IH * IW * IC_BLOCK
                        + (ih)*IW * IC_BLOCK + ic * ID * IH * IW * MB_BLOCK
                        + ow_beg * SW;

                intel_sub_group_block_write_us(
                        slm_s + SW * OW_BLOCK, cvt_f32_to_bf16(0));
                intel_sub_group_block_write_us(
                        slm_s + SW * OW_BLOCK + SUB_GROUP_SIZE,
                        cvt_f32_to_bf16(0));
                intel_sub_group_block_write_us(
                        slm_s + SW * OW_BLOCK + 2 * SUB_GROUP_SIZE,
                        cvt_f32_to_bf16(0));
                if (ow_beg > 0 && sglid < PW) {
                    slm_s[SUB_GROUP_SIZE * SW + sglid] = src1[sglid - PW];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
#if IW <= SUB_GROUP_SIZE * SW
                if (ow_beg * SW + SW * sglid < IW)
#endif
                    slm_s[PW + SUB_GROUP_SIZE * SW + SW * sglid]
                            = src1[SW * sglid];
                if (SW == 2) {
#if IW <= SUB_GROUP_SIZE * SW
                    if (ow_beg * SW + SW * sglid + 1 < IW)
#endif
                        slm_s[PW + SUB_GROUP_SIZE * SW + SW * sglid + 1]
                                = src1[SW * sglid + 1];
                }

                for (int ow = ow_beg; ow < min(ow_beg + OWB, OW);
                        ow += OW_BLOCK) {

                    int iw = ow * SW;
                    short16 blockA = cvt_f32_to_bf16(0);
                    short16 blockB = cvt_f32_to_bf16(0);
                    src1 = src + (id)*IH * IW * IC_BLOCK + (ih)*IW * IC_BLOCK
                            + (iw)*IC_BLOCK + ic * ID * IH * IW * MB_BLOCK;

                    if (!skip) {
                        BLOCK_READ_SRC(src1, blockA, (KW + KW % 2), 0)
                    }

                    const __global DST_DATA_T *diff_dst2_ = diff_dst1
                            + (ow + 0) * OC_BLOCK
                            + lchan * OC_BLOCK * (OW_BLOCK / 2);
                    BLOCK_READ_DST8(diff_dst2_, blockB, 0, 0);
#if WITH_BIAS == 1
                    for (int i = 0; i < OW_LOOP_BLOCK / 2; i++) {
#if OW % OW_BLOCK != 0
                        if (ow + lchan * (OW_LOOP_BLOCK / 2) + i < OW) {
#endif
                            bias_loc += cvt_bf16_to_f32(blockB[2 * i]);
                            bias_loc2 += cvt_bf16_to_f32(blockB[2 * i + 1]);
#if OW % OW_BLOCK != 0
                        }
#endif
                    }
#endif
                    intel_sub_group_block_write_us4(slm_oc
                                    + (oc % 2) * OW_BLOCK * 2 * (OC_BLOCK / 2)
                                    + lchan * (OW_BLOCK / 2) * (OC_BLOCK / 2),
                            as_ushort4(blockB.s0246));
                    intel_sub_group_block_write_us4(slm_oc
                                    + (oc % 2) * OW_BLOCK * 2 * (OC_BLOCK / 2)
                                    + OW_BLOCK * (OC_BLOCK / 2)
                                    + lchan * (OW_BLOCK / 2) * (OC_BLOCK / 2),
                            as_ushort4(blockB.s1357));

                    barrier(CLK_LOCAL_MEM_FENCE);
                    blockB = cvt_f32_to_bf16(0);

                    BLOCK_READ_SLM8(slm_oc, blockB, 0);
                    blockC00 = mmad8x8_bf16(
                            as_uint8(blockA), as_int8(blockB), blockC00);
                    BLOCK_READ_SLM8(
                            slm_oc + OW_BLOCK * (OC_BLOCK / 2), blockB, 0);
                    blockC01 = mmad8x8_bf16(
                            as_uint8(blockA), as_int8(blockB), blockC01);
                    BLOCK_READ_SLM8(
                            slm_oc + 2 * OW_BLOCK * (OC_BLOCK / 2), blockB, 0);
                    blockC10 = mmad8x8_bf16(
                            as_uint8(blockA), as_int8(blockB), blockC10);
                    BLOCK_READ_SLM8(
                            slm_oc + 3 * OW_BLOCK * (OC_BLOCK / 2), blockB, 0);
                    blockC11 = mmad8x8_bf16(
                            as_uint8(blockA), as_int8(blockB), blockC11);
                }
            }
        src += G * IC * ID * IH * IW * MB_BLOCK;
    }

#if WITH_BIAS == 1
    if (do_bias && oc * OC_BLOCK + sglid < (OC_WO_PADDING)) {
        atomic_add_global(diff_bias, bias_loc);
        atomic_add_global(diff_bias + 8, bias_loc2);
    }
#endif

    diff_wei += oc / 2 * KD * KH * KW * IC * 2 * OC_BLOCK
            + g * OC * IC * KD * KH * KW;
    if (ksp >= KH * KWB * KD * IC) return;

    for (int i = 0; i < KW; i++) {

        atomic_add_global(diff_wei + kd * KH * KW * IC * OC_BLOCK
                        + kh * KW * IC * OC_BLOCK + i * IC * OC_BLOCK
                        + ic * OC_BLOCK + sglid,
                blockC00[i]);
        atomic_add_global(diff_wei + kd * KH * KW * IC * OC_BLOCK
                        + kh * KW * IC * OC_BLOCK + i * IC * OC_BLOCK
                        + ic * OC_BLOCK + 8 + sglid,
                blockC01[i]);
        atomic_add_global(diff_wei + KD * KH * KW * IC * OC_BLOCK
                        + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
                        + i * IC * OC_BLOCK + ic * OC_BLOCK + sglid,
                blockC10[i]);
        atomic_add_global(diff_wei + KD * KH * KW * IC * OC_BLOCK
                        + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
                        + i * IC * OC_BLOCK + ic * OC_BLOCK + 8 + sglid,
                blockC11[i]);
    }

#endif
}
#endif
