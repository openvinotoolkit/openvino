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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

// SLM buffers pipeline when the both SRC and WEI are in SLM
#if SLM_WEI
#define USE_SLM_PIPE 1
#define NUM_SLM_BUFF 2
#else
#define USE_SLM_PIPE 0
#define NUM_SLM_BUFF 1
#endif

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL \
    ((IC - (IC & ~(IC_BLOCK - 1)) + (IC_BLOCK / 8) - 1) / (IC_BLOCK / 8))
#else
#define IC_NBLOCKS_TAIL 8
#endif

#if SRC_DT_F16
#define MMAD8X8 mmad8x8_f16
#define MMAD8X4 mmad8x4_f16
#define MMAD8X2 mmad8x2_f16
#elif SRC_DT_BF16
#define MMAD8X8 mmad8x8_bf16
#define MMAD8X4 mmad8x4_bf16
#define MMAD8X2 mmad8x2_bf16
#else
#define MMAD8X8 mmad8x8
#define MMAD8X4 mmad8x4
#define MMAD8X4 mmad8x2
#endif

#if OW_BLOCK == 2

#define BLOCK 2
#define ACC_DATA_BLOCK SRC_MMAD_ACC_DATA2_T
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA2_T
#define READ_BLOCK intel_sub_group_block_read2
#define WRITE_LOCAL block_write2
#define MMAD MMAD8X2

#elif OW_BLOCK == 4

#define BLOCK 4
#define ACC_DATA_BLOCK SRC_MMAD_ACC_DATA4_T
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA4_T
#define READ_BLOCK intel_sub_group_block_read4
#define WRITE_LOCAL block_write4
#define MMAD MMAD8X4

#elif OW_BLOCK == 8

#define BLOCK 8
#define ACC_DATA_BLOCK SRC_MMAD_ACC_DATA8_T
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA8_T
#define READ_BLOCK intel_sub_group_block_read8
#define WRITE_LOCAL block_write8
#define MMAD MMAD8X8

#else
#error "Not expected"
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#if SLM_WEI
#define WEI wei_tmp
#define BLOCK_READ_WEI(data, idx) \
    data = as_int8(block_read8((__local uint *)&wei_tmp[idx]));
#else
#define WEI wei
#define BLOCK_READ_WEI(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#endif

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

#define READ_FROM_GLOBAL 0
#define WRITE_TO_LOCAL 1
#define READ_WRITE 2

inline void copy_src_to_slm(const __global SRC_DATA_T *src,
        __local uint *S_part, const int iw, const int ow,
        const bool left_nozero_tail, const bool right_nozero_tail,
        const bool right_tail, uint *Sreg_0, uint *Sreg_1, uint *Sreg_2,
        uint8 *Sreg_block, const int mode) {
#if SLM_WORKING_GROUPS < OW_NCHUNK
    if (iw + PW < IW) {
#endif
#if OW_NCHUNK > LWS_1
        /* Copy tails in case of multigroups */
        if (ow < OW) {
#if PW > 0
            if (left_nozero_tail) {
                for (int i = -PW - min(iw, 0); i < 0; i++) {
                    if (mode == READ_FROM_GLOBAL) {
                        Sreg_0[i + PW] = intel_sub_group_block_read(
                                (const __global uint *)(&src[i * IC_BLOCK]));
                    } else if (mode == WRITE_TO_LOCAL) {
                        block_write(S_part + i * 8, Sreg_0[i + PW]);
                    } else {
                        block_write(S_part + i * 8,
                                intel_sub_group_block_read((const __global uint
                                                *)(&src[i * IC_BLOCK])));
                    }
                }
            }
#endif
            if (right_nozero_tail) {
                int buffer_last = (KW - 1) * (1 + DW) - PW;
                int src_last = IW - iw - SW * OW_BLOCK - PW;
                for (int i = SW * OW_BLOCK;
                        i < SW * OW_BLOCK + min(buffer_last, src_last); i++) {
                    if (mode == READ_FROM_GLOBAL) {
                        Sreg_1[i] = intel_sub_group_block_read(
                                (const __global uint *)(&src[i * IC_BLOCK]));
                    } else if (mode == WRITE_TO_LOCAL) {
                        block_write(S_part + i * 8, Sreg_1[i]);
                    } else {
                        block_write(S_part + i * 8,
                                intel_sub_group_block_read((const __global uint
                                                *)(&src[i * IC_BLOCK])));
                    }
                }
                for (int i = SW * OW_BLOCK + min(buffer_last, src_last);
                        i < SW * OW_BLOCK + buffer_last; i++) {
                    if (mode == WRITE_TO_LOCAL || mode == READ_WRITE) {
                        block_write(S_part + i * 8, 0);
                    }
                }
            }
#endif
#if OW_SLM_TAIL != OW_BLOCK * SW
            /* Copy last block to SLM */
            if (right_tail) {
                for (int i = 0; i < OW_SLM_TAIL; i++) {
                    if (mode == READ_FROM_GLOBAL) {
                        Sreg_2[i] = intel_sub_group_block_read(
                                (const __global uint *)(&src[i * IC_BLOCK]));
                    } else if (mode == WRITE_TO_LOCAL) {
                        block_write(S_part + i * 8, Sreg_2[i]);
                    } else {
                        block_write(S_part + i * 8,
                                intel_sub_group_block_read((const __global uint
                                                *)(&src[i * IC_BLOCK])));
                    }
                }
            } else {
#endif
                /* Copy block to SLM */
                unroll_for(int i = 0; i < SW * OW_BLOCK; i += OW_BLOCK) {
                    if (mode == READ_FROM_GLOBAL) {
                        Sreg_block[i] = READ_BLOCK(
                                (const __global uint *)(&src[i * IC_BLOCK]));
                    } else if (mode == WRITE_TO_LOCAL) {
                        WRITE_LOCAL(S_part + i * 8, Sreg_block[i]);
                    } else {
                        WRITE_LOCAL(S_part + i * 8,
                                READ_BLOCK((const __global uint
                                                *)(&src[i * IC_BLOCK])));
                    }
                }
#if OW_SLM_TAIL != OW_BLOCK * SW
            }
#endif
#if OW_NCHUNK > LWS_1
        }
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
    }
#endif
}

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
xe_hp_conv_fwd_ow_block(const __global SRC_DATA_T *src,
        const __global WEI_DATA_T *wei, const __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc) {
    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int subg_local_id = get_sub_group_local_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);
    const int g = (group_oc + oc) * (OC_CALC_BLOCK / OC_BLOCK) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;
    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = OW_BLOCK * (gohw % OW_PADDED);
    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;
    const int local_ow = OW_BLOCK * sp;
    const int local_iw = local_ow * SW;
    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih - PH;

    __local uint S_slice[SRC_SLM_SIZE * NUM_SLM_BUFF];
    __local uint *S_part = S_slice + 32 / 4 * (sp * SW * OW_BLOCK + PW);
    __local uint *S_work = S_slice + 32 / 4 * (sp * SW * OW_BLOCK);

    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw > -PW;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW) && (iw + PW < IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);
    const bool empty = (iw + PW >= IW);

    dst += OC_CALC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);
    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw + PW);
    wei += WEI_BLOCK * KD * KH * KW * (group_oc + oc) * IC_NCHUNK;
#if OC_BLOCK == 16
    __global DST_DATA_T *dst1 = dst + OW * OH * OD * OC_BLOCK;
#endif

    /* Prepare S_slice tails */
    for (int idx = 0; idx < NUM_SLM_BUFF; idx++) {
#if PW > 0
        if (left_tail) {
            for (int i = 0; i < PW; i++) {
                block_write(S_slice + idx * SRC_SLM_SIZE + i * 8, 0);
            }
        }
#endif
#if ZERO_TAIL > 0
        if (right_tail) {
            for (int i = OW_SLM_TAIL;
                    i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW; i++) {
                block_write(S_part + idx * SRC_SLM_SIZE + i * 8, 0);
            }
        }
#if SLM_WORKING_GROUPS < OW_NCHUNK
        if (empty) {
            for (int i = -PW; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                    i++) {
                block_write(S_part + idx * SRC_SLM_SIZE + i * 8, 0);
            }
        }
#endif
#endif
    }

    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    SRC_DATA_BLOCK_T S0;

#if SLM_WEI
#define WEI_SLM_BUFF_SIZE (KW * OC_GROUP * WEI_BLOCK) // size of single buffer
    __local WEI_DATA_T wei_loc[WEI_SLM_BUFF_SIZE * NUM_SLM_BUFF];
    __local WEI_DATA_T *wei_loc_base = wei_loc + KW * WEI_BLOCK * oc;
#endif

    int idx_read = 0;
    int idx_use = 0;
    bool is_first_mul = true;

#define COMBINED_LOOP (IC_NCHUNK * KD * KH)

    for (int loop_idx = 0; loop_idx < COMBINED_LOOP + NUM_SLM_BUFF - 1;
            loop_idx++) {

        int kh = loop_idx % KH;
        int ii = loop_idx / KH;
        int kd = ii % KD;
        int ic_chunk = ii / KD;

        bool last_iters = loop_idx >= COMBINED_LOOP;
        bool skip_kd = kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID;
        bool skip_kh = kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH;
        bool use_zero = skip_kh || skip_kd || last_iters;

#define MINSIZE(x) (x) < 1 ? 1 : (x)
        uint Sreg_0[MINSIZE(PW)] = {0};
        uint Sreg_1[MINSIZE((KW - 1) * (1 + DW) - PW)] = {0};
        uint Sreg_2[MINSIZE(OW_SLM_TAIL)] = {0};
        uint8 Sreg_block[SW] = {0};

#if !USE_SLM_PIPE
        barrier(CLK_LOCAL_MEM_FENCE);
        copy_src_to_slm(src, S_part, iw, ow, left_nozero_tail,
                right_nozero_tail, right_tail, Sreg_0, Sreg_1, Sreg_2,
                Sreg_block, use_zero ? WRITE_TO_LOCAL : READ_WRITE);
#endif

#if SLM_WEI
        const __global WEI_DATA_T *wei_copy_from
                = wei + sp * KW * WEI_BLOCK / LWS_1;
        __local WEI_DATA_T *wei_copy_to = wei_loc_base
                + sp * KW * WEI_BLOCK / LWS_1 + WEI_SLM_BUFF_SIZE * idx_read;

        uint4 Wreg[KW];
        __local uint *S_part_next = S_part + SRC_SLM_SIZE * idx_read;

        if (!use_zero) {
            copy_src_to_slm(src, S_part_next, iw, ow, left_nozero_tail,
                    right_nozero_tail, right_tail, Sreg_0, Sreg_1, Sreg_2,
                    Sreg_block, READ_FROM_GLOBAL);
            for (int i = 0; i < KW; i++) {
                Wreg[i] = intel_sub_group_block_read4(
                        (__global uint *)&wei_copy_from[i * 4 * IC_BLOCK]);
            }
        } else {
            for (int i = 0; i < KW; i++) {
                Wreg[i] = 0;
            }
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

#if !USE_SLM_PIPE
        for (int kw = 0; kw < KW; kw++) {
            int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;

            unroll_for(int i = 0; i < OW_BLOCK; i++) {
                S0[i] = block_read(S_work + (kw * (1 + DW) + SW * i) * 8);
            }

            if (!use_zero) {
                BLOCK_READ_WEI(W0, 0);
                if (OC > 8) BLOCK_READ_WEI(W1, 8 * IC_BLOCK);
                if (OC > 16) BLOCK_READ_WEI(W2, 16 * IC_BLOCK);
                if (OC > 24) BLOCK_READ_WEI(W3, 24 * IC_BLOCK);
            }

            C00 = MMAD(S0, W0, C00);
            if (OC > 8) C01 = MMAD(S0, W1, C01);
            if (OC > 16) C02 = MMAD(S0, W2, C02);
            if (OC > 24) C03 = MMAD(S0, W3, C03);

            WEI += WEI_BLOCK;
        } // kw loop
#else // USE_SLM_PIPE == 1
        if (!is_first_mul) {
            // Use already written buffers
            __local WEI_DATA_T *wei_tmp
                    = wei_loc_base + WEI_SLM_BUFF_SIZE * idx_use;
            __local uint *S_work_curr = S_work + SRC_SLM_SIZE * idx_use;

            for (int kw = 0; kw < KW; kw++) {
                int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;

                unroll_for(int i = 0; i < OW_BLOCK; i++) {
                    S0[i] = block_read(
                            S_work_curr + (kw * (1 + DW) + SW * i) * 8);
                }
                BLOCK_READ_WEI(W0, 0);
                if (OC > 8) BLOCK_READ_WEI(W1, 8 * IC_BLOCK);
                if (OC > 16) BLOCK_READ_WEI(W2, 16 * IC_BLOCK);
                if (OC > 24) BLOCK_READ_WEI(W3, 24 * IC_BLOCK);
                C00 = MMAD(S0, W0, C00);
                if (OC > 8) C01 = MMAD(S0, W1, C01);
                if (OC > 16) C02 = MMAD(S0, W2, C02);
                if (OC > 24) C03 = MMAD(S0, W3, C03);
                WEI += WEI_BLOCK;
            } // kw loop
            idx_use ^= 1;
        }
        is_first_mul = false;

        // Write next buffers to SLM
        copy_src_to_slm(src, S_part_next, iw, ow, left_nozero_tail,
                right_nozero_tail, right_tail, Sreg_0, Sreg_1, Sreg_2,
                Sreg_block, WRITE_TO_LOCAL);
        for (int i = 0; i < KW; i++) {
            block_write4(
                    (__local uint *)&wei_copy_to[i * 4 * IC_BLOCK], Wreg[i]);
        }
        idx_read ^= 1;
        wei += WEI_BLOCK * KW;
#endif

        src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);

        if (kh == KH - 1) {
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
            if (kd == KD - 1) {
                src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
            }
        }
    } // combined loop

    if (ow < OW) {
        float4 tmp;
#if OC_BLOCK == 32
        DST_DATA4_T dst_pack0[BLOCK];
#else
        DST_DATA2_T dst_pack0[BLOCK];
        DST_DATA2_T dst_pack1[BLOCK];
#endif
        DST_DATA4_T D0[BLOCK];

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

#if WITH_SUM
#if OW_BLOCK == 4
#if OC_BLOCK == 32
        *(DST_DATA16_T *)D0 = BLOCK_READ_DST16(dst);
#else
        DST_DATA8_T D00 = BLOCK_READ_DST8(dst);
        DST_DATA8_T D01;
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D01 = BLOCK_READ_DST8(dst1);
        } else {
            D01 = 0;
        }
        *(DST_DATA16_T *)D0 = (DST_DATA16_T)(D00.s01, D01.s01, D00.s23, D01.s23,
                D00.s45, D01.s45, D00.s67, D01.s67);
#endif
#endif
#if OW_BLOCK == 8
#if OC_BLOCK == 32
        *(DST_DATA16_T *)(D0 + 0) = BLOCK_READ_DST16(dst);
        *(DST_DATA16_T *)(D0 + 4) = BLOCK_READ_DST16(dst + 16 * 8);
#else
        DST_DATA16_T D00 = BLOCK_READ_DST16(dst);
        DST_DATA16_T D01;
        if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                        <= OC_PADDED) {
            D01 = BLOCK_READ_DST16(dst1);
        } else {
            D01 = 0;
        }
        *(DST_DATA16_T *)D0 = (DST_DATA16_T)(D00.s01, D01.s01, D00.s23, D01.s23,
                D00.s45, D01.s45, D00.s67, D01.s67);
        *(DST_DATA16_T *)(D0 + 4) = (DST_DATA16_T)(D00.s89, D01.s89, D00.sab,
                D01.sab, D00.scd, D01.scd, D00.sef, D01.sef);
#endif
#endif
#endif

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#if OC_BLOCK == 32
#define CONVERT_PACK(idx) \
    do { \
        dst_pack0[idx] = CONVERT_DST_DATA4_T(tmp); \
    } while (0)

#define PACK_DST(C0, C1, C2, C3, D) \
    do { \
        for (int n_i = 0; n_i < OW_BLOCK; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            const int po_mb = group_mb * MB_BLOCK; \
            const int po_oc \
                    = (group_oc * OC_BLOCK + oc * OC_BLOCK) % (OC * G); \
            const float4 dni \
                    = convert_float4(SUM_TO_REF(AS_SUM_DATA4_T(D[n_i]))); \
            APPLY_POST_OPS_TRY_BURST(tmp, float, dni, float, po_mb, 1, po_oc, \
                    4 * SUB_GROUP_SIZE, subg_local_id); \
            CONVERT_PACK(n_i); \
        } \
    } while (0)
#else
#define CONVERT_PACK(idx) \
    do { \
        dst_pack0[idx] = TO_DST2(tmp.s01); \
        dst_pack1[idx] = TO_DST2(tmp.s23); \
    } while (0)

#define PACK_DST(C0, C1, C2, C3, D) \
    do { \
        for (int n_i = 0; n_i < OW_BLOCK; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            const int po_mb = group_mb * MB_BLOCK; \
            const int po_oc \
                    = (group_oc * OC_BLOCK + oc * OC_BLOCK) % (OC * G); \
            const float4 dni \
                    = convert_float4(SUM_TO_REF(AS_SUM_DATA4_T(D[n_i]))); \
            APPLY_POST_OPS_TRY_BURST(tmp, float, dni, float, po_mb, 1, po_oc, \
                    4 * SUB_GROUP_SIZE, subg_local_id); \
            CONVERT_PACK(n_i); \
        } \
    } while (0)
#endif

        PACK_DST(C00, C01, C02, C03, D0);
#if OW_TAIL
        if (ow + OW_BLOCK > OW) {
#if OC_BLOCK == 32
            for (int i = 0; i < OW_TAIL; i++) {
                BLOCK_WRITE_DST4(&dst[i * 32], dst_pack0[i]);
            }
#else
            for (int i = 0; i < OW_TAIL; i++) {
                BLOCK_WRITE_DST2(&dst[i * 16], dst_pack0[i]);
            }
            if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                    || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                            <= OC_PADDED) {
                for (int i = 0; i < OW_TAIL; i++) {
                    BLOCK_WRITE_DST2(&dst1[i * 16], dst_pack1[i]);
                }
            }
#endif
        } else {
#endif

#if OW_BLOCK == 2
            BLOCK_WRITE_DST4(dst, *(DST_DATA4_T *)dst_pack0);
            if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                    || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                            <= OC_PADDED) {
                BLOCK_WRITE_DST4(dst1, *(DST_DATA4_T *)dst_pack1);
            }
#endif

#if OW_BLOCK == 4
#if OC_BLOCK == 32
            BLOCK_WRITE_DST16(dst, *(DST_DATA16_T *)dst_pack0);
#else
            BLOCK_WRITE_DST8(dst, *(DST_DATA8_T *)dst_pack0);
            if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                    || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                            <= OC_PADDED) {
                BLOCK_WRITE_DST8(dst1, *(DST_DATA8_T *)dst_pack1);
            }
#endif
#endif

#if OW_BLOCK == 8
#if OC_BLOCK == 32
            BLOCK_WRITE_DST16(dst, *(DST_DATA16_T *)dst_pack0);
            BLOCK_WRITE_DST16(dst + 16 * 8, *(DST_DATA16_T *)(dst_pack0 + 4));
#else
            BLOCK_WRITE_DST8(dst, *(DST_DATA8_T *)dst_pack0);
            BLOCK_WRITE_DST8(dst + 8 * 8, *(DST_DATA8_T *)(dst_pack0 + 4));
            if (OC % OC_CALC_BLOCK == 0 || OC % OC_CALC_BLOCK > OC_BLOCK
                    || OC_CALC_BLOCK * (group_oc + oc) + OC_CALC_BLOCK
                            <= OC_PADDED) {
                BLOCK_WRITE_DST8(dst1, *(DST_DATA8_T *)dst_pack1);
                BLOCK_WRITE_DST8(dst1 + 8 * 8, *(DST_DATA8_T *)(dst_pack1 + 4));
            }
#endif
#endif
#if OW_TAIL
        }
#endif
    }
}
