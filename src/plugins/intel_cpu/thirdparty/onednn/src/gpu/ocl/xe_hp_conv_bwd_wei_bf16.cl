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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_types.h"

#define USHORT_PER_READ (16 * SUB_GROUP_SIZE)
#define INT_PER_READ (USHORT_PER_READ / 2)

#define WORKGROUP_SIZE (LWS_0 / SUB_GROUP_SIZE)

#define MAX_SGID_IC (MB_BLK_WORKGROUP * IC_BLK_WORKGROUP)
#define MAX_SGID_OC (MB_BLK_WORKGROUP * OC_BLK_WORKGROUP)
#define MAX_SGID_COMPUTE \
    ((OC_BLK_WORKGROUP / OC_BLK_SUBGROUP) \
            * (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP))

// Using hard-code strides instead of SRC_OFF/DST_OFF/WEI_OFF
// because compiler generates ugly code for SRC_OFF
#define SRC_W_STRIDE \
    (2 * MB_BLOCK * IC_BLOCK) // factor of 2 is because src fmt is NChw32n16c
#define SRC_H_STRIDE (IW * SRC_W_STRIDE)
#define SRC_D_STRIDE (IH * SRC_H_STRIDE)
#define SRC_C_STRIDE (ID * SRC_D_STRIDE)
#define SRC_MB_STRIDE (G * IC / IC_BLOCK * SRC_C_STRIDE)

// DST fmt is NChw32n16c
#define DST_W_STRIDE (2 * MB_BLOCK * OC_BLOCK)
#define DST_H_STRIDE (OW * DST_W_STRIDE)
#define DST_D_STRIDE (OH * DST_H_STRIDE)
#define DST_C_STRIDE (OD * DST_D_STRIDE)
#define DST_MB_STRIDE (G * OC / OC_BLOCK * DST_C_STRIDE)

#if USE_DPASW == 1
//NOTE: Dpasw can only be used when IC_BLK_WORKGROUP is multiple of 2
#define GEMM_IC_blk(o, i) \
    do { \
        ACC[o][2 * i] \
                = __dpasw(as_uint4(D[o]), as_int8(S[i][0]), ACC[o][2 * i]); \
        ACC[o][2 * i + 1] = __dpasw( \
                as_uint4(D[o]), as_int8(S[i][1]), ACC[o][2 * i + 1]); \
    } while (0)
#else
#define GEMM_IC_blk(o, i) \
    do { \
        ACC[o][2 * i] = mmad8x8_bf16( \
                as_uint8(D[o]), as_int8(S[i][0]), ACC[o][2 * i]); \
        ACC[o][2 * i + 1] = mmad8x8_bf16( \
                as_uint8(D[o]), as_int8(S[i][1]), ACC[o][2 * i + 1]); \
    } while (0)
#endif

#if USE_DPASW == 1

#if OC_BLK_SUBGROUP == 2
#define READ_DST() \
    do { \
        D[0] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset]); \
        D[1] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + INT_PER_READ]); \
        D[2] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + 2 * INT_PER_READ]); \
        D[3] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + 3 * INT_PER_READ]); \
    } while (0)
#else // OC_BLK_SUBGROUP == 1
#define READ_DST() \
    do { \
        D[0] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset]); \
        D[1] = block_read4(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + INT_PER_READ]); \
    } while (0)
#endif

#else // use normal dpas

#if OC_BLK_SUBGROUP == 2
#define READ_DST() \
    do { \
        D[0] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset]); \
        D[1] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + INT_PER_READ]); \
        D[2] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + 2 * INT_PER_READ]); \
        D[3] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + 3 * INT_PER_READ]); \
    } while (0)
#else // OC_BLK_SUBGROUP == 1
#define READ_DST() \
    do { \
        D[0] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset]); \
        D[1] = block_read8(&diff_dst_loc_read[loc_dst_compute_blk_offset \
                + INT_PER_READ]); \
    } while (0)
#endif // OC_BLK_SUBGROUP
#endif // USE DPAS_W

#define READ_SRC(i_c) \
    do { \
        S[i_c][0] = block_read8(&src_loc_read[loc_src_compute_blk_offset \
                + 2 * i_c * INT_PER_READ]); \
        S[i_c][1] = block_read8(&src_loc_read[loc_src_compute_blk_offset \
                + (2 * i_c + 1) * INT_PER_READ]); \
    } while (0)

#define ZERO_SRC(i_c) \
    do { \
        S[i_c][0] = 0; \
        S[i_c][1] = 0; \
    } while (0)

#define PACK(i) as_uint((short2)(D_tmp[0][i], D_tmp[1][i]))

#if WITH_BIAS
#define CONVERT_TO_F32(x) cvt_bf16_to_f32(x)

#define READ_DST_GLOBAL() \
    do { \
        dst_off = (size_t)n_block * DST_MB_STRIDE + od * DST_D_STRIDE \
                + oh * DST_H_STRIDE + ow * DST_W_STRIDE \
                + n_block_inner * MB_BLOCK * OC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &diff_dst[dst_off + USHORT_PER_READ]); \
    } while (0)

#define WRITE_DST() \
    do { \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[1].s0) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[1].s2) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[1].s4) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[1].s6) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[1].s8) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[1].sa) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[1].sc) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[1].se)); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s0) \
                + CONVERT_TO_F32(Dt[1].odd.s0); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s1) \
                + CONVERT_TO_F32(Dt[1].odd.s1); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s2) \
                + CONVERT_TO_F32(Dt[1].odd.s2); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s3) \
                + CONVERT_TO_F32(Dt[1].odd.s3); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s4) \
                + CONVERT_TO_F32(Dt[1].odd.s4); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s5) \
                + CONVERT_TO_F32(Dt[1].odd.s5); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s6) \
                + CONVERT_TO_F32(Dt[1].odd.s6); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s7) \
                + CONVERT_TO_F32(Dt[1].odd.s7); \
        vstore16((ushort16)(Dt[0].even, Dt[1].even), sg_loc_id, \
                diff_dst_loc_write[buf_num]); \
        vstore16((ushort16)(Dt[0].odd, Dt[1].odd), sg_loc_id + 8, \
                diff_dst_loc_write[buf_num]); \
    } while (0)

#else //WITHOUT  BIAS

#define READ_DST_GLOBAL() \
    do { \
        dst_off = (size_t)n_block * DST_MB_STRIDE + od * DST_D_STRIDE \
                + oh * DST_H_STRIDE + ow * DST_W_STRIDE \
                + n_block_inner * MB_BLOCK * OC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &diff_dst[dst_off + USHORT_PER_READ]); \
    } while (0)
#define WRITE_DST() \
    do { \
        vstore16((ushort16)(Dt[0].even, Dt[1].even), sg_loc_id, \
                diff_dst_loc_write[buf_num]); \
        vstore16((ushort16)(Dt[0].odd, Dt[1].odd), sg_loc_id + 8, \
                diff_dst_loc_write[buf_num]); \
    } while (0)

#endif // WITH_BIAS

#define READ_SRC_GLOBAL() \
    do { \
        src_off = (size_t)n_block * SRC_MB_STRIDE + id * SRC_D_STRIDE \
                + ih * SRC_H_STRIDE + iw * SRC_W_STRIDE \
                + n_block_inner * MB_BLOCK * IC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&src[src_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &src[src_off + USHORT_PER_READ]); \
    } while (0)
#define WRITE_SRC() \
    do { \
        block_write8(src_loc_write[buf_num], \
                (uint8)(as_uint(Dt[0].s02), as_uint(Dt[0].s46), \
                        as_uint(Dt[0].s8A), as_uint(Dt[0].sCE), \
                        as_uint(Dt[1].s02), as_uint(Dt[1].s46), \
                        as_uint(Dt[1].s8A), as_uint(Dt[1].sCE))); \
        block_write8(&src_loc_write[buf_num][INT_PER_READ], \
                (uint8)(as_uint(Dt[0].s13), as_uint(Dt[0].s57), \
                        as_uint(Dt[0].s9B), as_uint(Dt[0].sDF), \
                        as_uint(Dt[1].s13), as_uint(Dt[1].s57), \
                        as_uint(Dt[1].s9B), as_uint(Dt[1].sDF))); \
    } while (0)

// READ_SRC reads 16n block of src (block layout: 2c8n8c2n) from SLM
#if OC_BLK_SUBGROUP == 2
#define COMPUTE(i_c) \
    do { \
        GEMM_IC_blk(0, i_c); \
        GEMM_IC_blk(1, i_c); \
        GEMM_IC_blk(2, i_c); \
        GEMM_IC_blk(3, i_c); \
    } while (0)
#elif OC_BLK_SUBGROUP == 1
#define COMPUTE(i_c) \
    do { \
        GEMM_IC_blk(0, i_c); \
        GEMM_IC_blk(1, i_c); \
    } while (0)
#else
#error UNEXPECTED OC_BLK_SUBGROUP
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
xe_hp_conv_bwd_wei_bf16(const __global ushort *src, __global float *diff_wei,
        __global float *diff_bias, const __global ushort *diff_dst) {

    const int gid[2] = {get_group_id(0), get_group_id(1)};
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();

    // blocks which subgroup will read from global memory
    // e.g. threads TO,T1 read the same oc_block but different mb_block
    const int sgid_n_block = sg_id % MB_BLK_WORKGROUP;
    const int sgid_c_block = sg_id / MB_BLK_WORKGROUP;

    // compute blocks
    // threads T0, T1 compute the same oc_block but different ic_block
    const int sg_oc_blk = sg_id / (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP);
    const int sg_ic_blk = sg_id % (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP);

    const int workgroup_id = gid[0];
    const int group_ic = (workgroup_id % (IC / (IC_BLK_WORKGROUP * IC_BLOCK)))
            * IC_BLK_WORKGROUP;
    const int group_oc = (workgroup_id / (IC / (IC_BLK_WORKGROUP * IC_BLOCK)))
            * OC_BLK_WORKGROUP;

    const int group_g = (gid[1] / K_WORKGROUPS) / (KD * KH * KW);
    const int group_k_block = (gid[1] % K_WORKGROUPS) * K_BLOCKS;
    const int kd = (gid[1] / K_WORKGROUPS / KH / KW) % KD;
    const int kh = (gid[1] / K_WORKGROUPS / KW) % KH;
    const int kw = (gid[1] / K_WORKGROUPS) % KW;

    const int od_start = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
    const int oh_start = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
    const int ow_start = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);

    const int od_end
            = OD - max(0, (PD_R - (KD - 1 - kd) * (1 + DD) + SD - 1) / SD) - 1;
    const int oh_end
            = OH - max(0, (PH_R - (KH - 1 - kh) * (1 + DH) + SH - 1) / SH) - 1;
    const int ow_end
            = OW - max(0, (PW_R - (KW - 1 - kw) * (1 + DW) + SW - 1) / SW) - 1;

    // total accumulation dimension for given (kd,kh,kw)
    const int total_od = od_end - od_start + 1;
    const int total_oh = oh_end - oh_start + 1;
    const int total_ow = ow_end - ow_start + 1;
    const int mb_blk_rnd_up = (MB + MB_BLK_WORKGROUP * MB_BLOCK - 1)
            / (MB_BLK_WORKGROUP * MB_BLOCK);
    const int total_k_blocks = mb_blk_rnd_up * total_od * total_oh * total_ow;

    // last thread might do extra work if total_k_blocks % K_BLOCKS != 0
    const int max_k_blocks = ((gid[1] % K_WORKGROUPS) == K_WORKGROUPS - 1)
            ? max(0, total_k_blocks - group_k_block)
            : min(max(0, total_k_blocks - group_k_block), K_BLOCKS);

#if MB_BLK_WORKGROUP == 1 && MB > 16
    int n_block_inner = group_k_block
            % 2; // factor 2 because fmt is 32n16c and mb_block = 16
    int od = od_start + ((group_k_block / 2 / total_ow / total_oh) % total_od);
    int oh = oh_start + ((group_k_block / 2 / total_ow) % total_oh);
    int ow = ow_start + ((group_k_block / 2) % total_ow);

    int n_block = group_k_block / 2 / (total_od * total_oh * total_ow);
#else
    int n_block_inner = 0;
    int od = od_start + ((group_k_block / total_ow / total_oh) % total_od);
    int oh = oh_start + ((group_k_block / total_ow) % total_oh);
    int ow = ow_start + (group_k_block % total_ow);

    int n_block = group_k_block / (total_od * total_oh * total_ow);
#endif

    const int group_id = od * SD - PD + kd * (1 + DD);
    const int group_ih = oh * SH - PH + kh * (1 + DH);
    const int group_iw = ow * SW - PW + kw * (1 + DW);
    int id = group_id;
    int ih = group_ih;
    int iw = group_iw;

    // each subgroup may read (SRC:MB_BLOCK * IC_BLOCK + DST:MB_BLOCK * OC_BLOCK)
    // elements from global memory
    bool write_src_to_slm = sg_id < MAX_SGID_IC;
#if MAX_SGID_IC < WORKGROUP_SIZE
    if (write_src_to_slm)
#endif
        src += sgid_n_block * MB_BLOCK * IC_BLOCK
                + (group_g * IC / IC_BLOCK + group_ic + sgid_c_block)
                        * SRC_C_STRIDE;

    bool write_dst_to_slm = sg_id < MAX_SGID_OC;
#if MAX_SGID_OC < WORKGROUP_SIZE
    if (write_dst_to_slm)
#endif
        diff_dst += sgid_n_block * MB_BLOCK * OC_BLOCK
                + (group_g * OC / OC_BLOCK + group_oc + sgid_c_block)
                        * DST_C_STRIDE;

    bool compute_block = sg_id < MAX_SGID_COMPUTE;
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
    if (compute_block)
#endif
        diff_wei += WEI_OFF(group_g,
                (group_oc + sg_oc_blk * OC_BLK_SUBGROUP) * OC_BLOCK,
                (group_ic + sg_ic_blk * IC_BLK_SUBGROUP) * IC_BLOCK, kd, kh,
                kw);

#if WITH_BIAS
    float2 BIAS_ACC = 0.0f;
    bool compute_bias = group_ic == 0 && kd == min(PD, KD - 1)
            && kh == min(PH, KH - 1) && kw == min(PW, KW - 1)
            && write_dst_to_slm;
    size_t bia_off;
    volatile __global atomic_float *dbias;
    bia_off = group_g * OC + (group_oc + sgid_c_block) * OC_BLOCK;
    dbias = (volatile __global atomic_float *)&diff_bias[bia_off];
#endif // WITH_BIAS

    uint8 S[2][2];

#if USE_DPASW == 1
    uint4 D[4];
#else
    uint8 D[4];
#endif
    ushort16 Dt[2];

    float8 ACC[4][4] = {0.0f};

    __local uint src_slm[SRC_SLM_SIZE];
    __local uint diff_dst_slm[DST_SLM_SIZE];

    int src_slm_offset = MB_BLOCK * IC_BLK_WORKGROUP * IC_BLOCK / 2;
    int dst_slm_offset = MB_BLOCK * OC_BLK_WORKGROUP * OC_BLOCK / 2;

    const int loc_src_compute_blk_offset
            = sg_ic_blk * MB_BLOCK * IC_BLK_SUBGROUP * IC_BLOCK / 2;

#if USE_DPASW == 1
    const int loc_dst_compute_blk_offset
            = sg_oc_blk * MB_BLOCK * OC_BLK_SUBGROUP * OC_BLOCK / 2
            + sgid_mod_2 * SUB_GROUP_SIZE * 4;
#else
    const int loc_dst_compute_blk_offset
            = sg_oc_blk * MB_BLOCK * OC_BLK_SUBGROUP * OC_BLOCK / 2;
#endif

    const int loc_src_write_offset
            = sgid_c_block * USHORT_PER_READ + sgid_n_block * src_slm_offset;
    const int loc_dst_write_offset
            = sgid_c_block * USHORT_PER_READ + sgid_n_block * dst_slm_offset;
    __local uint *src_loc_write[NUM_BUF];
    __local ushort *diff_dst_loc_write[NUM_BUF];

    for (int i = 0; i < NUM_BUF; i++) {
        src_loc_write[i]
                = &src_slm[i * (SRC_SLM_SIZE / NUM_BUF) + loc_src_write_offset];
        diff_dst_loc_write[i]
                = (__local ushort *)&diff_dst_slm[i * (DST_SLM_SIZE / NUM_BUF)
                        + loc_dst_write_offset];
    }

    const __local uint *src_loc_read;
    const __local uint *diff_dst_loc_read;

    int k_blk_iter = 0;

    size_t src_off, dst_off;
    int buf_num = 0;

    for (; buf_num < min(max_k_blocks, NUM_BUF - 1); ++buf_num) {
#if MAX_SGID_IC < WORKGROUP_SIZE
        if (write_src_to_slm) {
#endif
            // Each subgroups reads block of 16n16c from global memory
            READ_SRC_GLOBAL();
            // Reorder the block to 2c8n8c2n before and write to SLM.
            // If mb_blk_workgroup=2
            // layout of a single src buffer in SLM will be 2n2Xc8n8c2n,
            // else it will be 2Xc8n8c2n (where X = IC_BLK_WORKGROUP).
            WRITE_SRC();
#if MAX_SGID_IC < WORKGROUP_SIZE
        }
#endif

#if MAX_SGID_OC < WORKGROUP_SIZE
        if (write_dst_to_slm) {
#endif
            // Each subgroups reads block of 16n16c from global memory
            READ_DST_GLOBAL();
            // Transpose the block to 16c16n before and write to SLM.
            // If mb_blk_workgroup=2,
            // layout of a single src buffer in SLM will be 2nXc16n,
            // else it will be Xc16n (where X = OC_BLK_WORKGROUP).
            WRITE_DST();
#if MAX_SGID_OC < WORKGROUP_SIZE
        }
#endif

#if MB_BLK_WORKGROUP == 1 && MB > 16
        n_block_inner++;
        if (n_block_inner == 2) {
            n_block_inner = 0;
            ow++;
            iw += SW;
        }
#else
        ow++;
        iw += SW;
#endif

        if (ow == ow_end + 1) {
            ow = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);
            oh++;
            iw = ow * SW - PW + kw * (1 + DW);
            ih += SH;
        }
        if (oh == oh_end + 1) {
            oh = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
            od++;
            ih = oh * SH - PH + kh * (1 + DH);
            id += SD;
        }
        if (od == od_end + 1) {
            od = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
            id = od * SD - PD + kd * (1 + DD);
            n_block++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int k_blk = 0; k_blk < max_k_blocks; ++k_blk) {

        buf_num = ((k_blk_iter % NUM_BUF) + NUM_BUF - 1) % NUM_BUF;

        src_loc_read
                = &src_slm[(k_blk_iter % NUM_BUF) * (SRC_SLM_SIZE / NUM_BUF)];
        diff_dst_loc_read = &diff_dst_slm[(k_blk_iter % NUM_BUF)
                * (DST_SLM_SIZE / NUM_BUF)];

        k_blk_iter++;

#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        if (compute_block) {
#endif
            // Read first 16n block of diff_dst
            // (block layout: Xc16c16n, X=OC_BLK_SUBGROUP) from SLM
            READ_DST();
            READ_SRC(0);
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        } else {
            ZERO_SRC(0);
        }
#endif
        // Compute 2Xo16i (X=OC_BLK_SUBGROUP) with reduction on first block of 16n
        COMPUTE(0);

#if IC_BLK_SUBGROUP == 2
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        if (compute_block) {
#endif
            READ_SRC(1);
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        } else {
            ZERO_SRC(1);
        }
#endif
        // Compute next IC_BLOCK, i.e.2Xo16i (X=OC_BLK_SUBGROUP)
        // with reduction on first block of 16n
        COMPUTE(1);
#endif // IC_BLK_SUBGROUP == 2

        if (k_blk < max_k_blocks - (NUM_BUF - 1)) {
#if MAX_SGID_IC < WORKGROUP_SIZE
            if (write_src_to_slm) {
#endif
                READ_SRC_GLOBAL();
                WRITE_SRC();
#if MAX_SGID_IC < WORKGROUP_SIZE
            }
#endif
#if MAX_SGID_OC < WORKGROUP_SIZE
            if (write_dst_to_slm) {
#endif
                READ_DST_GLOBAL();
                WRITE_DST();
#if MAX_SGID_OC < WORKGROUP_SIZE
            }
#endif

#if MB_BLK_WORKGROUP == 1 && MB > 16
            n_block_inner++;
            if (n_block_inner == 2) {
                n_block_inner = 0;
                ow++;
                iw += SW;
            }
#else
            ow++;
            iw += SW;
#endif
            if (ow == ow_end + 1) {
                ow = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);
                oh++;
                iw = ow * SW - PW + kw * (1 + DW);
                ih += SH;
            }
            if (oh == oh_end + 1) {
                oh = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
                od++;
                ih = oh * SH - PH + kh * (1 + DH);
                id += SD;
            }
            if (od == od_end + 1) {
                od = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
                id = od * SD - PD + kd * (1 + DD);
                n_block++;
            }
        }

#if MB_BLK_WORKGROUP == 2
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        if (compute_block) {
#endif
            src_loc_read += src_slm_offset;
            diff_dst_loc_read += dst_slm_offset;

            // Read second 16n block of diff_dst (block size: 2c16n16c) from SLM
            READ_DST();
            READ_SRC(0);
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        } else {
            ZERO_SRC(0);
        }
#endif
        // Reduce on the same block(32o32i) with reduction on second block of 16n
        COMPUTE(0);
#if IC_BLK_SUBGROUP == 2
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        if (compute_block) {
#endif
            READ_SRC(1);
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
        } else {
            ZERO_SRC(0);
        }
#endif
        COMPUTE(1);
#endif // IC_BLK_SUBGROUP == 2
#endif // MB_BLK_WORKGROUP == 2

        if (k_blk < max_k_blocks - (NUM_BUF - 1)) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    volatile __global atomic_float *diff_wei_write;

#define WRITE_WEI(i_o, i_i) \
    do { \
        diff_wei_write = (volatile __global atomic_float *)&diff_wei[WEI_OFF( \
                0, i_o * 8, i_i * IC_BLOCK + sg_loc_id, 0, 0, 0)]; \
        atomic_add_global(&diff_wei_write[0], ACC[i_o][2 * i_i].s0); \
        atomic_add_global(&diff_wei_write[16], ACC[i_o][2 * i_i].s1); \
        atomic_add_global(&diff_wei_write[32], ACC[i_o][2 * i_i].s2); \
        atomic_add_global(&diff_wei_write[48], ACC[i_o][2 * i_i].s3); \
        atomic_add_global(&diff_wei_write[64], ACC[i_o][2 * i_i].s4); \
        atomic_add_global(&diff_wei_write[80], ACC[i_o][2 * i_i].s5); \
        atomic_add_global(&diff_wei_write[96], ACC[i_o][2 * i_i].s6); \
        atomic_add_global(&diff_wei_write[112], ACC[i_o][2 * i_i].s7); \
        diff_wei_write += SUB_GROUP_SIZE; \
        atomic_add_global(&diff_wei_write[0], ACC[i_o][2 * i_i + 1].s0); \
        atomic_add_global(&diff_wei_write[16], ACC[i_o][2 * i_i + 1].s1); \
        atomic_add_global(&diff_wei_write[32], ACC[i_o][2 * i_i + 1].s2); \
        atomic_add_global(&diff_wei_write[48], ACC[i_o][2 * i_i + 1].s3); \
        atomic_add_global(&diff_wei_write[64], ACC[i_o][2 * i_i + 1].s4); \
        atomic_add_global(&diff_wei_write[80], ACC[i_o][2 * i_i + 1].s5); \
        atomic_add_global(&diff_wei_write[96], ACC[i_o][2 * i_i + 1].s6); \
        atomic_add_global(&diff_wei_write[112], ACC[i_o][2 * i_i + 1].s7); \
    } while (0)

#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
    if (compute_block) {
#endif
        if (max_k_blocks > 0) {
            WRITE_WEI(0, 0);
            WRITE_WEI(1, 0);
#if OC_BLK_SUBGROUP == 2
            WRITE_WEI(2, 0);
            WRITE_WEI(3, 0);
#endif

#if IC_BLK_SUBGROUP == 2
            WRITE_WEI(0, 1);
            WRITE_WEI(1, 1);
#if OC_BLK_SUBGROUP == 2
            WRITE_WEI(2, 1);
            WRITE_WEI(3, 1);
#endif
#endif
        }
#if MAX_SGID_COMPUTE < WORKGROUP_SIZE
    }
#endif

#if WITH_BIAS
#define COMPUTE_BIAS(nblk) \
    do { \
        dst_off = n * DST_MB_STRIDE + od * DST_D_STRIDE + oh * DST_H_STRIDE \
                + ow * DST_W_STRIDE + nblk * MB_BLOCK * OC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off + USHORT_PER_READ]); \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[1].s0) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[1].s2) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[1].s4) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[1].s6) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[1].s8) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[1].sa) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[1].sc) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[1].se)); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s0) \
                + CONVERT_TO_F32(Dt[1].odd.s0); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s1) \
                + CONVERT_TO_F32(Dt[1].odd.s1); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s2) \
                + CONVERT_TO_F32(Dt[1].odd.s2); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s3) \
                + CONVERT_TO_F32(Dt[1].odd.s3); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s4) \
                + CONVERT_TO_F32(Dt[1].odd.s4); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s5) \
                + CONVERT_TO_F32(Dt[1].odd.s5); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s6) \
                + CONVERT_TO_F32(Dt[1].odd.s6); \
        BIAS_ACC[1] += CONVERT_TO_F32(Dt[0].odd.s7) \
                + CONVERT_TO_F32(Dt[1].odd.s7); \
    } while (0)

    // handle padded region for bias computation
    // first thread in spatial gws dimension, handles the left padding
    if (compute_bias && gid[1] % K_WORKGROUPS == 0) {
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = 0; od < od_start; ++od) {
                for (oh = 0; oh < OH; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = 0; oh < oh_start; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_start; oh < OH; ++oh) {
                    for (ow = 0; ow < ow_start; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
    }

    // last thread handles the right padding
    if (compute_bias && gid[1] % K_WORKGROUPS == K_WORKGROUPS - 1) {
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_end + 1; oh < OH; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_end + 1; od < OD; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < od_end + 1; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_end + 1; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
    }
    if (compute_bias) {
        atomic_add_global(&dbias[sg_loc_id], BIAS_ACC.s0);
        atomic_add_global(&dbias[sg_loc_id + SUB_GROUP_SIZE], BIAS_ACC.s1);
    }
#endif
}
