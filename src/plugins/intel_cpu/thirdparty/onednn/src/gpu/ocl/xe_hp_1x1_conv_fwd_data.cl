/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* licensed under the apache license, version 2.0 (the "license");
* you may not use this file except in compliance with the license.
* you may obtain a copy of the license at
*
*     http://www.apache.org/licenses/license-2.0
*
* unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "as is" basis,
* without warranties or conditions of any kind, either express or implied.
* see the license for the specific language governing permissions and
* limitations under the license.
*******************************************************************************/

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#if SRC_DT_F16
#define MMAD8X8 mmad8x8_f16
#define MMAD8X4 mmad8x4_f16
#elif SRC_DT_BF16
#define MMAD8X8 mmad8x8_bf16
#define MMAD8X4 mmad8x4_bf16
#else
#define MMAD8X8 mmad8x8
#define MMAD8X4 mmad8x4
#endif

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL \
    ((IC - (IC & ~(IC_BLOCK - 1)) + (IC_BLOCK / 8) - 1) / (IC_BLOCK / 8))
#else
#define IC_NBLOCKS_TAIL 8
#endif

#define SRC_SP (IW * IH * ID)
#define SRC_MB_STRIDE IC_BLOCK
#define SRC_SP_STRIDE (SRC_MB_STRIDE * MB_BLOCK)
#define SRC_ICB_STRIDE (SRC_SP_STRIDE * SRC_SP)

#define DST_SP (OW * OH * OD)
#define DST_MB_STRIDE OC_BLOCK
#define DST_SP_STRIDE (DST_MB_STRIDE * MB_BLOCK)
#define DST_OCB_STRIDE (DST_SP_STRIDE * DST_SP)

#define WEI_BLOCK_STRIDE (IC_BLOCK * OC_CALC_BLOCK)
#define OC_BLOCK_NUMBER (OC_CALC_BLOCK / OC_BLOCK)

#if (MB_BLOCK == 32) || (SW == 1 && SH == 1 && SD == 1)
#define BLOCK_READ_SRC_4x32(data, idx) \
    data = AS_SRC_MMAD_DATA4_T( \
            intel_sub_group_block_read4((__global uint *)&src[idx]));
#define BLOCK_READ_SRC_8x32(data, idx) \
    data = AS_SRC_MMAD_DATA8_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));
#else
#define BLOCK_READ_SRC_4x32(data, idx) \
    do { \
        unroll_for(uint _i = 0; _i < 4; ++_i) { \
            data[_i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read( \
                    (__global uint *)&src[idx + _i * SW * IC_BLOCK])); \
        } \
    } while (0);
#define BLOCK_READ_SRC_8x32(data, idx) \
    do { \
        unroll_for(uint _i = 0; _i < 8; ++_i) { \
            data[_i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read( \
                    (__global uint *)&src[idx + _i * SW * IC_BLOCK])); \
        } \
    } while (0);
#endif // SW == 1 && SH == 1 && SD == 1

#if SP_BLOCK == 4
#define BLOCK0 4
#define ACC_DATA_BLOCK SRC_MMAD_ACC_DATA4_T
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA4_T
#define BLOCK_READ_SRC BLOCK_READ_SRC_4x32
#define MMAD_FULL0 MMAD8X4
#else
#define BLOCK0 8
#define ACC_DATA_BLOCK SRC_MMAD_ACC_DATA8_T
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA8_T
#define BLOCK_READ_SRC BLOCK_READ_SRC_8x32
#define MMAD_FULL0 MMAD8X8
#endif

#if SP_BLOCK == 12
#define BLOCK1 4
#define ACC_DATA_BLOCK1 SRC_MMAD_ACC_DATA4_T
#define SRC_DATA_BLOCK_T1 SRC_MMAD_DATA4_T
#define BLOCK_READ_SRC1 BLOCK_READ_SRC_4x32
#define MMAD_FULL1 MMAD8X4
#else
#define BLOCK1 8
#define ACC_DATA_BLOCK1 SRC_MMAD_ACC_DATA8_T
#define SRC_DATA_BLOCK_T1 SRC_MMAD_DATA8_T
#define BLOCK_READ_SRC1 BLOCK_READ_SRC_8x32
#define MMAD_FULL1 MMAD8X8
#endif

#if USE_WEI_SLM
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(block_read((__local uint *)&wei_tmp[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(block_read8((__local uint *)&wei_tmp[idx]));
#define WEI wei_tmp
#else
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#define WEI wei
#endif

#if BIA_DT_F32
#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));
#elif BIA_DT_F16
#define BLOCK_READ_BIA(data, idx) \
    data = convert_float4(as_half4( \
            intel_sub_group_block_read_us4((__global ushort *)&bias[idx])));
#elif BIA_DT_BF16
#define BLOCK_READ_BIA(data, idx) \
    data = cvt_bf16_to_f32( \
            intel_sub_group_block_read_us4((__global ushort *)&bias[idx]));
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

// Reads (n * 4) elements per work-item.
void block_read_dst(int n, DST_DATA_T *d, const __global DST_DATA_T *dst);

// Writes (n * 4) elements per work-item.
void block_write_dst(int n, const DST_DATA_T *d, __global DST_DATA_T *dst);

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
xe_hp_1x1_conv_fwd(const __global SRC_DATA_T *src,
        const __global WEI_DATA_T *wei, const __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint mb_group_id = get_group_id(2);
    const uint ic_group_id
            = oc_group_id * (OC_CALC_BLOCK / OC_BLOCK) / OC_NCHUNK * IC_NCHUNK;

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

    // Spatial
    const uint sp = get_global_id(1);
    const int sp_local_id = get_local_id(1);

    // IC splitting
    const uint ic_split_id = get_local_id(0) / SUB_GROUP_SIZE;
#define OWB ((OW + SP_BLOCK - 1) / SP_BLOCK)

    const uint od = sp / (OWB * OH);
    const uint ohw = sp % (OWB * OH);
    const uint oh = ohw / OWB;
    const uint ow = (ohw % OWB) * SP_BLOCK;

    const uint id = SD * od;
    const uint ih = SH * oh;
    const uint iw = SW * ow;

    // Source (At ic = 0)
#if MB_BLOCK == 32
    src += mb_group_id * SRC_ICB_STRIDE * IC_NCHUNK * G; // MB offset
#else
    src += mb_group_id * SRC_ICB_STRIDE * IC_NCHUNK * G; // MB offset
#endif
    src += ic_group_id * SRC_ICB_STRIDE
            + ic_split_id * SRC_ICB_STRIDE * IC_NCHUNK / IC_SPLIT; // IC offset
    src += (id * IH * IW + ih * IW + iw) * SRC_SP_STRIDE; // SP offset

    // Destination
#if MB_BLOCK == 32
    dst += mb_group_id * DST_OCB_STRIDE * OC_NCHUNK * G; // MB offset
#else
    dst += mb_group_id * DST_OCB_STRIDE * OC_NCHUNK * G; // MB offset
#endif
    dst += oc_group_id * DST_OCB_STRIDE * OC_BLOCK_NUMBER; // OC offset
    dst += (od * OH * OW + oh * OW + ow) * SRC_SP_STRIDE; // SP offset
#if OC_BLOCK == 16
    __global DST_DATA_T *dst1 = dst + DST_OCB_STRIDE;
#endif
    // Weights
    wei += oc_group_id * WEI_BLOCK_STRIDE * IC_NCHUNK
            + ic_split_id * WEI_BLOCK_STRIDE * IC_NCHUNK / IC_SPLIT;
    // Output accumulators:

    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK1 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    ACC_DATA_BLOCK1 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    ACC_DATA_BLOCK1 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

#if IC_SPLIT > 1
#define C_SLM_SIZE (LWS_0 * LWS_1 * LWS_2)
    __local ACC_DATA_BLOCK L_C00[C_SLM_SIZE], L_C01[C_SLM_SIZE],
            L_C02[C_SLM_SIZE], L_C03[C_SLM_SIZE];
#if MB_BLOCK == 32 || SP_BLOCK > 8
#error "Not expected"
#endif
#endif

#define WEI_LOOP_STRIDE (IC_LOOP_GROUPS * WEI_BLOCK_STRIDE)
#if USE_WEI_SLM

#if USE_DOUBLE_BUFFER
#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (wei_last > wei) { \
        const __global WEI_DATA_T *wei_copy_from \
                = wei + sp_local_id * WEI_LOOP_STRIDE / LWS_1; \
        __local WEI_DATA_T *wei_copy_to = wei_slm \
                + sp_local_id * WEI_LOOP_STRIDE / LWS_1 \
                + read_ind * WEI_LOOP_STRIDE; \
        for (int bl = 0; bl < IC_LOOP_GROUPS; bl++) { \
            block_write4((__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK], \
                    intel_sub_group_block_read4((__global uint \
                                    *)&wei_copy_from[bl * 4 * IC_BLOCK])); \
        } \
    } \
    __local WEI_DATA_T *wei_tmp = wei_slm + calc_ind * WEI_LOOP_STRIDE; \
    read_ind ^= 1; \
    calc_ind ^= 1;

    __local WEI_DATA_T wei_slm[WEI_LOOP_STRIDE * 2];
    const __global WEI_DATA_T *wei_last = wei + WEI_BLOCK_STRIDE * IC_NCHUNK;
    int read_ind = 0;
    int calc_ind = 1;
    READ_SLM()
    wei += WEI_LOOP_STRIDE;

#else

#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const __global WEI_DATA_T *wei_copy_from \
            = wei + sp_local_id * WEI_LOOP_STRIDE / LWS_1; \
    __local WEI_DATA_T *wei_copy_to = wei_slm \
            + sp_local_id * WEI_LOOP_STRIDE / LWS_1 \
            + ic_split_id * WEI_LOOP_STRIDE; \
    for (int bl = 0; bl < IC_LOOP_GROUPS; bl++) { \
        block_write4((__local uint *)&wei_copy_to[bl * 4 * IC_BLOCK], \
                intel_sub_group_block_read4( \
                        (__global uint *)&wei_copy_from[bl * 4 * IC_BLOCK])); \
    } \
    __local WEI_DATA_T *wei_tmp = wei_slm + ic_split_id * WEI_LOOP_STRIDE; \
    barrier(CLK_LOCAL_MEM_FENCE);

    __local WEI_DATA_T wei_slm[IC_SPLIT * WEI_LOOP_STRIDE];

#endif
#endif // USE_WEI_SLM

    for (uint ic_block_id = 0;
            ic_block_id < (IC_NCHUNK / IC_SPLIT) / IC_LOOP_GROUPS;
            ++ic_block_id) {
#if USE_WEI_SLM
        READ_SLM()
#endif // USE_WEI_SLM

        __attribute__((opencl_unroll_hint)) // attr:no-format
        for (int i = 0; i < IC_LOOP_GROUPS; ++i) {
            SRC_DATA_BLOCK_T S0 = 0;
            SRC_DATA_BLOCK_T1 S1 = 0;
            SRC_DATA_BLOCK_T1 S2 = 0;
            SRC_DATA_BLOCK_T1 S3 = 0;
#if USE_WEI_SLM && SP_TAIL
            if (sp < OW * OH * OD)
#endif // SP_TAIL
            {
#if OUT_SP_TAIL
                if (ow + SP_BLOCK > OW) {
#if OUT_SP_TAIL < 8
                    S0 = 0;
                    for (int i = 0; i < OUT_SP_TAIL; ++i) {
                        S0[i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read(
                                (__global uint *)&src[i * SW * IC_BLOCK]));
                    }
#else
                    BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
                    S1 = 0;
                    for (int i = 8; i < OUT_SP_TAIL; ++i) {
                        S1[i - 8] = AS_SRC_MMAD_DATA_T(
                                intel_sub_group_block_read((__global uint
                                                *)&src[i * SW * IC_BLOCK]));
                    }
#endif
                } else
#endif // OUT_SP_TAIL

                {
                    BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
#if (MB_BLOCK == 32 && MB > 8)
                    BLOCK_READ_SRC1(S1, 8 * IC_BLOCK);
#elif SP_BLOCK > 8
                    BLOCK_READ_SRC1(S1, 8 * SW * IC_BLOCK);
#endif

#if (MB_BLOCK == 32 && MB > 16)
                    BLOCK_READ_SRC1(S2, 16 * IC_BLOCK);
#endif

#if (MB_BLOCK == 32 && MB > 24)
                    BLOCK_READ_SRC1(S3, 24 * IC_BLOCK);
#endif
                }
            }

            int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;
#if IC % IC_BLOCK != 0
            if (ic_block_id == IC_NCHUNK - 1) {
                unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                        BLOCK_READ_WHT_1x32(W0[i], (i + 0) * IC_BLOCK);
                if (OC > 8)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W1[i], (i + 8) * IC_BLOCK);
                if (OC > 16)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W2[i], (i + 16) * IC_BLOCK);
                if (OC > 24)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W3[i], (i + 24) * IC_BLOCK);
            } else
#endif // IC % IC_BLOCK != 0
            {
                BLOCK_READ_WHT_8x32(W0, 0);
                if (OC > 8) BLOCK_READ_WHT_8x32(W1, 8 * IC_BLOCK);
                if (OC > 16) BLOCK_READ_WHT_8x32(W2, 16 * IC_BLOCK);
                if (OC > 24) BLOCK_READ_WHT_8x32(W3, 24 * IC_BLOCK);
            }

            C00 = MMAD_FULL0(S0, W0, C00);
            if (OC > 8) C01 = MMAD_FULL0(S0, W1, C01);
            if (OC > 16) C02 = MMAD_FULL0(S0, W2, C02);
            if (OC > 24) C03 = MMAD_FULL0(S0, W3, C03);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
            C10 = MMAD_FULL1(S1, W0, C10);
            if (OC > 8) C11 = MMAD_FULL1(S1, W1, C11);
            if (OC > 16) C12 = MMAD_FULL1(S1, W2, C12);
            if (OC > 24) C13 = MMAD_FULL1(S1, W3, C13);
#endif

#if MB_BLOCK == 32 && MB > 16
            C20 = MMAD_FULL1(S2, W0, C20);
            if (OC > 8) C21 = MMAD_FULL1(S2, W1, C21);
            if (OC > 16) C22 = MMAD_FULL1(S2, W2, C22);
            if (OC > 24) C23 = MMAD_FULL1(S2, W3, C23);
#endif

#if MB_BLOCK == 32 && MB > 24
            C30 = MMAD_FULL1(S3, W0, C30);
            if (OC > 8) C31 = MMAD_FULL1(S3, W1, C31);
            if (OC > 16) C32 = MMAD_FULL1(S3, W2, C32);
            if (OC > 24) C33 = MMAD_FULL1(S3, W3, C33);
#endif
            src += SRC_ICB_STRIDE;
            WEI += WEI_BLOCK_STRIDE;
        }
#if USE_WEI_SLM
        wei += WEI_LOOP_STRIDE;
#endif
    }

#if IC_SPLIT > 1
    if (ic_split_id > 0) {
        int idx = get_local_id(0) * LWS_1 + sp_local_id;
        L_C00[idx] = C00;
        L_C01[idx] = C01;
        L_C02[idx] = C02;
        L_C03[idx] = C03;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_split_id == 0) {
        for (int i = 1; i < IC_SPLIT; i++) {
            int idx = (get_local_id(0) + i * SUB_GROUP_SIZE) * LWS_1
                    + sp_local_id;
            C00 += L_C00[idx];
            C01 += L_C01[idx];
            C02 += L_C02[idx];
            C03 += L_C03[idx];
        }
    } else
        return;
#endif

    float2 tmp2;
    float4 tmp4;
    DST_DATA2_T dst_pack2[8];
    DST_DATA4_T dst_pack4[8];
    DST_DATA2_T D00[BLOCK0] = {0};
    DST_DATA2_T D01[BLOCK1] = {0};
    DST_DATA2_T D02[BLOCK1] = {0};
    DST_DATA2_T D03[BLOCK1] = {0};
    DST_DATA2_T D10[BLOCK0] = {0};
    DST_DATA2_T D11[BLOCK1] = {0};
    DST_DATA2_T D12[BLOCK1] = {0};
    DST_DATA2_T D13[BLOCK1] = {0};
    DST_DATA4_T D0[BLOCK0] = {0};
    DST_DATA4_T D1[BLOCK1] = {0};
    DST_DATA4_T D2[BLOCK1] = {0};
    DST_DATA4_T D3[BLOCK1] = {0};

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, oc_group_id * OC_CALC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, oc_group_id * OC_CALC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS4() tmp4 = fma(tmp4, (float4)SCALE, bia);
#define QUANTIZE_ADD_BIAS2(id) \
    tmp2 = fma(tmp2, (float2)SCALE, (float2)(bia[2 * id], bia[2 * id + 1]));
#else
#define QUANTIZE_ADD_BIAS4() tmp4 *= SCALE;
#define QUANTIZE_ADD_BIAS2(id) tmp2 *= SCALE;
#endif

#if WITH_SUM
    if (OUT_SP_TAIL && ow + SP_BLOCK > OW) {
#if OUT_SP_TAIL < 8
#if OC_BLOCK == 32
        block_read_dst(OUT_SP_TAIL, D0, dst);
#else
        block_read_dst(OUT_SP_TAIL, D00, dst);
        if ((OC - 1) % OC_CALC_BLOCK + 1 > OC_BLOCK
                || oc_group_id * OC_CALC_BLOCK + OC_BLOCK
                        < OC_NCHUNK * OC_BLOCK) {
            block_read_dst(OUT_SP_TAIL, D10, dst1);
        }
#endif
#else
#if OC_BLOCK == 32
        block_read_dst(BLOCK0, D0, dst);
        block_read_dst(OUT_SP_TAIL - 8, D1, dst + 8 * OC_BLOCK);
#else
        block_read_dst(BLOCK0, D00, dst);
        block_read_dst(OUT_SP_TAIL - 8, D01, dst + 8 * OC_BLOCK);
        if ((OC - 1) % OC_CALC_BLOCK + 1 > OC_BLOCK
                || oc_group_id * OC_CALC_BLOCK + OC_BLOCK
                        < OC_NCHUNK * OC_BLOCK) {
            block_read_dst(BLOCK0, D10, dst1);
            block_read_dst(OUT_SP_TAIL - 8, D11, dst1 + 8 * OC_BLOCK);
        }
#endif
#endif
    } else {
#if OC_BLOCK == 32
        block_read_dst(BLOCK0, D0, dst);
        if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8)) {
            block_read_dst(BLOCK1, D1, dst + 8 * OC_BLOCK);
        }
        if (MB_BLOCK == 32 && MB > 16) {
            block_read_dst(BLOCK1, D2, dst + 16 * OC_BLOCK);
        }
        if (MB_BLOCK == 32 && MB > 24) {
            block_read_dst(BLOCK1, D3, dst + 24 * OC_BLOCK);
        }
#else
        block_read_dst(BLOCK0, D00, dst);
        if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8)) {
            block_read_dst(BLOCK1, D01, dst + 8 * OC_BLOCK);
        }
        if (MB_BLOCK == 32 && MB > 16) {
            block_read_dst(BLOCK1, D02, dst + 16 * OC_BLOCK);
        }
        if (MB_BLOCK == 32 && MB > 24) {
            block_read_dst(BLOCK1, D03, dst + 24 * OC_BLOCK);
        }
        if ((OC - 1) % OC_CALC_BLOCK + 1 > OC_BLOCK
                || oc_group_id * OC_CALC_BLOCK + OC_BLOCK
                        < OC_NCHUNK * OC_BLOCK) {
            block_read_dst(BLOCK0, D10, dst1);
            if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8)) {
                block_read_dst(BLOCK1, D11, dst1 + 8 * OC_BLOCK);
            }
            if (MB_BLOCK == 32 && MB > 16) {
                block_read_dst(BLOCK1, D12, dst1 + 16 * OC_BLOCK);
            }
            if (MB_BLOCK == 32 && MB > 24) {
                block_read_dst(BLOCK1, D13, dst1 + 24 * OC_BLOCK);
            }
        }
#endif
    }
#endif // with_sum

#define PACK4(C0, C1, C2, C3, idx) \
    do { \
        tmp4[0] = C0[idx]; \
        tmp4[1] = C1[idx]; \
        tmp4[2] = C2[idx]; \
        tmp4[3] = C3[idx]; \
    } while (0)

#define CONVERT_PACK4(idx) \
    do { \
        dst_pack4[idx] = CONVERT_DST_DATA4_T(tmp4); \
    } while (0)

#define APPLY_POST_OPS_COMMON( \
        dcntr, n_i, accumulator, D, oc_stride, mb_stride) \
    { \
        int po_mb; \
        if (MB_BLOCK == 32) \
            po_mb = (mb_group_id * MB_BLOCK + mb_stride * 8 + n_i) % MB; \
        else \
            po_mb = mb_group_id % MB; \
        const int po_oc \
                = (oc_group_id * OC_BLOCK * OC_BLOCK_NUMBER + oc_stride) \
                % (OC * G); \
        float4 dni; \
        unroll_for(int didx = 0; didx < dcntr; ++didx) { \
            dni[didx] = convert_float(SUM_TO_REF(AS_SUM_DATA_T(D[didx]))); \
        } \
        APPLY_POST_OPS_TRY_BURST(accumulator, float, dni, float, po_mb, 1, \
                po_oc, dcntr *SUB_GROUP_SIZE, sg_local_id); \
        if (OC % 32) { /* clear zero pad area */ \
            unroll_for(int didx = 0; didx < dcntr; ++didx) { \
                if (po_oc + didx * SUB_GROUP_SIZE + sg_local_id >= OC) { \
                    accumulator[didx] = 0.f; \
                } \
            } \
        } \
    }

#define STORE_DST4(n, C0, C1, C2, C3, D, dst_ptr, mb_stride) \
    do { \
        for (int n_i = 0; n_i < n; n_i++) { \
            PACK4(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS4(); \
            APPLY_POST_OPS_COMMON(4, n_i, tmp4, D[n_i], 0, mb_stride); \
            CONVERT_PACK4(n_i); \
        } \
        block_write_dst(n, dst_pack4, dst_ptr); \
    } while (0)

#define PACK2(C0, C1, idx) \
    do { \
        tmp2[0] = C0[idx]; \
        tmp2[1] = C1[idx]; \
    } while (0)

#define CONVERT_PACK2(idx) \
    do { \
        dst_pack2[idx] = TO_DST2(tmp2); \
    } while (0)

#define STORE_DST2(n, C0, C1, D, dst_ptr, mb_stride, id) \
    do { \
        for (int n_i = 0; n_i < n; n_i++) { \
            PACK2(C0, C1, n_i); \
            QUANTIZE_ADD_BIAS2(id); \
            APPLY_POST_OPS_COMMON( \
                    2, n_i, tmp2, D[n_i], id *OC_BLOCK, mb_stride); \
            CONVERT_PACK2(n_i); \
        } \
        block_write_dst(n, dst_pack2, dst_ptr); \
    } while (0)

#if USE_WEI_SLM && SP_TAIL
    if (sp < OW * OH * OD)
#endif
    {
        if (OUT_SP_TAIL && ow + SP_BLOCK > OW) {
#if OC_BLOCK == 32
            STORE_DST4(
                    min(BLOCK0, OUT_SP_TAIL), C00, C01, C02, C03, D0, dst, 0);
            STORE_DST4(OUT_SP_TAIL - 8, C10, C11, C12, C13, D1,
                    dst + 8 * OC_BLOCK, 1);
#else
            STORE_DST2(min(BLOCK0, OUT_SP_TAIL), C00, C01, D00, dst, 0, 0);
            STORE_DST2(
                    OUT_SP_TAIL - 8, C10, C11, D01, dst + 8 * OC_BLOCK, 1, 0);
            if ((OC - 1) % OC_CALC_BLOCK + 1 > OC_BLOCK
                    || oc_group_id * OC_CALC_BLOCK + OC_BLOCK
                            < OC_NCHUNK * OC_BLOCK) {
                STORE_DST2(min(BLOCK0, OUT_SP_TAIL), C02, C03, D10, dst1, 0, 1);
                STORE_DST2(OUT_SP_TAIL - 8, C12, C13, D11, dst1 + 8 * OC_BLOCK,
                        1, 1);
            }
#endif
        } else {
#if OC_BLOCK == 32
            STORE_DST4(BLOCK0, C00, C01, C02, C03, D0, dst, 0);
            if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8))
                STORE_DST4(
                        BLOCK1, C10, C11, C12, C13, D1, dst + 8 * OC_BLOCK, 1);
            if (MB_BLOCK == 32 && MB > 16)
                STORE_DST4(
                        BLOCK1, C20, C21, C22, C23, D2, dst + 16 * OC_BLOCK, 2);
            if (MB_BLOCK == 32 && MB > 24)
                STORE_DST4(
                        BLOCK1, C30, C31, C32, C33, D3, dst + 24 * OC_BLOCK, 3);
#else
            STORE_DST2(BLOCK0, C00, C01, D00, dst, 0, 0);
            if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8))
                STORE_DST2(BLOCK1, C10, C11, D01, dst + 8 * OC_BLOCK, 1, 0);
            if (MB_BLOCK == 32 && MB > 16)
                STORE_DST2(BLOCK1, C20, C21, D02, dst + 16 * OC_BLOCK, 2, 0);
            if (MB_BLOCK == 32 && MB > 24)
                STORE_DST2(BLOCK1, C30, C31, D03, dst + 24 * OC_BLOCK, 3, 0);
            if ((OC - 1) % OC_CALC_BLOCK + 1 > OC_BLOCK
                    || oc_group_id * OC_CALC_BLOCK + OC_BLOCK
                            < OC_NCHUNK * OC_BLOCK) {
                STORE_DST2(BLOCK0, C02, C03, D10, dst1, 0, 1);
                if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8))
                    STORE_DST2(
                            BLOCK1, C12, C13, D11, dst1 + 8 * OC_BLOCK, 1, 1);
                if (MB_BLOCK == 32 && MB > 16)
                    STORE_DST2(
                            BLOCK1, C22, C23, D12, dst1 + 16 * OC_BLOCK, 2, 1);
                if (MB_BLOCK == 32 && MB > 24)
                    STORE_DST2(
                            BLOCK1, C32, C33, D13, dst1 + 24 * OC_BLOCK, 3, 1);
            }
#endif
        }
    }
}

// Reads (n * 4) elements per work-item.
void block_read_dst(int n, DST_DATA_T *d, const __global DST_DATA_T *dst) {
#if OC_BLOCK == 32
    const int nelems = n * 4;
#else
    const int nelems = n * 2;
#endif
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < nelems / 16 * 16; i += 16) {
        *((DST_DATA16_T *)&d[i]) = BLOCK_READ_DST16(dst + i * 8);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 16 * 16; i < nelems / 8 * 8; i += 8) {
        *((DST_DATA8_T *)&d[i]) = BLOCK_READ_DST8(dst + i * 8);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 8 * 8; i < nelems / 4 * 4; i += 4) {
        *((DST_DATA4_T *)&d[i]) = BLOCK_READ_DST4(dst + i * 8);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 4 * 4; i < nelems; i += 2) {
        *((DST_DATA2_T *)&d[i]) = BLOCK_READ_DST2(dst + i * 8);
    }
}

// Writes (n * 4) elements per work-item.
void block_write_dst(int n, const DST_DATA_T *d, __global DST_DATA_T *dst) {
#if OC_BLOCK == 32
    const int nelems = n * 4;
#else
    const int nelems = n * 2;
#endif
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < nelems / 16 * 16; i += 16) {
        BLOCK_WRITE_DST16(dst + i * 8, *((DST_DATA16_T *)&d[i]));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 16 * 16; i < nelems / 8 * 8; i += 8) {
        BLOCK_WRITE_DST8(dst + i * 8, *((DST_DATA8_T *)&d[i]));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 8 * 8; i < nelems / 4 * 4; i += 4) {
        BLOCK_WRITE_DST4(dst + i * 8, *((DST_DATA4_T *)&d[i]));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 4 * 4; i < nelems; i += 2) {
        BLOCK_WRITE_DST2(dst + i * 8, *((DST_DATA2_T *)&d[i]));
    }
}
