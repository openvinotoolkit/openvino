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
#include "gpu/ocl/ocl_zero_points.h"

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL ((IC - (IC & ~(IC_BLOCK - 1)) + 3) / 4)
#else
#define IC_NBLOCKS_TAIL 8
#endif

#define OWB ((OW + SP_BLOCK - 1) / SP_BLOCK)
#define HAS_PADDING (PW > 0 || PH > 0 || PD > 0)
#define USE_SP_BLOCK (SW == 1 && SH == 1 && SW == 1) || HAS_PADDING

#define SRC_SP (IW * IH * ID)
#define SRC_MB_STRIDE IC_BLOCK
#define SRC_SP_STRIDE (SRC_MB_STRIDE * MB_BLOCK)
#define SRC_ICB_STRIDE (SRC_SP_STRIDE * SRC_SP)

#define DST_SP (OW * OH * OD)
#define DST_MB_STRIDE IC_BLOCK
#define DST_SP_STRIDE (DST_MB_STRIDE * MB_BLOCK)
#define DST_OCB_STRIDE (DST_SP_STRIDE * DST_SP)

#define WEI_BLOCK_STRIDE (4 * 8 * 8 * 4)

#if (MB_BLOCK == 32) \
        || ((PW == 0 && PH == 0 && PW == 0) \
                && (SW == 1 && SH == 1 && SD == 1))
#define BLOCK_READ_SRC_4x32(data, idx, sp_off) \
    data = AS_SRC_MMAD_DATA4_T( \
            intel_sub_group_block_read4((__global uint *)&src[idx]));
#define BLOCK_READ_SRC_8x32(data, idx, sp_off) \
    data = AS_SRC_MMAD_DATA8_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));
#else
#define BLOCK_READ_SRC_4x32(data, idx, sp_off) \
    do { \
        unroll_for(uint _i = 0; _i < 4; ++_i) { \
            if (HAS_PADDING) { \
                PAD_BLOCK_READ(data[_i], src, sp, _i + sp_off); \
            } else { \
                data[_i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read( \
                        (__global uint *)&src[idx + _i * SW * IC_BLOCK])); \
            } \
        } \
    } while (0);
#define BLOCK_READ_SRC_8x32(data, idx, sp_off) \
    do { \
        unroll_for(uint _i = 0; _i < 8; ++_i) { \
            if (HAS_PADDING) { \
                PAD_BLOCK_READ(data[_i], src, sp, _i + sp_off); \
            } else { \
                data[_i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read( \
                        (__global uint *)&src[idx + _i * SW * IC_BLOCK])); \
            } \
        } \
    } while (0);
#endif // SW == 1 && SH == 1 && SD == 1

#if SP_BLOCK == 4
#define BLOCK0 4
#define ACC_DATA_BLOCK int4
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA4_T
#define BLOCK_READ_SRC BLOCK_READ_SRC_4x32

DECLARE_MMAD_EMU(mmad_tail0, idot4, IC_NBLOCKS_TAIL, 4, SRC_DATA_BLOCK_T, int8,
        ACC_DATA_BLOCK)

#define MMAD_FULL0 mmad8x4
#define MMAD_TAIL0 mmad_tail0
#else
#define BLOCK0 8
#define ACC_DATA_BLOCK int8
#define SRC_DATA_BLOCK_T SRC_MMAD_DATA8_T
#define BLOCK_READ_SRC BLOCK_READ_SRC_8x32

DECLARE_MMAD_EMU(mmad_tail0, idot4, IC_NBLOCKS_TAIL, 8, SRC_DATA_BLOCK_T, int8,
        ACC_DATA_BLOCK)

#define MMAD_FULL0 mmad8x8
#define MMAD_TAIL0 mmad_tail0
#endif

#if SP_BLOCK == 12
#define BLOCK1 4
#define ACC_DATA_BLOCK1 int4
#define SRC_DATA_BLOCK_T1 SRC_MMAD_DATA4_T
#define DST_DATA_BLOCK_T1 uint4
#define BLOCK_READ_SRC1 BLOCK_READ_SRC_4x32

DECLARE_MMAD_EMU(mmad_tail1, idot4, IC_NBLOCKS_TAIL, 4, SRC_DATA_BLOCK_T1, int8,
        ACC_DATA_BLOCK1)

#define MMAD_FULL1 mmad8x4
#define MMAD_TAIL1 mmad_tail1
#else
#define BLOCK1 8
#define ACC_DATA_BLOCK1 int8
#define SRC_DATA_BLOCK_T1 SRC_MMAD_DATA8_T
#define DST_DATA_BLOCK_T1 uint8
#define BLOCK_READ_SRC1 BLOCK_READ_SRC_8x32
DECLARE_MMAD_EMU(mmad_tail1, idot4, IC_NBLOCKS_TAIL, 8, SRC_DATA_BLOCK_T1, int8,
        ACC_DATA_BLOCK1)
#define MMAD_FULL1 mmad8x8
#define MMAD_TAIL1 mmad_tail1
#endif

#if INT8_WEI_SLM
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(block_read((__local uint *)&wei_tmp[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(block_read8((__local uint *)&wei_tmp[idx]));
#else
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#endif

#if OC % OC_BLOCK == 0
#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#else
#define BLOCK_READ_BIA(data, idx) \
    data = (float4)0; \
    int i; \
    for (i = idx; i < idx + OC_BLOCK && i < OC - (OC % SUB_GROUP_SIZE); \
            i += SUB_GROUP_SIZE) { \
        data[(i - idx) / SUB_GROUP_SIZE] = as_float( \
                intel_sub_group_block_read((__global uint *)&bias[i])); \
    } \
    if (i < OC && (i - idx) / SUB_GROUP_SIZE < 4 \
            && get_sub_group_local_id() < OC % SUB_GROUP_SIZE) { \
        data[(i - idx) / SUB_GROUP_SIZE] \
                = as_float(bias[i + get_sub_group_local_id()]); \
    }

#endif

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#define PAD_BLOCK_READ(data, src, sp, i) \
    do { \
        const int od = (sp + i) / (OW * OH); \
        const int ohw = (sp + i) % (OW * OH); \
        const int oh = ohw / OW; \
        const int ow = (ohw % OW); \
        const int id = SD * od - PD; \
        const int ih = SH * oh - PH; \
        const int iw = SW * ow - PW; \
        bool pad = ((PW > 0 || PH > 0 || PD > 0) \
                && (iw < 0 || ih < 0 || id < 0 || iw >= IW || ih >= IH \
                        || id >= ID)); \
        int off = id * IH * IW + ih * IW + iw; \
        data = pad ? 0 \
                   : AS_SRC_MMAD_DATA_T(intel_sub_group_block_read( \
                           (global uint *)&src[off * SRC_SP_STRIDE])); \
    } while (0);

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
xe_lp_1x1_conv_fwd_x8s8x(const __global SRC_DATA_T *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc,
        const __global int *src_compensation,
        const __global int *dst_compensation) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint mb_group_id = get_group_id(2);
    const uint ic_group_id = oc_group_id / OC_NCHUNK * IC_NCHUNK;

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

// Spatial
#if USE_SP_BLOCK
    const uint sp = get_global_id(1) * SP_BLOCK;
    const int sp_local_id = get_local_id(1);

    const uint od = sp / (OH * OW);
    const uint ohw = sp % (OH * OW);
    const uint oh = ohw / OW;
    const uint ow = (ohw % OW);
#else
    const uint sp = get_global_id(1);
    const int sp_local_id = get_local_id(1);

    const uint od = sp / (OWB * OH);
    const uint ohw = sp % (OWB * OH);
    const uint oh = ohw / OWB;
    const uint ow = (ohw % OWB) * SP_BLOCK;
#endif

    const uint id = SD * od;
    const uint ih = SH * oh;
    const uint iw = SW * ow;

#if MB_BLOCK == 32
    src += (mb_group_id % 2) * MB_BLOCK / 2 * SRC_MB_STRIDE; // MB block offset
    src += (mb_group_id / 2) * SRC_ICB_STRIDE * IC_NCHUNK * G; // MB offset
#else
    src += mb_group_id * SRC_ICB_STRIDE * IC_NCHUNK * G; // MB offset
#endif
    src += ic_group_id * SRC_ICB_STRIDE; // IC offset
#if !HAS_PADDING
    src += (id * IH * IW + ih * IW + iw) * SRC_SP_STRIDE; // SP offset
#endif
    // Destination
#if MB_BLOCK == 32
    dst += (mb_group_id % 2) * MB_BLOCK / 2 * DST_MB_STRIDE; // MB block offset
    dst += (mb_group_id / 2) * DST_OCB_STRIDE * OC_NCHUNK * G; // MB offset
#else
    dst += mb_group_id * DST_OCB_STRIDE * OC_NCHUNK * G; // MB offset
#endif
    dst += oc_group_id * DST_OCB_STRIDE; // OC offset
    dst += (od * OH * OW + oh * OW + ow) * SRC_SP_STRIDE; // SP offset

    // Weights
    wei += oc_group_id * WEI_BLOCK_STRIDE * IC_NCHUNK;
    // Output accumulators:

    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK1 C10 = 0, C11 = 0, C12 = 0, C13 = 0;

#if INT8_WEI_SLM
#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const __global char *wei_copy_from \
            = wei + sp_local_id * WEI_BLOCK_STRIDE / LWS_1; \
    __local char *wei_copy_to \
            = wei_slm + sp_local_id * WEI_BLOCK_STRIDE / LWS_1; \
    block_write4((__local uint *)wei_copy_to, \
            intel_sub_group_block_read4((__global uint *)wei_copy_from)); \
    __local char *wei_tmp = wei_slm; \
    barrier(CLK_LOCAL_MEM_FENCE);

    __local char wei_slm[WEI_BLOCK_STRIDE];
#endif // INT8_WEI_SLM

    for (uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id) {
        // Do not calculate if mb block is all zero padded
        if (MB % MB_BLOCK != 0 && MB_BLOCK == 32
                && mb_group_id * MB_BLOCK / 2 > MB) {
            break;
        }

#if INT8_WEI_SLM
        READ_SLM()
#if SP_TAIL
        if (ow < OW)
#endif // SP_TAIL
#endif // INT8_WEI_SLM
        {

            SRC_DATA_BLOCK_T S0 = 0;
            SRC_DATA_BLOCK_T1 S1 = 0;
#if OUT_SP_TAIL
#if USE_SP_BLOCK
            if (od * OH * OW + oh * OW + ow + SP_BLOCK > DST_SP) {
#else
            if (ow + SP_BLOCK > OW) {
#endif
#if OUT_SP_TAIL < 8
                for (int _i = 0; _i < OUT_SP_TAIL; ++_i) {
                    if (HAS_PADDING) {
                        PAD_BLOCK_READ(S0[_i], src, sp, _i);
                    } else {
                        S0[_i] = AS_SRC_MMAD_DATA_T(intel_sub_group_block_read(
                                (__global uint *)&src[_i * SW * IC_BLOCK]));
                    }
                }
#else
                BLOCK_READ_SRC(S0, 0 * IC_BLOCK, 0);
                for (int _i = 8; _i < OUT_SP_TAIL; ++_i) {
                    if (HAS_PADDING) {
                        PAD_BLOCK_READ(S1[_i - 8], src, sp, _i)
                    } else {
                        S1[_i - 8] = AS_SRC_MMAD_DATA_T(
                                intel_sub_group_block_read((__global uint
                                                *)&src[_i * SW * IC_BLOCK]));
                    }
                }
#endif
            } else
#endif // OUT_SP_TAIL

            {

                BLOCK_READ_SRC(S0, 0 * IC_BLOCK, 0);

#if (MB_BLOCK == 32 && MB > 8)
                BLOCK_READ_SRC1(S1, 8 * IC_BLOCK, 0);
#elif SP_BLOCK > 8
                BLOCK_READ_SRC1(S1, 8 * SW * IC_BLOCK, 8);
#endif
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

                C00 = MMAD_TAIL0(S0, W0, C00);
                if (OC > 8) C01 = MMAD_TAIL0(S0, W1, C01);
                if (OC > 16) C02 = MMAD_TAIL0(S0, W2, C02);
                if (OC > 24) C03 = MMAD_TAIL0(S0, W3, C03);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
                C10 = MMAD_TAIL1(S1, W0, C10);
                if (OC > 8) C11 = MMAD_TAIL1(S1, W1, C11);
                if (OC > 16) C12 = MMAD_TAIL1(S1, W2, C12);
                if (OC > 24) C13 = MMAD_TAIL1(S1, W3, C13);
#endif // (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
            } else
#endif // IC % IC_BLOCK != 0
            {
                BLOCK_READ_WHT_8x32(W0, 0);
                if (OC > 8) BLOCK_READ_WHT_8x32(W1, 8 * IC_BLOCK);
                if (OC > 16) BLOCK_READ_WHT_8x32(W2, 16 * IC_BLOCK);
                if (OC > 24) BLOCK_READ_WHT_8x32(W3, 24 * IC_BLOCK);

                C00 = MMAD_FULL0(S0, W0, C00);
                if (OC > 8) C01 = MMAD_FULL0(S0, W1, C01);
                if (OC > 16) C02 = MMAD_FULL0(S0, W2, C02);
                if (OC > 24) C03 = MMAD_FULL0(S0, W3, C03);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
                C10 = MMAD_FULL1(S1, W0, C10);
                if (OC > 8) C11 = MMAD_FULL1(S1, W1, C11);
                if (OC > 16) C12 = MMAD_FULL1(S1, W2, C12);
                if (OC > 24) C13 = MMAD_FULL1(S1, W3, C13);
#endif // (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
            }
        }

        src += SRC_ICB_STRIDE;
        wei += WEI_BLOCK_STRIDE;
    }

#if WITH_SRC_ZPOINTS
    int4 src_comp = as_int4(intel_sub_group_block_read4(
            (__global uint *)(&src_compensation[oc_group_id * OC_BLOCK])));

    C00 -= src_comp.s0;
    C01 -= src_comp.s1;
    C02 -= src_comp.s2;
    C03 -= src_comp.s3;
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
    C10 -= src_comp.s0;
    C11 -= src_comp.s1;
    C12 -= src_comp.s2;
    C13 -= src_comp.s3;
#endif // (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
#endif // WITH_SRC_ZPOINTS

    float4 tmp;
    DST_DATA4_T dst_pack[8];
    DST_DATA4_T D0[BLOCK0] = {0};
    DST_DATA4_T D1[BLOCK1] = {0};

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, oc_group_id * OC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, oc_group_id * OC_BLOCK);
#define QUANTIZE_ADD_BIAS() tmp = SCALE * fma(tmp, (float4)1, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#if WITH_SUM
#if USE_SP_BLOCK
    if (OUT_SP_TAIL && od * OH * OW + oh * OW + ow + SP_BLOCK > DST_SP) {
#else
    if (OUT_SP_TAIL && ow + SP_BLOCK > OW) {
#endif
#if OUT_SP_TAIL < 8
        block_read_dst(OUT_SP_TAIL, D0, dst);
#else
        block_read_dst(BLOCK0, D0, dst);
        block_read_dst(OUT_SP_TAIL - 8, D1, dst + 8 * OC_BLOCK);
#endif
    } else {
        block_read_dst(BLOCK0, D0, dst);
        if (SP_BLOCK > 8 || (MB_BLOCK == 32 && MB > 8)) {
            block_read_dst(BLOCK1, D1, dst + 8 * OC_BLOCK);
        }
    }
#endif // with_sum

#if WITH_DST_ZPOINTS
    int4 dst_zp = read_dst_zero_points_32c(
            dst_compensation, oc_group_id * OC_BLOCK);
#define ADD_DST_COMPENSATION() tmp += convert_float4(dst_zp);
#else
#define ADD_DST_COMPENSATION()
#endif // WITH_DST_ZPOINTS

#if WITH_SRC_ZPOINTS
#define ZERO_PAD_DST() tmp = zero_pad_dst_32c(tmp, oc_group_id * OC_BLOCK);
#else
#define ZERO_PAD_DST()
#endif // WITH_SRC_ZPOINTS

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define CONVERT_PACK(idx) \
    do { \
        dst_pack[idx] = CONVERT_DST_DATA4_T(tmp); \
    } while (0)

#define STORE_DST(n, C0, C1, C2, C3, D, dst_ptr, mb_stride) \
    do { \
        for (int n_i = 0; n_i < n; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            int po_mb; \
            if (MB_BLOCK == 32) \
                po_mb = (mb_group_id * MB_BLOCK / 2 + mb_stride * 8 + n_i); \
            else \
                po_mb = mb_group_id % MB; \
            if (MB % MB_BLOCK == 0 || po_mb < MB) { \
                QUANTIZE_ADD_BIAS(); \
                const int po_oc = (oc_group_id * OC_BLOCK) % (OC * G); \
                float4 dni = convert_float4(AS_SUM_DATA4_T(D[n_i])); \
                APPLY_POST_OPS_TRY_BURST(tmp, float, dni, float, po_mb, 1, \
                        po_oc, 4 * SUB_GROUP_SIZE, sg_local_id); \
                ADD_DST_COMPENSATION(); \
                ZERO_PAD_DST(); \
            } \
            CONVERT_PACK(n_i); \
        } \
        block_write_dst(n, dst_pack, dst_ptr); \
    } while (0)

#if INT8_WEI_SLM && SP_TAIL
    if (ow < OW)
#endif
    {
#if USE_SP_BLOCK
        if (OUT_SP_TAIL && od * OH * OW + oh * OW + ow + SP_BLOCK > DST_SP) {
#else
        if (OUT_SP_TAIL && ow + SP_BLOCK > OW) {
#endif
            STORE_DST(min(BLOCK0, OUT_SP_TAIL), C00, C01, C02, C03, D0, dst, 0);
            STORE_DST(OUT_SP_TAIL - 8, C10, C11, C12, C13, D1,
                    dst + 8 * OC_BLOCK, 1);
        } else {
            STORE_DST(BLOCK0, C00, C01, C02, C03, D0, dst, 0);
            if (SP_BLOCK > 8 || MB_BLOCK == 32)
                STORE_DST(
                        BLOCK1, C10, C11, C12, C13, D1, dst + 8 * OC_BLOCK, 1);
        }
    }
}

// Reads (n * 4) elements per work-item.
void block_read_dst(int n, DST_DATA_T *d, const __global DST_DATA_T *dst) {
    int nelems = n * 4;
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < nelems / 16 * 16; i += 16) {
        *((DST_DATA16_T *)&d[i]) = BLOCK_READ_DST16(dst + i * 8);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 16 * 16; i < nelems / 8 * 8; i += 8) {
        *((DST_DATA8_T *)&d[i]) = BLOCK_READ_DST8(dst + i * 8);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 8 * 8; i < nelems; i += 4) {
        *((DST_DATA4_T *)&d[i]) = BLOCK_READ_DST4(dst + i * 8);
    }
}

// Writes (n * 4) elements per work-item.
void block_write_dst(int n, const DST_DATA_T *d, __global DST_DATA_T *dst) {
    int nelems = n * 4;
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < nelems / 16 * 16; i += 16) {
        BLOCK_WRITE_DST16(dst + i * 8, *((DST_DATA16_T *)&d[i]));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 16 * 16; i < nelems / 8 * 8; i += 8) {
        BLOCK_WRITE_DST8(dst + i * 8, *((DST_DATA8_T *)&d[i]));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = nelems / 8 * 8; i < nelems; i += 4) {
        BLOCK_WRITE_DST4(dst + i * 8, *((DST_DATA4_T *)&d[i]));
    }
}
