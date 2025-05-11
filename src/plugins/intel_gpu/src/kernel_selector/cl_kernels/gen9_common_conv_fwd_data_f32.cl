// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

#define WITH_ELTWISE 1

#if WITH_ELTWISE == 1
//#include "ocl_post_ops.h"    // Use CLDNN activation
#endif

#define ODHW_SIZE (OD_FULL * OH_FULL * OW_FULL)
#define IDHW_SIZE (ID_FULL * IH_FULL * IW_FULL)
#define KDHW_SIZE (KD * KH * KW)

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

// Use CLDNN activation
#define DO_ELTWISE(blockC, nelems, alpha, beta) \
    do { \
        for (uint i = 0; i < nelems; i++) \
            blockC[i] = ACTIVATION(blockC[i], ACTIVATION_PARAMS); \
    } while (0)

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if SUB_GROUP_SIZE != 1
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE) // attr:no-format
#endif
KERNEL(gen9_common_conv_fwd_f32_kernel)(
        const __global float *src,
        __global float *dst,
#if USE_IMAGE == 1
        __read_only image2d_t wei
#else
        const __global float *wei
#endif
#if WITH_BIAS
        , const __global float *bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{

const float eltwise_alpha = 0;
const float eltwise_beta = 0;
const float sum_scale = 1;

    const int output_offset = (OUTPUT_PAD_BEFORE_FEATURE_NUM / OC_BLOCK) * OD_FULL * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Z) * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Y) * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_X) * OC_BLOCK * MB_BLOCK;
    const int input_offset = (INPUT0_PAD_BEFORE_FEATURE_NUM / IC_BLOCK) * ID_FULL * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Z) * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Y) * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_X) * IC_BLOCK * MB_BLOCK;

#ifdef VER_16MB16C
#if GROUPED
    const int g = (int)get_group_id(0) / OC;
    const int oc = (int)get_group_id(0) % OC;
#else
    const int g = 0;
    const int oc = get_group_id(0);
#endif
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    int mb = get_group_id(2) * MB_BLOCK;

    const int goc = oc;

#if CASE_3D
    const int od = sp / (OW * OH);
    const int ohw = sp % (OW * OH);
    const int id = od * SD - PD;
#else
    const int od = 0;
    const int id = 0;
    const int ohw = sp;
#endif
    const int oh = ohw / OW;
    const int ow = ohw % OW;

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;

    __global float *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK * MB_BLOCK
            + g * OC * ODHW_SIZE * MB_BLOCK + oh * OW_FULL * OC_BLOCK * MB_BLOCK
            + ow * OC_BLOCK * MB_BLOCK + od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK;

    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_BLOCK * MB_BLOCK
            + ih * IW_FULL * IC_BLOCK * MB_BLOCK + g * IDHW_SIZE * IC * MB_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK;

#if USE_IMAGE == 1
    int2 coordB0 = (int2)((oc * OC_BLOCK) * sizeof(uint), 0);
    int2 coordB1 = (int2)((oc * OC_BLOCK) * sizeof(uint), 8);
#else
    wei += goc * KDHW_SIZE * OC_BLOCK * IC_BLOCK + g * FILTER_GROUPS_PITCH;
#endif

#if WITH_BIAS
    float8 blockC00 = bias[oc * OC_BLOCK + local_id + g * OC];
    float8 blockC01 = bias[oc * OC_BLOCK + local_id + g * OC];
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1) || (HAS_PAD_W && KW == 1))
    if (!(id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW)) {
#endif
#if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
                            || iw + kw * (1 + DW) < 0
                            || iw + kw * (1 + DW) >= IW
#if CASE_3D
                            || id + kd * (1 + DD) < 0
                            || id + kd * (1 + DD) >= ID) {
#else
                    ) {
#endif
#if USE_IMAGE == 1
                        coordB0.y += IC;
                        coordB1.y += IC;
#endif
                        continue;
                    }

                    const __global float *src1 = src
                            + kd * (1 + DD) * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK
                            + kh * (1 + DH) * IW_FULL * IC_BLOCK * MB_BLOCK
                            + kw * (1 + DW) * IC_BLOCK * MB_BLOCK;
                    const __global float *wei1 = wei
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;
#else
    const __global float *src1 = src;
    const __global float *wei1 = wei;
#endif
                    int icb = 0;
                    do {
#define TRANSPOSE_8(_block, _col) \
    (float8)(_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7), _result); \
        _result = FMA8(_blockB1.s0, TRANSPOSE_8(_blockA, 8), _result); \
        _result = FMA8(_blockB1.s1, TRANSPOSE_8(_blockA, 9), _result); \
        _result = FMA8(_blockB1.s2, TRANSPOSE_8(_blockA, 10), _result); \
        _result = FMA8(_blockB1.s3, TRANSPOSE_8(_blockA, 11), _result); \
        _result = FMA8(_blockB1.s4, TRANSPOSE_8(_blockA, 12), _result); \
        _result = FMA8(_blockB1.s5, TRANSPOSE_8(_blockA, 13), _result); \
        _result = FMA8(_blockB1.s6, TRANSPOSE_8(_blockA, 14), _result); \
        _result = FMA8(_blockB1.s7, TRANSPOSE_8(_blockA, 15), _result); \
    }

#if USE_IMAGE == 1
                        float8 blockB00 = as_float8(
                                intel_sub_group_block_read8(wei, coordB0));
                        float8 blockB01 = as_float8(
                                intel_sub_group_block_read8(wei, coordB1));
#else
        float8 blockB00 = as_float8(
                _sub_group_block_read8((const __global uint *)wei1));
        float8 blockB01 = as_float8(_sub_group_block_read8(
                (const __global uint *)(wei1 + 8 * IC_BLOCK)));
#endif
                        float8 blockA;

                        blockA = as_float8(_sub_group_block_read8(
                                (const __global uint *)(src1)));

                        MULTIPLY_BLOCKS_8x8(
                                blockC00, blockA, blockB00, blockB01);

                        blockA = as_float8(_sub_group_block_read8(
                                (const __global uint *)(src1 + 8 * IC_BLOCK)));

                        MULTIPLY_BLOCKS_8x8(
                                blockC01, blockA, blockB00, blockB01);

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
                        src1 += IC_BLOCK * IDHW_SIZE * MB_BLOCK;
#if USE_IMAGE == 1
                        coordB0.y += IC_BLOCK;
                        coordB1.y += IC_BLOCK;
#else
        wei1 += OC * KDHW_SIZE * IC_BLOCK;
#endif
                        icb += IC_BLOCK;
                    } while (icb < IC);
#if KH != 1 || KW != 1 || KD != 1
                }
#endif
#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1) || (HAS_PAD_W && KW == 1))
    }
#endif

#if WITH_SUM == 1
    float8 blockS00 = as_float8(
            _sub_group_block_read8((const __global uint *)dst_write0));
    float8 blockS01 = as_float8(_sub_group_block_read8(
            (const __global uint *)(dst_write0 + 8 * OC_BLOCK)));

#if SUM_SCALE == 1
    blockC00 += blockS00;
    blockC01 += blockS01;
#else
    blockC00 = fma(blockS00, (float8)sum_scale, blockC00);
    blockC01 = fma(blockS01, (float8)sum_scale, blockC01);
#endif
#endif // with_sum
#if WITH_ELTWISE == 1
    DO_ELTWISE(blockC00, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(blockC01, 8, eltwise_alpha, eltwise_beta);
#endif

#if OUTPUT_LEFTOVERS
    if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
        for (int i = 0; i < 8; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
            { FUSED_OPS_SCALAR1; blockC01[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED) {
                dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
                dst_write0[8 * OC_BLOCK + i * OC_BLOCK + local_id] = blockC01[i];
            }
        }
    } else

#endif // OUTPUT_LEFTOVERS
    {

#if HAS_FUSED_OPS
    { FUSED_OPS_VEC0; blockC00 = FUSED_OPS_RESULT_VEC0; }
    { FUSED_OPS_VEC1; blockC01 = FUSED_OPS_RESULT_VEC1; }
#endif
     _sub_group_block_write8(
             (__global unsigned int *)(&dst_write0[0]), as_uint8(blockC00));
     _sub_group_block_write8(
             (__global unsigned int *)(&dst_write0[8 * OC_BLOCK]),
             as_uint8(blockC01));
    }
#endif // ver_16mb16c

#ifdef VER_8OW16C
#if IC == 3
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    const int ocb_mb = get_group_id(2);
    const int ocb = ocb_mb / (MB);
    const int mb = ocb_mb % (MB);
    const int oc = (ocb * OCB) / OC_BLOCK + get_group_id(0);

#if CASE_3D
    const int od = sp / (OWB * OHB);
    const int ohw = sp % (OWB * OHB);
    const int id = od * SD - PD;
#else
    const int od = 0;
    const int id = 0;
    const int ohw = sp;
#endif
    const int oh = (ohw / OWB) * OH_BLOCK;
    const int ow = (ohw % OWB) * OW_BLOCK;

#if WITH_BIAS
    float8 blockC00 = bias[oc * OC_BLOCK + local_id];
#if OCB == 32
    float8 blockC01 = bias[oc * OC_BLOCK + local_id + 16];
#endif
#else
#if OW_BLOCK != 8
    float blockC00[OW_BLOCK] = {0.0f};
#if OCB == 32
    float blockC01[OW_BLOCK] = {0.0f};
#endif
#else
    float8 blockC00 = 0.0f;
#if OCB == 32
    float8 blockC01 = 0.0f;
#endif
#endif
#endif

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;
#if NHWC == 1
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_FULL + ih * IW_FULL * IC_FULL + id * IH_FULL * IW_FULL * IC_FULL;
#else
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw + ih * IW_FULL + id * IH_FULL * IW_FULL;
#endif

    wei += oc * OC_BLOCK * IC * KDHW_SIZE;

    for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh) {

            if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
#if CASE_3D
                    || id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
#else
            ) {
#endif
                continue;
            }
#if NHWC == 1
            const __global float *src1 = src + kd * (1 + DD) * IH * IW_FULL * IC_FULL
                    + kh * (1 + DH) * IW_FULL * IC_FULL + local_id;
#define SP_OFF IC
#else
            const __global float *src1 = src + kd * (1 + DD) * IH_FULL * IW_FULL
                    + kh * (1 + DH) * IW_FULL + local_id * IDHW_SIZE;
#define SP_OFF 1
#endif

            float tempA[SW * OW_BLOCK + KW * (1 + DW)];
            int k = iw;
            if (local_id < 3) {
#if OW % OW_BLOCK != 0 || HAS_PAD_W
                if (k < 0 || k + SW * OW_BLOCK + KW * (1 + DW) >= IW) {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        if (k >= 0 && k < IW)
                            tempA[i] = src1[i * SP_OFF];
                        else
                            tempA[i] = 0.0f;
                        k++;
                    }
                } else {
#endif
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        tempA[i] = src1[i * SP_OFF];
                    }
#if OW % OW_BLOCK != 0 || HAS_PAD_W
                }
#endif
            }
            __attribute__((opencl_unroll_hint(KW))) // attr:no-format
            for (int kw = 0; kw < KW; ++kw) {

                const __global float *wei1 = wei + kd * KH * KW * OC_BLOCK * IC
                        + kh * KW * OC_BLOCK * IC + kw * OC_BLOCK * IC;

#define TRANSPOSE_1(_block, _col) (float)(_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((float)(a), (float)b, (float)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB0, _blockB1, _blockB2) \
    { \
        _result = FMA8(_blockB0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB2, TRANSPOSE_1(_blockA, 2), _result); \
    }

                float blockB00 = as_float(_sub_group_block_read(
                        (const __global uint *)wei1));
                float blockB01 = as_float(_sub_group_block_read(
                        (const __global uint *)(wei1 + OC_BLOCK)));
                float blockB02 = as_float(_sub_group_block_read(
                        (const __global uint *)(wei1 + 2 * OC_BLOCK)));

                float blockA[OW_BLOCK] = {0.0f};
                if (local_id < 3) {
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = tempA[kw * (1 + DW) + i * SW];
                    }
                }
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(blockC00[i], blockA[i], blockB00,
                            blockB01, blockB02);
                }
#if OCB == 32
                wei1 += KD * KH * KW * IC * OC_BLOCK;
                blockB00 = as_float(_sub_group_block_read(
                        (const __global uint *)wei1));
                blockB01 = as_float(_sub_group_block_read(
                        (const __global uint *)(wei1 + OC_BLOCK)));
                blockB02 = as_float(_sub_group_block_read(
                        (const __global uint *)(wei1 + 2 * OC_BLOCK)));

                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(blockC01[i], blockA[i], blockB00,
                            blockB01, blockB02);
                }
#endif

#undef TRANSPOSE_BLOCK_1
#undef MULTIPLY_BLOCKS_8x8
            }
        }
    __global float *dst_write0 = dst + output_offset
            + (mb / MB_BLOCK) * OC_FULL * ODHW_SIZE * MB_BLOCK
            + oc * OC_BLOCK * MB_BLOCK * ODHW_SIZE
            + od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK + oh * OW_FULL * OC_BLOCK * MB_BLOCK
            + ow * OC_BLOCK * MB_BLOCK + (mb % MB_BLOCK) * OC_BLOCK;
#if OCB == 32
    __global float *dst_write1 = dst_write0 + OC_BLOCK * MB_BLOCK * ODHW_SIZE;
#endif
#if WITH_SUM == 1
    float8 blockS00, blockS01;
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            blockS00[i] = as_float(_sub_group_block_read((const __global
                            uint *)&dst_write0[i * OC_BLOCK * MB_BLOCK]));
#if OCB == 32
            blockS01[i] = as_float(_sub_group_block_read((const __global
                            uint *)&dst_write1[i * OC_BLOCK * MB_BLOCK]));
#endif
        }
    } else {
        for (int i = 0; i < OW_BLOCK; i++) {
            blockS00[i] = as_float(_sub_group_block_read((const __global
                            uint *)&dst_write0[i * OC_BLOCK * MB_BLOCK]));
#if OCB == 32
            blockS01[i] = as_float(_sub_group_block_read((const __global
                            uint *)&dst_write1[i * OC_BLOCK * MB_BLOCK]));
#endif
        }
    }
    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        blockC00[i] += blockS00[i];
#if OCB == 32
        blockC01[i] += blockS01[i];
#endif
#else
        blockC00[i] = fma(blockS00[i], (float)sum_scale, blockC00[i]);
#if OCB == 32
        blockC01[i] = fma(blockS01[i], (float)sum_scale, blockC01[i]);
#endif
#endif
    }
#endif
#if WITH_ELTWISE == 1
    DO_ELTWISE(blockC00, OW_BLOCK, eltwise_alpha, eltwise_beta);
#if OCB == 32
    DO_ELTWISE(blockC01, OW_BLOCK, eltwise_alpha, eltwise_beta);
#endif
#endif

#if OW % OW_BLOCK != 0
    if (ow + OW_BLOCK > OW) {
        for (int i = 0; i < OW - OW_LAST; i++) {
#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif
            _sub_group_block_write((__global unsigned int *)(&dst_write0[i
                                                * OC_BLOCK * MB_BLOCK]),
                    as_uint(blockC00[i]));
#if OCB == 32
#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR1; blockC01[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            _sub_group_block_write(
                    (__global unsigned int
                                    *)(&dst_write0[i * OC_BLOCK * MB_BLOCK
                            + OC_BLOCK * MB_BLOCK * ODHW_SIZE]),
                    as_uint(blockC01[i]));
#endif
        }
    } else {
#endif
#if OW_BLOCK != 8 || MB_BLOCK != 1
        __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
        for (int i = 0; i < OW_BLOCK; i++) {
#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif
            _sub_group_block_write((__global unsigned int *)(&dst_write0[i
                                                * OC_BLOCK * MB_BLOCK]),
                    as_uint(blockC00[i]));
#if OCB == 32
#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR1; blockC01[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            _sub_group_block_write(
                    (__global unsigned int
                                    *)(&dst_write0[i * OC_BLOCK * MB_BLOCK
                            + OC_BLOCK * MB_BLOCK * ODHW_SIZE]),
                    as_uint(blockC01[i]));
#endif
        }
#else
#if HAS_FUSED_OPS
    { FUSED_OPS_VEC0; blockC00 = FUSED_OPS_RESULT_VEC0; }
#endif
    _sub_group_block_write8(
            (__global unsigned int *)(&dst_write0[0]), as_uint8(blockC00));
#if OCB == 32
#if HAS_FUSED_OPS
    { FUSED_OPS_VEC1; blockC01 = FUSED_OPS_RESULT_VEC1; }
#endif
    _sub_group_block_write8((__global unsigned int *)(&dst_write0[OC_BLOCK
                                         * MB_BLOCK * ODHW_SIZE]),
            as_uint8(blockC01));
#endif
#endif
#if OW % OW_BLOCK != 0
    }
#endif

#else // IC == 3
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
#if GROUPED
    const int ocb_mb_g = get_group_id(2);
#if OC == 8
    const int g = ocb_mb_g / MB;
    const int ocb_mb = ocb_mb_g % MB;
#else
    const int g = ocb_mb_g / (MB * (OC / OCB));
    const int ocb_mb = ocb_mb_g % (MB * (OC / OCB));
#endif
#else
    const int ocb_mb = get_group_id(2);
    const int g = 0;
#endif
    const int ocb = ocb_mb / (MB);
    const int mb = ocb_mb % (MB);
    const int oc = (ocb * OCB) / OC_BLOCK + get_group_id(0);
    const int goc = oc;

#if CASE_3D
    const int od = sp / (OWB * OHB);
    const int ohw = sp % (OWB * OHB);
    const int id = od * SD - PD;
#else
    const int od = 0;
    const int id = 0;
    const int ohw = sp;
#endif
    const int oh = (ohw / OWB) * OH_BLOCK;
    const int ow = (ohw % OWB) * OW_BLOCK;

#if WITH_BIAS
#if OW_BLOCK != 8 && OW_BLOCK != 16
    float blockC00[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++)
        blockC00[i] = bias[oc * OC_BLOCK + local_id + g * OC];
#else
    float8 blockC00 = bias[oc * OC_BLOCK + local_id + g * OC];
#if OW_BLOCK == 16
    float8 blockC01 = blockC00;
#endif
#endif
#else
#if OW_BLOCK != 8 && OW_BLOCK != 16
    float blockC00[OW_BLOCK] = {0.0f};
#else
    float8 blockC00 = 0.0f;
#if OW_BLOCK == 16
    float8 blockC01 = 0.0f;
#endif
#endif
#endif

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;
#if IC == 8 && G != 1
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_BLOCK + ih * IW_FULL * IC_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK + (g / 2) * IDHW_SIZE * IC * 2;
#else
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_BLOCK + ih * IW_FULL * IC_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK + g * IDHW_SIZE * IC;
#endif
    wei += goc * KDHW_SIZE * OC_BLOCK * IC + g * FILTER_GROUPS_PITCH;

    const bool do_if = iw < 0 || iw + SW * OW_BLOCK + KW * (1 + DW) >= IW;

#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1))
    if (!(id < 0 || id >= ID || ih < 0 || ih >= IH)) {
#endif
        int icb = 0;
        do {
#if KH != 1 || KW != 1 || KD != 1
            __attribute__((opencl_unroll_hint(1))) // attr:no-format
            for (int kd = 0; kd < KD; ++kd)
                    __attribute__((opencl_unroll_hint(1))) // attr:no-format
                    for (int kh = 0; kh < KH; ++kh) {

                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
#if CASE_3D
                        || id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
#else
                ) {
#endif
                    continue;
                }
                const __global float *src1 = src
                        + kd * (1 + DD) * IH_FULL * IW_FULL * IC_BLOCK
                        + kh * (1 + DH) * IW_FULL * IC_BLOCK;

                float tempA[SW * OW_BLOCK + KW * (1 + DW)];
                int k = iw;
                if (do_if) {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        if (k >= 0 && k < IW)
                            tempA[i] = as_float(_sub_group_block_read(
                                    (const __global uint
                                                    *)(&src1[i * IC_BLOCK])));
                        else
                            tempA[i] = 0.0f;
                        k++;
                    }
                } else {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        tempA[i] = as_float(_sub_group_block_read(
                                (const __global uint *)(&src1[i * IC_BLOCK])));
                    }
                }
                __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                for (int kw = 0; kw < KW; ++kw) {

                    const __global float *wei1 = wei
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;

#else
        const __global float *src1 = src;
        const __global float *wei1 = wei;
#endif
#define TRANSPOSE_1(_block, _col) (float)(_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((float)(a), (float)b, (float)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_1(_blockA, 2), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_1(_blockA, 3), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_1(_blockA, 4), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_1(_blockA, 5), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_1(_blockA, 6), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_1(_blockA, 7), _result); \
        _result = FMA8(_blockB1.s0, TRANSPOSE_1(_blockA, 8), _result); \
        _result = FMA8(_blockB1.s1, TRANSPOSE_1(_blockA, 9), _result); \
        _result = FMA8(_blockB1.s2, TRANSPOSE_1(_blockA, 10), _result); \
        _result = FMA8(_blockB1.s3, TRANSPOSE_1(_blockA, 11), _result); \
        _result = FMA8(_blockB1.s4, TRANSPOSE_1(_blockA, 12), _result); \
        _result = FMA8(_blockB1.s5, TRANSPOSE_1(_blockA, 13), _result); \
        _result = FMA8(_blockB1.s6, TRANSPOSE_1(_blockA, 14), _result); \
        _result = FMA8(_blockB1.s7, TRANSPOSE_1(_blockA, 15), _result); \
    }

#define MULTIPLY_BLOCKS_FIRST_HALF(_result, _blockA, _blockB)         \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_1(_blockA, 2), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_1(_blockA, 3), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_1(_blockA, 4), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_1(_blockA, 5), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_1(_blockA, 6), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_1(_blockA, 7), _result); \
    }

#define MULTIPLY_BLOCKS_SECOND_HALF(_result, _blockA, _blockB)         \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_1(_blockA, 8), _result);  \
        _result = FMA8(_blockB.s1, TRANSPOSE_1(_blockA, 9), _result);  \
        _result = FMA8(_blockB.s2, TRANSPOSE_1(_blockA, 10), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_1(_blockA, 11), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_1(_blockA, 12), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_1(_blockA, 13), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_1(_blockA, 14), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_1(_blockA, 15), _result); \
    }

#if IC == 8 && G != 1
                    float8 blockB00 = as_float8(_sub_group_block_read8(
                            (const __global uint *)wei1));
#else
                    float8 blockB00 = as_float8(_sub_group_block_read8(
                            (const __global uint *)wei1));
                    float8 blockB01 = as_float8(_sub_group_block_read8(
                            (const __global uint *)(wei1 + 8 * IC_BLOCK)));
#endif

#if KH != 1 || KW != 1 || KD != 1
                    float blockA[OW_BLOCK];
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = tempA[kw * (1 + DW) + SW * i];
                    }
#else
#if OW_BLOCK != 8 || HAS_PAD_W
        float blockA[OW_BLOCK];
#else
        float8 blockA;
#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        if (ow == OW_LAST) {
            for (int i = 0; i < OW - OW_LAST; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA[i] = 0.0f;
                } else {
#endif
                    blockA[i] = as_float(_sub_group_block_read(
                            (const __global uint *)(&src1[i * IC_BLOCK * SW])));
#if HAS_PAD_W
                }
#endif
            }
            for (int i = OW - OW_LAST; i < OW_BLOCK; i++)
                blockA[i] = 0.0f;
        } else {
#endif
#if SW != 1 || OW_BLOCK != 8 || HAS_PAD_W
            __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
            for (int i = 0; i < OW_BLOCK; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA[i] = 0.0f;
                } else {
#endif
                    blockA[i] = as_float(_sub_group_block_read(
                            (const __global uint *)(&src1[i * IC_BLOCK * SW])));
#if HAS_PAD_W
                }
#endif
            }
#else
        blockA = as_float8(
                _sub_group_block_read8((const __global uint *)(&src1[0])));
#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        }
#endif
#endif
#if OW_BLOCK != 16
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
#if IC == 8 && G != 1
                        if (g % 2 == 0) {
                            MULTIPLY_BLOCKS_FIRST_HALF(blockC00[i], blockA[i], blockB00);
                        } else {
                            MULTIPLY_BLOCKS_SECOND_HALF(blockC00[i], blockA[i], blockB00);
                        }
#else
                        MULTIPLY_BLOCKS_8x8(
                                blockC00[i], blockA[i], blockB00, blockB01);
#endif
                    }
#else
        __attribute__((opencl_unroll_hint(8))) // attr:no-format
        for (int i = 0; i < 8; i++) {
#if IC == 8 && G != 1
            if (g % 2 == 0) {
                MULTIPLY_BLOCKS_FIRST_HALF(blockC00[i], blockA[i], blockB00);
                MULTIPLY_BLOCKS_FIRST_HALF(blockC01[i], blockA[i + 8], blockB00);
            } else {
                MULTIPLY_BLOCKS_SECOND_HALF(blockC00[i], blockA[i], blockB00);
                MULTIPLY_BLOCKS_SECOND_HALF(blockC01[i], blockA[i + 8], blockB00);
            }
#else
            MULTIPLY_BLOCKS_8x8(blockC00[i], blockA[i], blockB00, blockB01);
            MULTIPLY_BLOCKS_8x8(blockC01[i], blockA[i + 8], blockB00, blockB01);
#endif
        }
#endif

#undef TRANSPOSE_1
#undef MULTIPLY_BLOCKS_8x8
#undef MULTIPLY_BLOCKS_FIRST_HALF
#undef MULTIPLY_BLOCKS_SECOND_HALF
#if KH != 1 || KW != 1 || KD != 1
                }
            }
#endif
            src += IC_BLOCK * IDHW_SIZE;
            wei += OC_BLOCK * KDHW_SIZE * IC_BLOCK;
            icb += IC_BLOCK;
        } while (icb < IC);
#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1))
    }
#endif
#if OC == 8 && G != 1
    __global float *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK + (g / 2) * OC_BLOCK * ODHW_SIZE + (g % 2) * OC
            + od * OH_FULL * OW_FULL * OC_BLOCK + oh * OW_FULL * OC_BLOCK + ow * OC_BLOCK;
#else
    __global float *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK + g * OC * ODHW_SIZE
            + od * OH_FULL * OW_FULL * OC_BLOCK + oh * OW_FULL * OC_BLOCK + ow * OC_BLOCK;
#endif

#if WITH_SUM == 1
#if OW_BLOCK != 8 && OW_BLOCK != 16
    float blockS00[OW_BLOCK];
#else
    float8 blockS00;
#if OW_BLOCK == 16
    float8 blockS01;
#endif
#endif
#if OW % OW_BLOCK != 0
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            blockS00[i] = as_float(_sub_group_block_read(
                    (const __global uint *)&dst_write0[i * OC_BLOCK]));
        }
    } else {
#endif
#if OW_BLOCK != 8 && OW_BLOCK != 16
        for (int i = 0; i < OW_BLOCK; i++) {
            blockS00[i] = as_float(_sub_group_block_read(
                    (const __global uint *)&dst_write0[i * OC_BLOCK]));
        }
#else
    blockS00 = as_float8(
            _sub_group_block_read8((const __global uint *)dst_write0));
#if OW_BLOCK == 16
    blockS01 = as_float8(_sub_group_block_read8(
            (const __global uint *)&dst_write0[8 * OC_BLOCK]));
#endif
#endif
#if OW % OW_BLOCK != 0
    }
#endif

#if OW_BLOCK != 16
    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        blockC00[i] += blockS00[i];
#else
        blockC00[i] = fma(blockS00[i], (float)sum_scale, blockC00[i]);
#endif
    }
#else
#if SUM_SCALE == 1
    blockC00 += blockS00;
    blockC01 += blockS01;
#else
    blockC00 = fma(blockS00, (float8)sum_scale, blockC00);
    blockC01 = fma(blockS01, (float8)sum_scale, blockC01);
#endif
#endif
#endif // with_sum
#if WITH_ELTWISE == 1
#if OW_BLOCK != 16
    DO_ELTWISE(blockC00, OW_BLOCK, eltwise_alpha, eltwise_beta);
#else
    DO_ELTWISE(blockC00, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(blockC01, 8, eltwise_alpha, eltwise_beta);
#endif
#endif

#if OW % OW_BLOCK != 0
    if (ow + OW_BLOCK > OW) {
        for (int i = 0; i < OW - OW_LAST; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif

#if OC == 8 && G != 1
            if (local_id < 8)
                dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
#else

#if OUTPUT_LEFTOVERS
            if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
                if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED)
                    dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
            } else
#endif
            {
                _sub_group_block_write(
                        (__global unsigned int *)(&dst_write0[i * OC_BLOCK]),
                        as_uint(blockC00[i]));
            }
#endif
        }
    } else {
#endif
#if OW_BLOCK != 8 && OW_BLOCK != 16
        __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
        for (int i = 0; i < OW_BLOCK; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif

#if OC == 8 && G != 1
            if (local_id < 8)
                dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
#else

#if OUTPUT_LEFTOVERS
            if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
                if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED)
                    dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
            } else
#endif
            {
                _sub_group_block_write(
                        (__global unsigned int *)(&dst_write0[i * OC_BLOCK]),
                        as_uint(blockC00[i]));
            }
#endif //  OC == 8 && G != 1
        }
#else

#if OC == 8 && G != 1
    for (int i = 0; i < 8; i++) {
#if HAS_FUSED_OPS
        { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif

        if (local_id < 8)
            dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
    }
#else

#if OUTPUT_LEFTOVERS
    if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
        for (int i = 0; i < 8; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; blockC00[i] = FUSED_OPS_RESULT_SCALAR0; }
#endif
            if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED)
                dst_write0[i * OC_BLOCK + local_id] = blockC00[i];
        }
    } else
#endif
    {

#if HAS_FUSED_OPS
    { FUSED_OPS_VEC0; blockC00 = FUSED_OPS_RESULT_VEC0; }
#endif
    _sub_group_block_write8(
            (__global unsigned int *)(&dst_write0[0]), as_uint8(blockC00));

    }
#endif //  OC == 8 && G != 1
#if OW_BLOCK == 16

#if OC == 8 && G != 1
    for (int i = 0; i < 8; i++) {
#if HAS_FUSED_OPS
        { FUSED_OPS_SCALAR1; blockC01[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif

        if (local_id < 8)
            dst_write0[(i + 8) * OC_BLOCK + local_id] = blockC01[i];
    }
#else
#if OUTPUT_LEFTOVERS
    if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
        for (int i = 0; i < 8; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR1; blockC01[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED)
                dst_write0[(i + 8) * OC_BLOCK + local_id] = blockC01[i];
        }
    } else
#endif
    {

#if HAS_FUSED_OPS
        { FUSED_OPS_VEC1; blockC01 = FUSED_OPS_RESULT_VEC1; }
#endif

        _sub_group_block_write8(
                (__global unsigned int *)(&dst_write0[8 * OC_BLOCK]),
                as_uint8(blockC01));

    }
#endif //  OC == 8 && G != 1
#endif
#endif
#if OW % OW_BLOCK != 0
    }
#endif

#endif // IC == 3
#endif
    return;
}
