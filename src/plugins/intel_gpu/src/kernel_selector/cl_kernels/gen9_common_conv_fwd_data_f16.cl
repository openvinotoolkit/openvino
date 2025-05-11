// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

#define WITH_ELTWISE 1

#if WITH_ELTWISE == 1
//#include "ocl_post_ops.h"    // Use CLDNN activation
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Use CLDNN activation
#define DO_ELTWISE(blockC, nelems, alpha, beta) \
    do { \
        for (uint i = 0; i < nelems; i++) \
            blockC[i] = ACTIVATION(blockC[i], ACTIVATION_PARAMS); \
    } while (0)

#define ODHW_SIZE (OD_FULL * OH_FULL * OW_FULL)
#define IDHW_SIZE (ID_FULL * IH_FULL * IW_FULL)
#define KDHW_SIZE (KD * KH * KW)

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if SUB_GROUP_SIZE != 1
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE) // attr:no-format
#endif
KERNEL(gen9_common_conv_fwd_f16_kernel)(
        const __global half *src,
        __global half *dst,
#if USE_IMAGE == 1
        __read_only image2d_t wei
#else
        const __global half *wei
#endif
#if WITH_BIAS
        , const __global half *bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const half eltwise_alpha = 0;
    const half eltwise_beta = 0;
    const half sum_scale_ = 1;

    half relu_negative_slope = eltwise_alpha;
    half sum_scale = sum_scale_;
    const int output_offset = (OUTPUT_PAD_BEFORE_FEATURE_NUM / OC_BLOCK) * OD_FULL * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Z) * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Y) * OW_FULL * OC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_X) * OC_BLOCK * MB_BLOCK;
    const int input_offset = (INPUT0_PAD_BEFORE_FEATURE_NUM / IC_BLOCK) * ID_FULL * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Z) * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Y) * IW_FULL * IC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_X) * IC_BLOCK * MB_BLOCK;

#if IC == 3 && OC % 32 == 0
#if MB % 2 == 0
    /* First convovution unrolled by MB2. */
    const int oc = get_group_id(0) * 2;
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    int mb = get_group_id(2) * 2;

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

#if OW_BLOCK == 8
#if WITH_BIAS
    half8 C00 = bias[oc * OC_BLOCK + local_id];
    half8 C01 = C00;
    half8 C10 = bias[(oc + 1) * OC_BLOCK + local_id];
    half8 C11 = C10;
#else
    half8 C00 = 0.0, C01 = 0.0;
    half8 C10 = 0.0, C11 = 0.0;
#endif
#else
#if WITH_BIAS
    half C00[OW_BLOCK];
    half C01[OW_BLOCK];
    half C10[OW_BLOCK];
    half C11[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++) {
        C00[i] = bias[oc * OC_BLOCK + local_id];
        C01[i] = bias[oc * OC_BLOCK + local_id];
        C10[i] = bias[(oc + 1) * OC_BLOCK + local_id];
        C11[i] = bias[(oc + 1) * OC_BLOCK + local_id];
    }
#else
    half C00[OW_BLOCK] = {0.0}, C01[OW_BLOCK] = {0.0};
    half C10[OW_BLOCK] = {0.0}, C11[OW_BLOCK] = {0.0};
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

#if CASE_3D
            if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
                continue;
            }
#endif
            if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH) {
                continue;
            }
#if NHWC == 1
            const __global half *src1 = src + kd * (1 + DD) * IH_FULL * IW_FULL * IC_FULL
                    + kh * (1 + DH) * IW_FULL * IC_FULL + local_id;
#define SP_OFF IC
#else
            const __global half *src1 = src + kd * (1 + DD) * IH_FULL * IW_FULL
                    + kh * (1 + DH) * IW_FULL + local_id * IDHW_SIZE;
            const __global half *src2 = src + kd * (1 + DD) * IH_FULL * IW_FULL
                    + kh * (1 + DH) * IW_FULL + local_id * IDHW_SIZE
                    + IC_FULL * IDHW_SIZE;
#define SP_OFF 1
#endif

            half tempA1[SW * OW_BLOCK + KW * (1 + DW)];
            half tempA2[SW * OW_BLOCK + KW * (1 + DW)];
            int k = iw;
            if (local_id < 3) {
                if (k < 0 || k + SW * OW_BLOCK + KW * (1 + DW) >= IW) {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        if (k >= 0 && k < IW) {
                            tempA1[i] = src1[i * SP_OFF];
                            tempA2[i] = src2[i * SP_OFF];
                        } else {
                            tempA1[i] = 0.0f;
                            tempA2[i] = 0.0f;
                        }
                        k++;
                    }
                } else {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW; i++) {
                        tempA1[i] = src1[i * SP_OFF];
                        tempA2[i] = src2[i * SP_OFF];
                    }
                }
            }
            __attribute__((opencl_unroll_hint(KW))) // attr:no-format
            for (int kw = 0; kw < KW; ++kw) {

                const __global half *wei1 = wei + kd * KH * KW * OC_BLOCK * IC
                        + kh * KW * OC_BLOCK * IC + kw * OC_BLOCK * IC;

#define TRANSPOSE_1(_block, _col) (half)(_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((half)(a), (half)b, (half)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB0, _blockB1, _blockB2) \
    { \
        _result = FMA8(_blockB0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB2, TRANSPOSE_1(_blockA, 2), _result); \
    }

                half blockB00 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)wei1));
                half blockB01 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1 + OC_BLOCK)));
                half blockB02 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1 + 2 * OC_BLOCK)));

                half blockA1[OW_BLOCK] = {0.0f};
                half blockA2[OW_BLOCK] = {0.0f};
                if (local_id < 3)
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA1[i] = tempA1[kw * (1 + DW) + i * SW];
                        blockA2[i] = tempA2[kw * (1 + DW) + i * SW];
                    }
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(
                            C00[i], blockA1[i], blockB00, blockB01, blockB02);
                    MULTIPLY_BLOCKS_8x8(
                            C01[i], blockA2[i], blockB00, blockB01, blockB02);
                }

                blockB00 = as_half(_sub_group_block_read_us((const __global
                                ushort *)&wei1[IC * KDHW_SIZE * OC_BLOCK]));
                blockB01 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1
                                + IC * KDHW_SIZE * OC_BLOCK + OC_BLOCK)));
                blockB02 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1
                                + IC * KDHW_SIZE * OC_BLOCK + 2 * OC_BLOCK)));

                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(
                            C10[i], blockA1[i], blockB00, blockB01, blockB02);
                    MULTIPLY_BLOCKS_8x8(
                            C11[i], blockA2[i], blockB00, blockB01, blockB02);
                }

#undef TRANSPOSE_BLOCK_1
#undef MULTIPLY_BLOCKS_8x8
            }
        }
    __global half *dst_write0 = dst + output_offset
            + (mb / MB_BLOCK) * OC_FULL * ODHW_SIZE * MB_BLOCK
            + oc * OC_BLOCK * MB_BLOCK * ODHW_SIZE
            + od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK + oh * OW_FULL * OC_BLOCK * MB_BLOCK
            + ow * OC_BLOCK * MB_BLOCK + (mb % MB_BLOCK) * OC_BLOCK;
    __global half *dst_write1 = dst + output_offset
            + ((mb + 1) / MB_BLOCK) * OC_FULL * ODHW_SIZE * MB_BLOCK
            + oc * OC_BLOCK * MB_BLOCK * ODHW_SIZE
            + od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK + oh * OW_FULL * OC_BLOCK * MB_BLOCK
            + ow * OC_BLOCK * MB_BLOCK + ((mb + 1) % MB_BLOCK) * OC_BLOCK;

#if WITH_SUM == 1
    half8 blockS00, blockS01, blockS10, blockS11;
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write0[i * OC_BLOCK * MB_BLOCK]));
            blockS10[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write0[OC_BLOCK * MB_BLOCK * ODHW_SIZE
                    + i * OC_BLOCK * MB_BLOCK]));
            blockS01[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write1[i * OC_BLOCK * MB_BLOCK]));
            blockS11[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write1[OC_BLOCK * MB_BLOCK * ODHW_SIZE
                    + i * OC_BLOCK * MB_BLOCK]));
        }
    } else {
        for (int i = 0; i < OW_BLOCK; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write0[i * OC_BLOCK * MB_BLOCK]));
            blockS10[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write0[OC_BLOCK * MB_BLOCK * ODHW_SIZE
                    + i * OC_BLOCK * MB_BLOCK]));
            blockS01[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write1[i * OC_BLOCK * MB_BLOCK]));
            blockS11[i] = as_half(_sub_group_block_read_us((const __global
                            ushort *)&dst_write1[OC_BLOCK * MB_BLOCK * ODHW_SIZE
                    + i * OC_BLOCK * MB_BLOCK]));
        }
    }
    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        C00[i] += blockS00[i];
        C10[i] += blockS10[i];
        C01[i] += blockS01[i];
        C11[i] += blockS11[i];
#else
        C00[i] = fma(blockS00[i], (half)sum_scale, C00[i]);
        C10[i] = fma(blockS10[i], (half)sum_scale, C10[i]);
        C01[i] = fma(blockS01[i], (half)sum_scale, C01[i]);
        C11[i] = fma(blockS11[i], (half)sum_scale, C11[i]);
#endif
    }
#endif // with_sum

#if WITH_ELTWISE == 1
    DO_ELTWISE(C00, OW_BLOCK, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C10, OW_BLOCK, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C01, OW_BLOCK, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C11, OW_BLOCK, eltwise_alpha, eltwise_beta);
#endif

    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C00[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[OC_BLOCK * MB_BLOCK
                                    * ODHW_SIZE
                            + i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C10[i]));

            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write1[i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C01[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write1[OC_BLOCK * MB_BLOCK
                                    * ODHW_SIZE
                            + i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C11[i]));
        }
    } else {
        for (int i = 0; i < OW_BLOCK; i++) {
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C00[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[OC_BLOCK * MB_BLOCK
                                    * ODHW_SIZE
                            + i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C10[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write1[i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C01[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write1[OC_BLOCK * MB_BLOCK
                                    * ODHW_SIZE
                            + i * OC_BLOCK * MB_BLOCK]),
                    as_ushort(C11[i]));
        }
    }

#else
    /* First convolution. */
    const int oc = get_group_id(0) * 2;
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    int mb = get_group_id(2);

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

#if OW_BLOCK == 8
#if WITH_BIAS
    half8 C00 = bias[oc * OC_BLOCK + local_id];
    half8 C10 = bias[(oc + 1) * OC_BLOCK + local_id];
#else
    half8 C00 = 0.0;
    half8 C10 = 0.0;
#endif
#else
#if WITH_BIAS
    half C00[OW_BLOCK];
    half C10[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++) {
        C00[i] = bias[oc * OC_BLOCK + local_id];
        C10[i] = bias[(oc + 1) * OC_BLOCK + local_id];
    }
#else
    half C00[OW_BLOCK] = {0.0};
    half C10[OW_BLOCK] = {0.0};
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

#if CASE_3D
            if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
                continue;
            }
#endif
            if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH) {
                continue;
            }
#if NHWC == 1
            const __global half *src1 = src + kd * (1 + DD) * IH_FULL * IW_FULL * IC_FULL
                    + kh * (1 + DH) * IW_FULL * IC_FULL + local_id;
#define SP_OFF IC
#else
            const __global half *src1 = src + kd * (1 + DD) * IH_FULL * IW_FULL
                    + kh * (1 + DH) * IW_FULL + local_id * IDHW_SIZE;
#define SP_OFF 1
#endif

            half tempA1[SW * OW_BLOCK + KW * (1 + DW)];
            int k = iw;
            if (local_id < 3) {
                if (k < 0 || k + SW * OW_BLOCK + KW * (1 + DW) >= IW) {
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        if (k >= 0 && k < IW) {
                            tempA1[i] = src1[i * SP_OFF];
                        } else {
                            tempA1[i] = 0.0f;
                        }
                        k++;
                    }
                } else {
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        tempA1[i] = src1[i * SP_OFF];
                    }
                }
            }

            for (int kw = 0; kw < KW; ++kw) {

                const __global half *wei1 = wei + kd * KH * KW * IC * OC_BLOCK
                        + kh * KW * OC_BLOCK * IC + kw * OC_BLOCK * IC;

#define TRANSPOSE_1(_block, _col) (half)(_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((half)(a), (half)b, (half)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB0, _blockB1, _blockB2) \
    { \
        _result = FMA8(_blockB0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB2, TRANSPOSE_1(_blockA, 2), _result); \
    }

                half blockB00 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)wei1));
                half blockB01 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1 + OC_BLOCK)));
                half blockB02 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1 + 2 * OC_BLOCK)));

                half8 blockA1 = 0.0f;
                if (local_id < 3)
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA1[i] = tempA1[kw * (1 + DW) + i * SW];
                    }
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(
                            C00[i], blockA1[i], blockB00, blockB01, blockB02);
                }

                blockB00 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1
                                + KDHW_SIZE * IC * OC_BLOCK)));
                blockB01 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1
                                + KDHW_SIZE * IC * OC_BLOCK + OC_BLOCK)));
                blockB02 = as_half(_sub_group_block_read_us(
                        (const __global ushort *)(wei1
                                + KDHW_SIZE * IC * OC_BLOCK + 2 * OC_BLOCK)));
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    MULTIPLY_BLOCKS_8x8(
                            C10[i], blockA1[i], blockB00, blockB01, blockB02);
                }

#undef TRANSPOSE_BLOCK_1
#undef MULTIPLY_BLOCKS_8x8
            }
        }
    __global half *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + oc * OC_BLOCK * ODHW_SIZE + od * OH_FULL * OW_FULL * OC_BLOCK
            + oh * OW_FULL * OC_BLOCK + ow * OC_BLOCK;

#if WITH_SUM == 1
    half8 blockS00, blockS10;
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[i * OC_BLOCK]));
            blockS10[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[ODHW_SIZE * OC_BLOCK
                            + i * OC_BLOCK]));
        }
    } else {
        for (int i = 0; i < OW_BLOCK; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[i * OC_BLOCK]));
            blockS10[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[ODHW_SIZE * OC_BLOCK
                            + i * OC_BLOCK]));
        }
    }
    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        C00[i] += blockS00[i];
        C10[i] += blockS10[i];
#else
        C00[i] = fma(blockS00[i], (half)sum_scale, C00[i]);
        C10[i] = fma(blockS10[i], (half)sum_scale, C10[i]);
#endif
    }
#endif
#if WITH_ELTWISE == 1
    DO_ELTWISE(C00, OW_BLOCK, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C10, OW_BLOCK, eltwise_alpha, eltwise_beta);
#endif

    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[i * OC_BLOCK]),
                    as_ushort(C00[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[ODHW_SIZE * OC_BLOCK
                            + i * OC_BLOCK]),
                    as_ushort(C10[i]));
        }
    } else {
        for (int i = 0; i < OW_BLOCK; i++) {
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[i * OC_BLOCK]),
                    as_ushort(C00[i]));
            _sub_group_block_write_us(
                    (__global ushort *)(&dst_write0[ODHW_SIZE * OC_BLOCK
                            + i * OC_BLOCK]),
                    as_ushort(C10[i]));
        }
    }

#endif
#endif
#if VER_16MB16C == 1 && MB % 32 == 0

    /*
  For now USE_32OC_UNROLL is always 0.
  TODO: Find a proper cross point for both cases
*/
#if OC % 32 == 0
#define USE_32OC_UNROLL 0
#else
#define USE_32OC_UNROLL 0
#endif

    /* Regular convolution unrolled by MB32. */
#if USE_32OC_UNROLL
    const int oc = get_group_id(0) * 2;
#else
#if GROUPED
    const int g = (int)get_group_id(0) / OC;
    const int oc = (int)get_group_id(0) % OC;
#else
    const int g = 0;
    const int oc = get_group_id(0);
#endif
#endif
    const int local_id = get_local_id(0);
    const int sp = get_group_id(1);
    int mb = get_group_id(2) * MB_BLOCK * 2;

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

#if WITH_BIAS
    half8 C00 = bias[oc * OC_BLOCK + local_id + g * OC];
    half8 C01 = C00, C02 = C00, C03 = C00;
#if USE_32OC_UNROLL
    half8 C10 = bias[(oc + 1) * OC_BLOCK + local_id];
    half8 C11 = C10, C12 = C10, C13 = C10;
#endif
#else
    half8 C00 = 0.0f, C01 = 0.0f, C02 = 0.0f, C03 = 0.0f;
#if USE_32OC_UNROLL
    half8 C10 = 0.0f, C11 = 0.0f, C12 = 0.0f, C13 = 0.0f;
#endif
#endif

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;
    src += input_offset + mb * IC_FULL * IDHW_SIZE + id * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK
            + ih * IW_FULL * IC_BLOCK * MB_BLOCK + iw * IC_BLOCK * MB_BLOCK
            + g * IC * IDHW_SIZE * MB_BLOCK;

    wei += goc * KDHW_SIZE * OC_BLOCK * FILTER_IFM_NUM + g * FILTER_GROUPS_PITCH;

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
                        continue;
                    }

                    const __global half *src1 = src
                            + kd * (1 + DD) * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK
                            + kh * (1 + DH) * IW_FULL * IC_BLOCK * MB_BLOCK
                            + kw * (1 + DW) * IC_BLOCK * MB_BLOCK;
                    const __global half *wei1 = wei
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;

#else
    const __global half *src1 = src;
    const __global half *wei1 = wei;
#endif
                    for (int icb = 0; icb < IC / IC_BLOCK; icb++) {

#define TRANSPOSE_8(_block, _col) \
    as_half8(_sub_group_shuffle(as_ushort8(_block), _col))

#define FMA8(a, b, c) fma((half8)(a), (half8)b, (half8)c)

#define MULTIPLY_BLOCKS_8x16(_result, _blockA, _blockB) \
    { \
        _result = FMA8(_blockB[0], TRANSPOSE_8(_blockA, 0), _result); \
        _result = FMA8(_blockB[1], TRANSPOSE_8(_blockA, 1), _result); \
        _result = FMA8(_blockB[2], TRANSPOSE_8(_blockA, 2), _result); \
        _result = FMA8(_blockB[3], TRANSPOSE_8(_blockA, 3), _result); \
        _result = FMA8(_blockB[4], TRANSPOSE_8(_blockA, 4), _result); \
        _result = FMA8(_blockB[5], TRANSPOSE_8(_blockA, 5), _result); \
        _result = FMA8(_blockB[6], TRANSPOSE_8(_blockA, 6), _result); \
        _result = FMA8(_blockB[7], TRANSPOSE_8(_blockA, 7), _result); \
        _result = FMA8(_blockB[8], TRANSPOSE_8(_blockA, 8), _result); \
        _result = FMA8(_blockB[9], TRANSPOSE_8(_blockA, 9), _result); \
        _result = FMA8(_blockB[10], TRANSPOSE_8(_blockA, 10), _result); \
        _result = FMA8(_blockB[11], TRANSPOSE_8(_blockA, 11), _result); \
        _result = FMA8(_blockB[12], TRANSPOSE_8(_blockA, 12), _result); \
        _result = FMA8(_blockB[13], TRANSPOSE_8(_blockA, 13), _result); \
        _result = FMA8(_blockB[14], TRANSPOSE_8(_blockA, 14), _result); \
        _result = FMA8(_blockB[15], TRANSPOSE_8(_blockA, 15), _result); \
    }
                        half16 W0 = as_half16(_sub_group_block_read8(
                                (const __global uint *)wei1));
#if USE_32OC_UNROLL
                        half16 W1 = as_half16(_sub_group_block_read8(
                                (const __global uint *)&wei1[IC * KDHW_SIZE
                                        * OC_BLOCK]));
#endif

                        half8 A0 = as_half8(_sub_group_block_read_us8(
                                (const __global ushort *)src1));
                        MULTIPLY_BLOCKS_8x16(C00, A0, W0);
#if USE_32OC_UNROLL
                        MULTIPLY_BLOCKS_8x16(C10, A0, W1);
#endif

                        A0 = as_half8(_sub_group_block_read_us8(
                                (const __global ushort *)&src1[8 * IC_BLOCK]));
                        MULTIPLY_BLOCKS_8x16(C01, A0, W0);
#if USE_32OC_UNROLL
                        MULTIPLY_BLOCKS_8x16(C11, A0, W1);
#endif

                        A0 = as_half8(_sub_group_block_read_us8(
                                (const __global ushort *)&src1[MB_BLOCK * IC_FULL
                                        * IDHW_SIZE]));
                        MULTIPLY_BLOCKS_8x16(C02, A0, W0);
#if USE_32OC_UNROLL
                        MULTIPLY_BLOCKS_8x16(C12, A0, W1);
#endif

                        A0 = as_half8(_sub_group_block_read_us8(
                                (const __global ushort *)&src1[MB_BLOCK * IC_FULL
                                                * IDHW_SIZE
                                        + 8 * IC_BLOCK]));
                        MULTIPLY_BLOCKS_8x16(C03, A0, W0);
#if USE_32OC_UNROLL
                        MULTIPLY_BLOCKS_8x16(C13, A0, W1);
#endif
                        src1 += IC_BLOCK * IDHW_SIZE * MB_BLOCK;
                        wei1 += IC_BLOCK * KDHW_SIZE * OC_BLOCK;
                    }
#if KH != 1 || KW != 1 || KD != 1
                }
#endif
#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1) || (HAS_PAD_W && KW == 1))
    }
#endif
    __global half *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK * MB_BLOCK
            + g * OC * ODHW_SIZE * MB_BLOCK + od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK
            + oh * OW_FULL * OC_BLOCK * MB_BLOCK + ow * OC_BLOCK * MB_BLOCK;
#if USE_32OC_UNROLL
    __global half *dst_write1 = dst_write0 + OC_BLOCK * ODHW_SIZE * MB_BLOCK;
#endif

#if WITH_SUM == 1
    half8 blockS00 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)dst_write0));
    half8 blockS01 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)(dst_write0 + 8 * OC_BLOCK)));
#if USE_32OC_UNROLL
    half8 blockS10 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)dst_write1));
    half8 blockS11 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)(dst_write1 + 8 * OC_BLOCK)));
#endif
#if SUM_SCALE == 1
    C00 += blockS00;
    C01 += blockS01;
#if USE_32OC_UNROLL
    C10 += blockS10;
    C11 += blockS11;
#endif
#else
    C00 = fma(blockS00, (half8)sum_scale, C00);
    C01 = fma(blockS01, (half8)sum_scale, C01);
#if USE_32OC_UNROLL
    C10 = fma(blockS10, (half8)sum_scale, C10);
    C11 = fma(blockS11, (half8)sum_scale, C11);
#endif
#endif
#endif

#if WITH_ELTWISE == 1
    DO_ELTWISE(C00, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C01, 8, eltwise_alpha, eltwise_beta);
#if USE_32OC_UNROLL
    DO_ELTWISE(C10, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C11, 8, eltwise_alpha, eltwise_beta);
#endif
#endif

#if OUTPUT_LEFTOVERS
    if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
        for (int i = 0; i < 8; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; C00[i] = FUSED_OPS_RESULT_SCALAR0; }
            { FUSED_OPS_SCALAR1; C00[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED) {
                dst_write0[i * OC_BLOCK + local_id] = C00[i];
                dst_write0[8 * OC_BLOCK + i * OC_BLOCK + local_id] = C01[i];
            }
        }
    } else
#endif // OUTPUT_LEFTOVERS
    {
#if HAS_FUSED_OPS
        { FUSED_OPS_VEC0; C00 = FUSED_OPS_RESULT_VEC0; }
        { FUSED_OPS_VEC1; C01 = FUSED_OPS_RESULT_VEC1; }
#endif
        _sub_group_block_write_us8(
            (__global ushort *)dst_write0, as_ushort8(C00));
        _sub_group_block_write_us8(
            (__global ushort *)&dst_write0[8 * OC_BLOCK], as_ushort8(C01));
#if USE_32OC_UNROLL
        _sub_group_block_write_us8(
            (__global ushort *)&dst_write1[0], as_ushort8(C10));
        _sub_group_block_write_us8(
            (__global ushort *)&dst_write1[8 * OC_BLOCK], as_ushort8(C11));
#endif
    }

#if WITH_SUM == 1
    half8 blockS02 = as_half8(
            _sub_group_block_read_us8((const __global ushort *)(dst_write0
                    + MB_BLOCK * OC_FULL * G * ODHW_SIZE)));
    half8 blockS03 = as_half8(
            _sub_group_block_read_us8((const __global ushort *)(dst_write0
                    + MB_BLOCK * OC_FULL * G * ODHW_SIZE + 8 * OC_BLOCK)));
#if USE_32OC_UNROLL
    half8 blockS12 = as_half8(
            _sub_group_block_read_us8((const __global ushort *)(dst_write1
                    + MB_BLOCK * OC_FULL * G * ODHW_SIZE)));
    half8 blockS13 = as_half8(
            _sub_group_block_read_us8((const __global ushort *)(dst_write1
                    + MB_BLOCK * OC_FULL * G * ODHW_SIZE + 8 * OC_BLOCK)));
#endif
#if SUM_SCALE == 1
    C02 += blockS02;
    C03 += blockS03;
#if USE_32OC_UNROLL
    C12 += blockS12;
    C13 += blockS13;
#endif
#else
    C02 = fma(blockS02, (half8)sum_scale, C02);
    C03 = fma(blockS03, (half8)sum_scale, C03);
#if USE_32OC_UNROLL
    C12 = fma(blockS12, (half8)sum_scale, C12);
    C13 = fma(blockS13, (half8)sum_scale, C13);
#endif
#endif
#endif
#if WITH_ELTWISE == 1
    DO_ELTWISE(C02, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C03, 8, eltwise_alpha, eltwise_beta);
#if USE_32OC_UNROLL
    DO_ELTWISE(C12, 8, eltwise_alpha, eltwise_beta);
    DO_ELTWISE(C13, 8, eltwise_alpha, eltwise_beta);
#endif
#endif

#if OUTPUT_LEFTOVERS
    if ((oc+1)*OC_BLOCK >= OC_NOTALLIGNED) {
        for (int i = 0; i < 8; i++) {

#if HAS_FUSED_OPS
            { FUSED_OPS_SCALAR0; C02[i] = FUSED_OPS_RESULT_SCALAR0; }
            { FUSED_OPS_SCALAR1; C03[i] = FUSED_OPS_RESULT_SCALAR1; }
#endif
            if (oc * OC_BLOCK + local_id < OC_NOTALLIGNED) {
                dst_write0[MB_BLOCK * OC_FULL * ODHW_SIZE
                        + i * OC_BLOCK + local_id] = C02[i];
                dst_write0[MB_BLOCK * OC_FULL * ODHW_SIZE + 8 * OC_BLOCK
                        + i * OC_BLOCK + local_id] = C03[i];
            }
        }
    } else
#endif // OUTPUT_LEFTOVERS
    {
#if HAS_FUSED_OPS
        { FUSED_OPS_VEC2; C02 = FUSED_OPS_RESULT_VEC2; }
        { FUSED_OPS_VEC3; C03 = FUSED_OPS_RESULT_VEC3; }
#endif
        _sub_group_block_write_us8(
                (__global ushort *)&dst_write0[MB_BLOCK * OC_FULL * ODHW_SIZE],
                as_ushort8(C02));
        _sub_group_block_write_us8(
                (__global ushort *)&dst_write0[MB_BLOCK * OC_FULL * ODHW_SIZE
                        + 8 * OC_BLOCK],
                as_ushort8(C03));
#if USE_32OC_UNROLL
        _sub_group_block_write_us8(
                (__global ushort *)&dst_write1[MB_BLOCK * OC_FULL * ODHW_SIZE],
                as_ushort8(C12));
        _sub_group_block_write_us8(
                (__global ushort *)&dst_write1[MB_BLOCK * OC_FULL * ODHW_SIZE
                        + 8 * OC_BLOCK],
                as_ushort8(C13));
#endif
    }
#endif

#if VER_8OW16C == 1 && (IC % 16 == 0 || (IC == 8 && G != 1))
    /* Regular convolution. */
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
    half blockC00[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++)
        blockC00[i] = bias[oc * OC_BLOCK + local_id + g * OC];
#else
    half8 blockC00 = bias[oc * OC_BLOCK + local_id + g * OC];
#if OW_BLOCK == 16
    half8 blockC01 = blockC00;
#endif
#endif
#else
#if OW_BLOCK != 8 && OW_BLOCK != 16
    half blockC00[OW_BLOCK] = {0.0f};
#else
    half8 blockC00 = 0.0f;
#if OW_BLOCK == 16
    half8 blockC01 = 0.0f;
#endif
#endif
#endif

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;

    /* shift input pointers */
#if IC == 8 && G != 1
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_BLOCK + ih * IW_FULL * IC_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK + (g / 2) * IDHW_SIZE * IC * 2;
#else
    src += input_offset + mb * IC_FULL * IDHW_SIZE + iw * IC_BLOCK + ih * IW_FULL * IC_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK + g * IC * IDHW_SIZE;
#endif
    wei += goc * KDHW_SIZE * IC * OC_BLOCK + g * FILTER_GROUPS_PITCH;

#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1))
    if (!(id < 0 || id >= ID || ih < 0 || ih >= IH)) {
#endif
        int icb = 0;
        do {
#if KH != 1 || KW != 1 || KD != 1
            for (int kd = 0; kd < KD; ++kd)
                for (int kh = 0; kh < KH; ++kh) {

#if CASE_3D
                    if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID)
                        continue;
#endif
                    if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                        continue;

                    const __global half *src1 = src
                            + kd * (1 + DD) * IH_FULL * IW_FULL * IC_BLOCK
                            + kh * (1 + DH) * IW_FULL * IC_BLOCK;

                    half tempA[SW * OW_BLOCK + KW * (1 + DW)];
                    int k = iw;
#if OW % OW_BLOCK != 0 || HAS_PAD_W
                    if (k < 0 || k + SW * OW_BLOCK + KW * (1 + DW) >= IW) {
                        __attribute__((opencl_unroll_hint(SW * OW_BLOCK
                                + KW * (1 + DW)))) // attr:no-format
                        for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW);
                                i++) {
                            if (k >= 0 && k < IW)
                                tempA[i] = as_half(
                                        _sub_group_block_read_us((
                                                const __global ushort *)(&src1[i
                                                * IC_BLOCK])));
                            else
                                tempA[i] = 0.0h;
                            k++;
                        }
                    } else {
#endif
                        __attribute__((opencl_unroll_hint(SW * OW_BLOCK
                                + KW * (1 + DW)))) // attr:no-format
                        for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW);
                                i++) {
                            tempA[i] = as_half(_sub_group_block_read_us(
                                    (const __global ushort
                                                    *)(&src1[i * IC_BLOCK])));
                        }
#if OW % OW_BLOCK != 0 || HAS_PAD_W
                    }
#endif

                    __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                    for (int kw = 0; kw < KW; ++kw) {

                        const __global half *wei1 = wei
                                + kd * KH * KW * IC_BLOCK * OC_BLOCK
                                + kh * KW * IC_BLOCK * OC_BLOCK
                                + kw * IC_BLOCK * OC_BLOCK;

#else
        const __global half *src1 = src;
        const __global half *wei1 = wei;
#endif
#define TRANSPOSE_1(_block, _col) (half)_sub_group_shuffle(_block, _col)

#define FMA8(a, b, c) fma((half)(a), (half)b, (half)c)

#define MULTIPLY_BLOCKS_8x16(_result, _blockA, _blockB, _blockB1) \
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
                        half8 blockB00
                                = as_half8(_sub_group_block_read_us8(
                                        (const __global ushort *)wei1));
#else
                        half8 blockB00
                                = as_half8(_sub_group_block_read_us8(
                                        (const __global ushort *)wei1));
                        half8 blockB01
                                = as_half8(_sub_group_block_read_us8(
                                        (const __global ushort *)(wei1
                                                + 8 * OC_BLOCK)));
#endif

#if KH != 1 || KW != 1 || KD != 1
                        half blockA[OW_BLOCK];
                        __attribute__((
                                opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                        for (int i = 0; i < OW_BLOCK; i++) {
                            blockA[i] = tempA[kw * (1 + DW) + SW * i];
                        }
#else
#if OW_BLOCK != 8 || HAS_PAD_W
        half blockA[OW_BLOCK];
#else
        half8 blockA;
#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        if (ow == OW_LAST) {
            for (int i = 0; i < OW - OW_LAST; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA[i] = 0.0f;
                } else {
#endif
                    blockA[i] = as_half(
                            _sub_group_block_read_us((const __global ushort
                                            *)(&src1[i * IC_BLOCK * SW])));
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
                    blockA[i] = as_half(
                            _sub_group_block_read_us((const __global ushort
                                            *)(&src1[i * IC_BLOCK * SW])));
#if HAS_PAD_W
                }
#endif
            }
#else
        blockA = as_half8(_sub_group_block_read_us8(
                (const __global ushort *)(&src1[0])));
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
                            MULTIPLY_BLOCKS_8x16(
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
            MULTIPLY_BLOCKS_8x16(blockC00[i], blockA[i], blockB00, blockB01);
            MULTIPLY_BLOCKS_8x16(
                    blockC01[i], blockA[i + 8], blockB00, blockB01);
#endif
        }
#endif

#undef TRANSPOSE_1
#undef MULTIPLY_BLOCKS_8x16
#undef MULTIPLY_BLOCKS_FIRST_HALF
#undef MULTIPLY_BLOCKS_SECOND_HALF
#if KH != 1 || KW != 1 || KD != 1
                    }
                }
#endif
            src += IDHW_SIZE * IC_BLOCK;
            wei += OC_BLOCK * KDHW_SIZE * IC_BLOCK;
            icb += IC_BLOCK;
        } while (icb < IC);
#if ((HAS_PAD_D && KD == 1) || (HAS_PAD_H && KH == 1))
    }
#endif

#if OC == 8 && G != 1
    __global half *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK + (g / 2) * OC_BLOCK * ODHW_SIZE + (g % 2) * OC
            + od * OH_FULL * OW_FULL * OC_BLOCK + oh * OW_FULL * OC_BLOCK + ow * OC_BLOCK;
#else
    __global half *dst_write0 = dst + output_offset + mb * OC_FULL * ODHW_SIZE
            + goc * ODHW_SIZE * OC_BLOCK + g * OC * ODHW_SIZE
            + od * OH_FULL * OW_FULL * OC_BLOCK + oh * OW_FULL * OC_BLOCK + ow * OC_BLOCK;
#endif

#if WITH_SUM == 1
#if OW_BLOCK != 8 && OW_BLOCK != 16
    half blockS00[OW_BLOCK];
#else
    half8 blockS00;
#if OW_BLOCK == 16
    half8 blockS01;
#endif
#endif
#if OW % OW_BLOCK != 0
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[i * OC_BLOCK]));
        }
    } else {
#endif
#if OW_BLOCK != 8 && OW_BLOCK != 16
        for (int i = 0; i < OW_BLOCK; i++) {
            blockS00[i] = as_half(_sub_group_block_read_us(
                    (const __global ushort *)&dst_write0[i * OC_BLOCK]));
        }
#else
    blockS00 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)dst_write0));
#if OW_BLOCK == 16
    blockS01 = as_half8(_sub_group_block_read_us8(
            (const __global ushort *)&dst_write0[8 * OC_BLOCK]));
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
        blockC00[i] = fma(blockS00[i], (half)sum_scale, blockC00[i]);
#endif
    }
#else
#if SUM_SCALE == 1
    blockC00 += blockS00;
    blockC01 += blockS01;
#else
    blockC00 = fma(blockS00, (half8)sum_scale, blockC00);
    blockC01 = fma(blockS01, (half8)sum_scale, blockC01);
#endif
#endif
#endif

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
                _sub_group_block_write_us(
                        (__global ushort *)(&dst_write0[i * OC_BLOCK]),
                        as_ushort(blockC00[i]));
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
                _sub_group_block_write_us(
                        (__global ushort *)(&dst_write0[i * OC_BLOCK]),
                        as_ushort(blockC00[i]));
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

        _sub_group_block_write_us8(
                (__global ushort *)(&dst_write0[0]), as_ushort8(blockC00));
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

    _sub_group_block_write_us8(
            (__global ushort *)(&dst_write0[8 * OC_BLOCK]),
            as_ushort8(blockC01));
    }
#endif //  OC == 8 && G != 1
#endif
#endif
#if OW % OW_BLOCK != 0
    }
#endif
#endif
}
