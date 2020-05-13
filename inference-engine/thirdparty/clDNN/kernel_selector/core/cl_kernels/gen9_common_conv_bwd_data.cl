/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl_types.h"
#include "include/fetch.cl"
#include "include/data_types.cl"

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#if BWD_DATA == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
KERNEL(gen9_common_conv_bwd_data_kernel)(
        const  __global DATA_T *diff_dst,
        __global DATA_T * restrict diff_src,
        const __global DATA_T *wei,
#if WITH_BIAS
        const __global DATA_T *bias,
#endif
#if HAS_FUSED_OPS_DECLS
        FUSED_OPS_DECLS,
#endif
        uint split_idx
        )
{
    const int input_offset = (INPUT0_PAD_BEFORE_FEATURE_NUM / OC_BLOCK) * OD_FULL * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Z) * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_Y) * OW_FULL * OC_BLOCK * MB_BLOCK +
                             (INPUT0_PAD_BEFORE_SIZE_X) * OC_BLOCK * MB_BLOCK;
    const int output_offset = (OUTPUT_PAD_BEFORE_FEATURE_NUM / IC_BLOCK) * ID_FULL * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Z) * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_Y) * IW_FULL * IC_BLOCK * MB_BLOCK +
                              (OUTPUT_PAD_BEFORE_SIZE_X) * IC_BLOCK * MB_BLOCK;
#if VER_16MB16C == 1
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);

    const int icb_mb = get_group_id(2);
    const int mb = icb_mb / (G * IC / ICB) * MB_BLOCK;
    const int icb = icb_mb % (G * IC / ICB);
    const int ic = (icb * ICB) / IC_BLOCK + get_group_id(0);

    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);

#if CASE_3D
    const int id = sp / (IW * IH);
    const int ihw = sp % (IW * IH);
#else
    const int id = 0;
    const int ihw = sp;
#endif
    const int ih = ihw / IW;
    const int iw = ihw % IW;

    diff_dst += input_offset + mb * OC_FULL * G * OD_FULL * OH_FULL * OW_FULL + g * OC * OD_FULL * OH_FULL * OW_FULL * MB_BLOCK;

#if WITH_BIAS
    DATA8_T blockC00 = (DATA8_T)bias[g * IC + gic * IC_BLOCK + local_id];
    DATA8_T blockC01 = (DATA8_T)bias[g * IC + gic * IC_BLOCK + local_id];
#else
    DATA8_T blockC00 = 0.0f;
    DATA8_T blockC01 = 0.0f;
#endif

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;
    int ocb = 0;
    do {
#if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {

                    if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH))
                        continue;
#if CASE_3D
                    if (id + PD < kd * (1 + DD)) continue;
                    int od = id - kd * (1 + DD) + PD;
                    if (od % SD != 0) continue;
                    od /= SD;
                    if (od >= OD) continue;
#endif

                    int ow = iw - kw * (1 + DW) + PW;
                    int oh = ih - kh * (1 + DH) + PH;
#if SW != 1 || SH != 1
                    if (ow % SW != 0 || oh % SH != 0) continue;
                    ow /= SW;
                    oh /= SH;
#endif
                    if (oh >= OH || ow >= OW) continue;

                    const __global DATA_T *diff_dst1 = diff_dst
                            + ow * OC_BLOCK * MB_BLOCK
                            + oh * OW_FULL * OC_BLOCK * MB_BLOCK;
#if CASE_3D
                    diff_dst1 += od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK;
#endif
                    const __global DATA_T *wei1 = wei
#if CASE_3D
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
#endif
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;
#else
        int ow = (iw + PW);
        int oh = (ih + PH);
#if CASE_3D
        int od = (id + PD);
#endif
        bool do_ker = true;
#if SW != 1 || SH != 1 || SD != 1
        do_ker = ow % SW == 0 && oh % SH == 0;
        ow /= SW;
        oh /= SH;
#if CASE_3D
        do_ker = do_ker && od % SD == 0;
        od /= SD;
#endif
#endif
#if PH != 0 || PW != 0 || PD != 0
        do_ker = do_ker && (oh < OH && ow < OW);
#if CASE_3D
        do_ker = do_ker && (od < OD);
#endif
#endif
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        if (do_ker) {
#endif
            const __global DATA_T *diff_dst1 = diff_dst
                    + ow * OC_BLOCK * MB_BLOCK + oh * OW_FULL * OC_BLOCK * MB_BLOCK;
#if CASE_3D
            diff_dst1 += od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK;
#endif
            const __global DATA_T *wei1 = wei;
#endif

#define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk) \
    { \
        (_block) = AS_DATA8_T( \
                BLOCK_READ8((const __global BLOCK_DATA_T *)((_diff_dst) \
                        + (mb_chunk)*OC_BLOCK))); \
    }

#define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk) \
    { \
        BLOCK_WRITE8((const __global BLOCK_DATA_T *)(&( \
                             _diff_src)[(mb_chunk)*IC_BLOCK]), \
                AS_BLOCK_DATA8_T((_block))); \
    }

#if DT_F32
#define TRANSPOSE_8(_block, _col) \
    (DATA8_T)(intel_sub_group_shuffle(_block, _col))
#else
#define TRANSPOSE_8(_block, _col) \
    (DATA8_T)(intel_sub_group_shuffle(_block[0], _col), \
            intel_sub_group_shuffle(_block[1], _col), \
            intel_sub_group_shuffle(_block[2], _col), \
            intel_sub_group_shuffle(_block[3], _col), \
            intel_sub_group_shuffle(_block[4], _col), \
            intel_sub_group_shuffle(_block[5], _col), \
            intel_sub_group_shuffle(_block[6], _col), \
            intel_sub_group_shuffle(_block[7], _col))
#endif

#define FMA8(a, b, c) fma((DATA8_T)(a), (DATA8_T)b, (DATA8_T)c)

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

                    DATA8_T blockA0, blockA1;
                    LOAD_DIFF_DST(blockA0, diff_dst1, 0);
                    LOAD_DIFF_DST(blockA1, diff_dst1, 8);
                    DATA8_T blockB00 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
                    DATA8_T blockB01 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)(wei1
                                    + 8 * IC_BLOCK)));
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA0, blockB00, blockB01);
                    MULTIPLY_BLOCKS_8x8(blockC01, blockA1, blockB00, blockB01);

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
#if KH != 1 || KW != 1 || KD != 1
                }
#else
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        }
#endif
#endif
        diff_dst += OC_BLOCK * OD_FULL * OH_FULL * OW_FULL * MB_BLOCK;
        wei += IC * KD * KH * KW * OC_BLOCK;
        ocb += OC_BLOCK;
    } while (ocb < OC);

    __global DATA_T *src_write0 = diff_src + OUTPUT_OFFSET + mb * IC_FULL * G * ID_FULL * IH_FULL * IW_FULL
            + gic * ID_FULL * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK
            + g * IC * ID_FULL * IH_FULL * IW_FULL * MB_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK + ih * IW_FULL * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);

#if HAS_FUSED_OPS
    {
        FUSED_OPS_BLOCK_C00;
        blockC00 = FUSED_OPS_RESULT_BLOCK_C00;
    }
    {
        FUSED_OPS_BLOCK_C01;
        blockC01 = FUSED_OPS_RESULT_BLOCK_C01;
    }
#endif

    SAVE_SRC_DIFF(blockC00, src_write0, 0);
    SAVE_SRC_DIFF(blockC01, src_write0, 8);

#endif
#if VER_8OW16C == 1
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    const int icb_mb = get_group_id(2);
    const int mb = icb_mb / (G * IC / ICB);
    const int icb = icb_mb % (G * IC / ICB);
    const int ic = (icb * ICB) / IC_BLOCK + get_group_id(0);

    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);

#if CASE_3D
    const int id = sp / (IWB * IH);
    const int ihw = sp % (IWB * IH);
#else
    const int id = 0;
    const int ihw = sp;
#endif
    const int ih = ihw / IWB;
    const int iw = (ihw % IWB) * IW_BLOCK;

    diff_dst += input_offset + mb * OC_FULL * G * OD_FULL * OH_FULL * OW_FULL + g * OC * OD_FULL * OH_FULL * OW_FULL * MB_BLOCK;
    DATA_T blockC00[IW_BLOCK] = {0.0f};

#if WITH_BIAS
    for (int i = 0; i < IW_BLOCK; i++)
        blockC00[i] = bias[g * IC + gic * IC_BLOCK + local_id];
#endif

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {

                if (ih + PH < kh * (1 + DH)) continue;
#if CASE_3D
                if (id + PD < kd * (1 + DD)) continue;
                int od = id - kd * (1 + DD) + PD;
                if (od % SD != 0) continue;
                od /= SD;
                if (od >= OD) continue;
#endif

                int oh = ih - kh * (1 + DH) + PH;
                if (oh % SH != 0) continue;
                oh /= SH;
                if (oh >= OH) continue;

                const __global DATA_T *diff_dst1
                        = diff_dst + oh * OW_FULL * OC_BLOCK * MB_BLOCK;
#if CASE_3D
                diff_dst1 += od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK;
#endif
                const __global DATA_T *wei1 = wei
#if CASE_3D
                        + kd * KH * KW * OC_BLOCK * IC_BLOCK
#endif
                        + kh * KW * OC_BLOCK * IC_BLOCK
                        + kw * OC_BLOCK * IC_BLOCK;
#else
    int oh = (ih + PH);
#if CASE_3D
    int od = (id + PD);
#endif
    bool do_ker = true;
#if SW != 1 || SH != 1 || SD != 1
    do_ker = oh % SH == 0;
    oh /= SH;
#if CASE_3D
    do_ker = do_ker && od % SD == 0;
    od /= SD;
#endif
#endif
#if PH != 0 || PW != 0 || PD != 0
    do_ker = do_ker && (oh < OH);
#if CASE_3D
    do_ker = do_ker && (od < OD);
#endif
#endif
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
    if (do_ker) {
#endif
        const __global DATA_T *diff_dst1
                = diff_dst + oh * OW_FULL * OC_BLOCK * MB_BLOCK;
#if CASE_3D
        diff_dst1 += od * OH_FULL * OW_FULL * OC_BLOCK * MB_BLOCK;
#endif
        const __global DATA_T *wei1 = wei;
#endif

                int ocb = 0;
                do {

#define TRANSPOSE_1(_block, _col) \
    (DATA_T)(intel_sub_group_shuffle(_block, _col))

#define FMA1(a, b, c) fma((DATA_T)(a), (DATA_T)b, (DATA_T)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA1(_blockB.s0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA1(_blockB.s1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA1(_blockB.s2, TRANSPOSE_1(_blockA, 2), _result); \
        _result = FMA1(_blockB.s3, TRANSPOSE_1(_blockA, 3), _result); \
        _result = FMA1(_blockB.s4, TRANSPOSE_1(_blockA, 4), _result); \
        _result = FMA1(_blockB.s5, TRANSPOSE_1(_blockA, 5), _result); \
        _result = FMA1(_blockB.s6, TRANSPOSE_1(_blockA, 6), _result); \
        _result = FMA1(_blockB.s7, TRANSPOSE_1(_blockA, 7), _result); \
        _result = FMA1(_blockB1.s0, TRANSPOSE_1(_blockA, 8), _result); \
        _result = FMA1(_blockB1.s1, TRANSPOSE_1(_blockA, 9), _result); \
        _result = FMA1(_blockB1.s2, TRANSPOSE_1(_blockA, 10), _result); \
        _result = FMA1(_blockB1.s3, TRANSPOSE_1(_blockA, 11), _result); \
        _result = FMA1(_blockB1.s4, TRANSPOSE_1(_blockA, 12), _result); \
        _result = FMA1(_blockB1.s5, TRANSPOSE_1(_blockA, 13), _result); \
        _result = FMA1(_blockB1.s6, TRANSPOSE_1(_blockA, 14), _result); \
        _result = FMA1(_blockB1.s7, TRANSPOSE_1(_blockA, 15), _result); \
    }

                    DATA8_T blockB00 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
                    DATA8_T blockB01 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)(wei1
                                    + 8 * IC_BLOCK)));
                    DATA_T blockA[IW_BLOCK];

                    __attribute__((
                            opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                    for (int i = 0; i < IW_BLOCK; i++) {
#if KW != 1
                        if (iw + i + PW < kw * (1 + DW)) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        int ow = iw + i - kw * (1 + DW) + PW;
#else
                int ow = iw + i + PW;
#endif
#if SW != 1
                        if (ow % SW != 0) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        ow /= SW;
#endif
                        if (ow >= OW) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        blockA[i] = AS_DATA_T(
                                BLOCK_READ((const __global BLOCK_DATA_T *)(&(
                                        diff_dst1)[ow * OC_BLOCK])));
                    }

                    __attribute__((
                            opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                    for (int i = 0; i < IW_BLOCK; i++) {
                        MULTIPLY_BLOCKS_8x8(
                                blockC00[i], blockA[i], blockB00, blockB01);
                    }

                    diff_dst1 += OC_BLOCK * OD_FULL * OH_FULL * OW_FULL * MB_BLOCK;
                    wei1 += IC * KD * KH * KW * OC_BLOCK;
                    ocb += OC_BLOCK;
                } while (ocb < OC);

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
#if KH != 1 || KW != 1 || KD != 1
            }
#else
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
    }
#endif
#endif

    __global DATA_T *src_write0 = diff_src + output_offset + mb * IC_FULL * G * ID_FULL * IH_FULL * IW_FULL
            + gic * ID_FULL * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK
            + g * IC * ID_FULL * IH_FULL * IW_FULL * MB_BLOCK
            + id * IH_FULL * IW_FULL * IC_BLOCK * MB_BLOCK + ih * IW_FULL * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    for (int i = 0; i < IW_BLOCK; i++) {
        blockC00[i] = ACTIVATION(blockC00[i], ACTIVATION_PARAMS);
        if (iw + i >= IW) continue;
#if HAS_FUSED_OPS
        FUSED_OPS_BLOCK_CI;
        blockC00[i] = FUSED_OPS_RESULT_BLOCK_CI;
#endif
        BLOCK_WRITE((__global BLOCK_DATA_T *)(&(src_write0)[i * IC_BLOCK]),
                AS_BLOCK_DATA_T(blockC00[i]));
    }
#endif
}
#endif
