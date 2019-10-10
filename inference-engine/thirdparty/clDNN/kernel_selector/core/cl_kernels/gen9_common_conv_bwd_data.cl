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

__attribute__((reqd_work_group_size(16, 1, 1)))
#    if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#    endif

KERNEL(gen9_common_conv_bwd_data_kernel)(
        const  __global DATA_T *diff_dst,
        __global DATA_T *diff_src,
        const __global DATA_T *wei,
#if WITH_BIAS
        const __global DATA_T *bias,
#endif
        uint split_idx)
{

#    if VER_16MB16C == 1 || VER_8OW16C == 1
    const int mb_unroll = 16;

    const int ic = get_group_id(0);
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    int mb = get_group_id(2) * mb_unroll;

#        if IS_DW
    const int g = ic * IC_BLOCK;
    const int gic = 0;
#        else
    const int g = split_idx;
    const int gic = ic;
#        endif

#        if CASE_3D
    const int id = sp / (IW * IH);
    const int ihw = sp % (IW * IH);
#        else
    const int id = 0;
    const int ihw = sp;
#        endif
    const int ih = ihw / IW;
    const int iw = ihw % IW;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;

#if WITH_BIAS
    DATA8_T blockC00 = bias[ic * IC_BLOCK + local_id];
    DATA8_T blockC01 = bias[ic * IC_BLOCK + local_id];
#else
    DATA8_T blockC00 = 0.0f;
    DATA8_T blockC01 = 0.0f;
#endif


    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK;

    int ocb = 0;
    do {
#        if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {

                    if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH))
                        continue;
#            if CASE_3D
                    if (id + PD < kd * (1 + DD))
                        continue;
                    int od = id - kd * (1 + DD) + PD;
                    if (od % SD != 0)
                        continue;
                    od /= SD;
                    if (od >= OD)
                        continue;
#            endif

                    int ow = iw - kw * (1 + DW) + PW;
                    int oh = ih - kh * (1 + DH) + PH;
                    if (ow % SW != 0 || oh % SH != 0)
                        continue;
                    ow /= SW;
                    oh /= SH;
                    if (oh >= OH || ow >= OW)
                        continue;

                    const __global DATA_T *diff_dst1 = diff_dst
                            + ow * OC_BLOCK * MB_BLOCK
                            + oh * OW * OC_BLOCK * MB_BLOCK;
#            if CASE_3D
                    diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#            endif
#            if IS_DW
                    const __global DATA_T *wei1 = wei
#                if CASE_3D
                            + kd * KH * KW * OC_BLOCK
#                endif
                            + kh * KW * OC_BLOCK + kw * OC_BLOCK;
#            else
                    const __global DATA_T *wei1 = wei
#                if CASE_3D
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
#                endif
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;
#            endif
#        else
        int ow = (iw + PW);
        int oh = (ih + PH);
#            if CASE_3D
        int od = (id + PD);
#            endif
        bool do_ker = true;
#            if SW != 1 || SH != 1 || SD != 1
        do_ker = ow % SW == 0 && oh % SH == 0;
        ow /= SW;
        oh /= SH;
#                if CASE_3D
        do_ker = do_ker && od % SD == 0;
        od /= SD;
#                endif
#            endif
#            if PH != 0 || PW != 0 || PD != 0
        do_ker = do_ker && (oh < OH && ow < OW);
#                if CASE_3D
        do_ker = do_ker && (od < OD);
#                endif
#            endif
#            if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        if (do_ker) {
#            endif
            const __global DATA_T *diff_dst1 = diff_dst
                    + ow * OC_BLOCK * MB_BLOCK + oh * OW * OC_BLOCK * MB_BLOCK;
#            if CASE_3D
            diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#            endif
            const __global DATA_T *wei1 = wei;
#        endif

#        if MB == MB_LAST
#            define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk)        \
                {                                                     \
                    (_block) = AS_DATA8_T(BLOCK_READ8( \
                            (const __global BLOCK_DATA_T *)((_diff_dst)       \
                                    + (mb_chunk)*OC_BLOCK)));         \
                }
#        else
#            define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk)                 \
                {                                                              \
                    if (mb == MB_LAST) {                                       \
                        for (int i = 0; i < min(8, MB - MB_LAST - (mb_chunk)); \
                                i++)                                           \
                            (_block)[i] = AS_DATA_T(BLOCK_READ( \
                                    (const __global BLOCK_DATA_T *)(&(                 \
                                            _diff_dst)[((mb_chunk) + i) * OC   \
                                            * G * OD * OH * OW])));            \
                    } else {                                                   \
                        for (int i = 0; i < 8; i++)                            \
                            (_block)[i] = AS_DATA_T(BLOCK_READ( \
                                    (const __global BLOCK_DATA_T *)(&(                 \
                                            _diff_dst)[((mb_chunk) + i) * OC   \
                                            * G * OD * OH * OW])));            \
                    }                                                          \
                }
#        endif

#        if MB == MB_LAST
#            define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk)        \
                {                                                     \
                    BLOCK_WRITE8(                     \
                            (__global unsigned int *)(&(              \
                                    _diff_src)[(mb_chunk)*IC_BLOCK]), \
                            AS_UINT8_T((_block)));                      \
                }
#        else
#            define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk)                 \
                {                                                              \
                    if (mb == MB_LAST) {                                       \
                        for (int i = 0; i < min(8, MB - MB_LAST - (mb_chunk)); \
                                i++) {                                         \
                            BLOCK_WRITE(                       \
                                    (__global unsigned int *)(&(               \
                                            _diff_src)[((mb_chunk) + i) * IC   \
                                            * G * ID * IH * IW]),              \
                                    AS_UINT_T((_block)[i]));                     \
                        }                                                      \
                    } else {                                                   \
                        for (int i = 0; i < 8; i++) {                          \
                            BLOCK_WRITE(                       \
                                    (__global unsigned int *)(&(               \
                                            _diff_src)[((mb_chunk) + i) * IC   \
                                            * G * ID * IH * IW]),              \
                                    AS_UINT_T((_block)[i]));                     \
                        }                                                      \
                    }                                                          \
                }
#        endif

#        if DT_F32
#        define TRANSPOSE_8(_block, _col) \
            (DATA8_T)(intel_sub_group_shuffle(_block, _col))
#        else
#        define TRANSPOSE_8(_block, _col)                     \
            (DATA8_T)(intel_sub_group_shuffle(_block[0], _col), \
                    intel_sub_group_shuffle(_block[1], _col), \
                    intel_sub_group_shuffle(_block[2], _col), \
                    intel_sub_group_shuffle(_block[3], _col), \
                    intel_sub_group_shuffle(_block[4], _col), \
                    intel_sub_group_shuffle(_block[5], _col), \
                    intel_sub_group_shuffle(_block[6], _col), \
                    intel_sub_group_shuffle(_block[7], _col))
#        endif

#        define FMA8(a, b, c) fma((DATA8_T)(a), (DATA8_T)b, (DATA8_T)c)

#        define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1)       \
            {                                                                  \
                _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0), _result);  \
                _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1), _result);  \
                _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2), _result);  \
                _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3), _result);  \
                _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4), _result);  \
                _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5), _result);  \
                _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6), _result);  \
                _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7), _result);  \
                _result = FMA8(_blockB1.s0, TRANSPOSE_8(_blockA, 8), _result); \
                _result = FMA8(_blockB1.s1, TRANSPOSE_8(_blockA, 9), _result); \
                _result = FMA8(                                                \
                        _blockB1.s2, TRANSPOSE_8(_blockA, 10), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s3, TRANSPOSE_8(_blockA, 11), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s4, TRANSPOSE_8(_blockA, 12), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s5, TRANSPOSE_8(_blockA, 13), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s6, TRANSPOSE_8(_blockA, 14), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s7, TRANSPOSE_8(_blockA, 15), _result);       \
            }

#        if IS_DW
                    DATA_T blockB00 = AS_DATA_T(BLOCK_READ(
                            (const __global BLOCK_DATA_T *)wei1));
#        else
            DATA8_T blockB00 = AS_DATA8_T(
                    BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
            DATA8_T blockB01 = AS_DATA8_T(BLOCK_READ8(
                    (const __global BLOCK_DATA_T *)(wei1 + 8 * IC_BLOCK)));
#        endif
                    DATA8_T blockA;

                    LOAD_DIFF_DST(blockA, diff_dst1, 0);
#        if IS_DW
                    blockC00 = fma(blockA, (DATA8_T)blockB00, blockC00);
#        else
            MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB00, blockB01);
#        endif

                    LOAD_DIFF_DST(blockA, diff_dst1, 8);
                    if ((mb != MB_LAST) || (MB % 16 > 8)) {
#        if IS_DW
                        blockC01 = fma(blockA, (DATA8_T)blockB00, blockC01);
#        else
                MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB00, blockB01);
#        endif
                    }

#        undef TRANSPOSE_BLOCK_8
#        undef MULTIPLY_BLOCKS_8x8
#        if KH != 1 || KW != 1 || KD != 1
                }
#        else
#            if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        }
#            endif
#        endif
        diff_dst += OC_BLOCK * OD * OH * OW * MB_BLOCK;
        wei += IC * KD * KH * KW * OC_BLOCK;
        ocb += OC_BLOCK;
    } while (ocb < OC);

    __global DATA_T *src_write0 = diff_src + mb * IC * G * ID * IH * IW
            + gic * ID * IH * IW * IC_BLOCK * MB_BLOCK
            + g * IC * ID * IH * IW * MB_BLOCK
            + id * IH * IW * IC_BLOCK * MB_BLOCK + ih * IW * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);

    SAVE_SRC_DIFF(blockC00, src_write0, 0);
    SAVE_SRC_DIFF(blockC01, src_write0, 8);

#    endif
}

