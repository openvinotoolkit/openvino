// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/mmad.cl"

#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif

#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif

#define ACCUMULATOR_TYPE int
#define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#define ACTIVATION_TYPE float
#define TO_ACTIVATION_TYPE(x) convert_float(x)

#if OUTPUT_X_BLOCK_SIZE == 8
    #define PACKED_TYPE_VEC MAKE_VECTOR_TYPE(PACKED_IN_TYPE, 8)
    #define ACCUMULATOR_TYPE_VEC int8
    #define TO_ACCUMULATOR_TYPE_VEC(x) convert_int8(x)
    #define ACTIVATION_TYPE_VEC float8
    #define TO_ACTIVATION_TYPE_VEC(x) convert_float8(x)
    #define MMAD MMAD_8x8
    #define BLOCK_WRITE(ptr, val) _sub_group_block_write8((__global uint*)(ptr), as_uint8(val));
#elif OUTPUT_X_BLOCK_SIZE == 4
    #define PACKED_TYPE_VEC MAKE_VECTOR_TYPE(PACKED_IN_TYPE, 4)
    #define ACCUMULATOR_TYPE_VEC int4
    #define TO_ACCUMULATOR_TYPE_VEC(x) convert_int4(x)
    #define ACTIVATION_TYPE_VEC float4
    #define TO_ACTIVATION_TYPE_VEC(x) convert_float4(x)
    #define MMAD MMAD_4x8
    #define BLOCK_WRITE(ptr, val) _sub_group_block_write4((__global uint*)(ptr), as_uint4(val));
#else
#error "convolution_gpu_mmad_b_fs_yx_fsv32: Unsupported block size"
#endif

__attribute__((reqd_work_group_size(8, OW_GROUP, 1)))
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(convolution_mmad_b_fs_yx_fsv32)(
    __global INPUT0_TYPE* input,
    __global PACKED_OUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
    , const __global COMPENSATION_TYPE *compensation
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint b = get_global_id(2);
    const uint fg = get_group_id(0);
#if OUTPUT_DIMS == 5
    const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
    const uint y = ((uint)get_global_id(1) / OUTPUT_SIZE_Z) % OUTPUT_SIZE_Y;
    const uint x = ((uint)get_global_id(1) / OUTPUT_SIZE_Z / OUTPUT_SIZE_Y) * OUTPUT_X_BLOCK_SIZE;
#else
    const uint z = 0;
    const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint x = ((uint)get_global_id(1) / OUTPUT_SIZE_Y) * OUTPUT_X_BLOCK_SIZE;
#endif

    const uint lid = get_sub_group_local_id();

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int input_z = z * STRIDE_SIZE_Z - PADDING_SIZE_Z;

    ACCUMULATOR_TYPE_VEC acc[4] = { 0 }; // 4*8 packed channels * OUTPUT_X_BLOCK_SIZE
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    ACCUMULATOR_TYPE_VEC acc_assym_weights = 0;
#endif

    const uint input_offset = INPUT0_GET_INDEX(b,0,0,0);

    uint filter_idx = fg * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z * ISV_SIZE * OSV_SIZE * IFM_BLOCKS;

    const int input_x_pitch = ISV_SIZE;
    const int input_y_pitch = (INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X) * input_x_pitch;
    const int input_z_pitch = (INPUT0_SIZE_Y + INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y) * input_y_pitch;
    const int input_fs_pitch = (INPUT0_SIZE_Z + INPUT0_PAD_BEFORE_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z) * input_z_pitch;
    int in_addr = input_offset + input_x * input_x_pitch + input_y * input_y_pitch + input_z * input_z_pitch;

    for (int icb = 0; icb < IFM_BLOCKS; ++icb) {
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        uchar4 m;
        __attribute__((opencl_unroll_hint(4)))
        for (int i = 0; i < 4; i++) {
            m[i] = icb*32 + lid*4 + i < INPUT0_FEATURE_NUM;
        }
        int mm = as_int(m);
        int8 multiplier = (int8)(sub_group_broadcast(mm, 0),
                                 sub_group_broadcast(mm, 1),
                                 sub_group_broadcast(mm, 2),
                                 sub_group_broadcast(mm, 3),
                                 sub_group_broadcast(mm, 4),
                                 sub_group_broadcast(mm, 5),
                                 sub_group_broadcast(mm, 6),
                                 sub_group_broadcast(mm, 7));
#endif

        __attribute__((opencl_unroll_hint(FILTER_SIZE_Z)))
        for (int kd = 0; kd < FILTER_SIZE_Z ; ++kd) {
            bool z_cross_fm = input_z + kd*DILATION_SIZE_Z < 0 || input_z + kd*DILATION_SIZE_Z >= INPUT0_SIZE_Z;
#if !ASYMMETRIC_DATA_QUANTIZATION
            if (z_cross_fm)
                continue;
#endif
            __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
            for (int kh = 0; kh < FILTER_SIZE_Y ; ++kh) {
                bool y_cross_fm = input_y + kh*DILATION_SIZE_Y < 0 || input_y + kh*DILATION_SIZE_Y >= INPUT0_SIZE_Y;
#if !ASYMMETRIC_DATA_QUANTIZATION
                if (y_cross_fm)
                    continue;
#endif

                PACKED_IN_TYPE line_cache[INPUT_LINE_SIZE] = {0};
                {

                    int xb = 0;
                    for (; xb < INPUT_LINE_SIZE; xb++) {

                        bool x_cross_fm = input_x + xb < 0 || input_x + xb >= INPUT0_SIZE_X;
                        if (y_cross_fm || x_cross_fm || z_cross_fm) {
#if ASYMMETRIC_DATA_QUANTIZATION
                            const int azp_idx = (icb*ISV_SIZE + 4*lid) % ACTIVATIONS_ZERO_POINTS_FEATURE_NUM;
                            line_cache[xb] = AS_TYPE(PACKED_IN_TYPE, ((const __global uint*)(activations_zp + azp_idx))[0]);
#else
                            line_cache[xb] = 0;
#endif
                        }
                        else
                        {
                            line_cache[xb] = AS_TYPE(PACKED_IN_TYPE, _sub_group_block_read((const __global uint*)(input + in_addr +
                                                                          icb * input_fs_pitch +
                                                                          kd * DILATION_SIZE_Z * input_z_pitch +
                                                                          kh * DILATION_SIZE_Y * input_y_pitch +
                                                                          xb * input_x_pitch)));
                        }
                    }
                }
                __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                for (uint kw = 0; kw < FILTER_SIZE_X ; ++kw) {
                    PACKED_TYPE_VEC src;
                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
                        src[i] = line_cache[kw*DILATION_SIZE_X + STRIDE_SIZE_X*i];
                    }

                    const uint f_off = filter_idx + icb * FILTER_SIZE_X*FILTER_SIZE_Y*FILTER_SIZE_Z*ISV_SIZE*OSV_SIZE
                                     + kd * ISV_SIZE * OSV_SIZE * FILTER_SIZE_X * FILTER_SIZE_Y
                                     + kh * ISV_SIZE * OSV_SIZE * FILTER_SIZE_X
                                     + kw * ISV_SIZE * OSV_SIZE;

                    int8 weights_data0 = as_int8(_sub_group_block_read8((const __global uint*)(weights + f_off + 0*8*ISV_SIZE)));
                    int8 weights_data1 = as_int8(_sub_group_block_read8((const __global uint*)(weights + f_off + 1*8*ISV_SIZE)));
                    int8 weights_data2 = as_int8(_sub_group_block_read8((const __global uint*)(weights + f_off + 2*8*ISV_SIZE)));
                    int8 weights_data3 = as_int8(_sub_group_block_read8((const __global uint*)(weights + f_off + 3*8*ISV_SIZE)));

                    acc[0] = MMAD(src, weights_data0, acc[0]); // 8 elements in 4*lid+0 out channel
                    acc[1] = MMAD(src, weights_data1, acc[1]); // 8 elements in 4*lid+1 out channel
                    acc[2] = MMAD(src, weights_data2, acc[2]); // 8 elements in 4*lid+2 out channel
                    acc[3] = MMAD(src, weights_data3, acc[3]); // 8 elements in 4*lid+3 out channel

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                    acc_assym_weights = MMAD(src, multiplier, acc_assym_weights);
#endif
                }
            }
        }
    }

#if BIAS_TERM
    const uint bias_index = fg*OSV_SIZE;
#endif

#if OUTPUT_IS_FP
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst[4];

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+0]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+1]);
        ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(acc[2][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+2]);
        ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(acc[3][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+3]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
        ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(acc[2][i]);
        ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(acc[3][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        const uint idx0 = fg*OSV_SIZE + 4*lid + 0;
        const uint idx1 = fg*OSV_SIZE + 4*lid + 1;
        const uint idx2 = fg*OSV_SIZE + 4*lid + 2;
        const uint idx3 = fg*OSV_SIZE + 4*lid + 3;

        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx0]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx1]);
        res2 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx2]);
        res3 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx3]);
#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV_SIZE + 4*lid + 0];
        res1 += compensation[fg*OSV_SIZE + 4*lid + 1];
        res2 += compensation[fg*OSV_SIZE + 4*lid + 2];
        res3 += compensation[fg*OSV_SIZE + 4*lid + 3];
#endif  // ASYMMETRIC_DATA_QUANTIZATION

#if HAS_FUSED_OPS
        { FUSED_OPS_0; dst[0][i] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; dst[1][i] = FUSED_OPS_RESULT_1; };
        { FUSED_OPS_2; dst[2][i] = FUSED_OPS_RESULT_2; };
        { FUSED_OPS_3; dst[3][i] = FUSED_OPS_RESULT_3; };
#else
        dst[0][i] = TO_OUTPUT_TYPE(res0);
        dst[1][i] = TO_OUTPUT_TYPE(res1);
        dst[2][i] = TO_OUTPUT_TYPE(res2);
        dst[3][i] = TO_OUTPUT_TYPE(res3);
#endif
    }

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
        for (int ofm = 0; ofm < 4; ofm++) {
#if OUTPUT_DIMS == 5
            const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + ofm + 4*lid, z, y, x+i);
#elif OUTPUT_DIMS <= 4
            const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + ofm + 4*lid, y, x+i);
#endif
            bool full_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
            bool full_f = OUTPUT_FEATURE_NUM % OSV_SIZE == 0 || fg * OSV_SIZE + 4 * lid + ofm < OUTPUT_FEATURE_NUM;
            if (full_x && full_f) {
                output[dst_index] = dst[ofm][i];
            }
        }
    }
#else  // OUTPUT_IS_FP
    MAKE_VECTOR_TYPE(PACKED_OUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst;

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+0]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+1]);
        ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(acc[2][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+2]);
        ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(acc[3][i]) + (ACTIVATION_TYPE)(biases[bias_index + 4*lid+3]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
        ACTIVATION_TYPE res2 = TO_ACTIVATION_TYPE(acc[2][i]);
        ACTIVATION_TYPE res3 = TO_ACTIVATION_TYPE(acc[3][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        const uint idx0 = fg*OSV_SIZE + 4*lid + 0;
        const uint idx1 = fg*OSV_SIZE + 4*lid + 1;
        const uint idx2 = fg*OSV_SIZE + 4*lid + 2;
        const uint idx3 = fg*OSV_SIZE + 4*lid + 3;

        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx0]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx1]);
        res2 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx2]);
        res3 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[idx3]);

#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV_SIZE + 4*lid + 0];
        res1 += compensation[fg*OSV_SIZE + 4*lid + 1];
        res2 += compensation[fg*OSV_SIZE + 4*lid + 2];
        res3 += compensation[fg*OSV_SIZE + 4*lid + 3];
#endif  // ASYMMETRIC_DATA_QUANTIZATION

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) pack;
#if HAS_FUSED_OPS
        { FUSED_OPS_0; pack[0] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; pack[1] = FUSED_OPS_RESULT_1; };
        { FUSED_OPS_2; pack[2] = FUSED_OPS_RESULT_2; };
        { FUSED_OPS_3; pack[3] = FUSED_OPS_RESULT_3; };
#else
        pack[0] = TO_OUTPUT_TYPE(res0);
        pack[1] = TO_OUTPUT_TYPE(res1);
        pack[2] = TO_OUTPUT_TYPE(res2);
        pack[3] = TO_OUTPUT_TYPE(res3);
#endif
        dst[i] = AS_PACKED_OUT_TYPE(pack);
    }

    const bool full_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X;
    const bool full_f = OUTPUT_FEATURE_NUM % OSV_SIZE == 0 || (fg + 1) * OSV_SIZE <= OUTPUT_FEATURE_NUM;
    if (full_x && full_f) {
#if OUTPUT_DIMS == 5
        const uint dst_index = (OUTPUT_GET_INDEX(b, fg*OSV_SIZE, z, y, x)) / 4;
#elif OUTPUT_DIMS <= 4
        const uint dst_index = (OUTPUT_GET_INDEX(b, fg*OSV_SIZE, y, x)) / 4;
#endif
        BLOCK_WRITE(output + dst_index, dst);
    } else {
#if OUTPUT_FEATURE_NUM % 4 == 0
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
            const bool full_it_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
            const bool full_sgl_f = OUTPUT_FEATURE_NUM % OSV_SIZE == 0 || fg * OSV_SIZE + 4 * lid < OUTPUT_FEATURE_NUM;
            if (full_it_x && full_sgl_f) {
#   if OUTPUT_DIMS == 5
                const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + 4*lid, z, y, x+i);
#   elif OUTPUT_DIMS <= 4
                const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + 4*lid, y, x+i);
#   endif
                output[dst_index/4] = dst[i];
            }
        }
#else  // OUTPUT_FEATURE_NUM % 4 == 0
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
            for (int ofm = 0; ofm < 4; ++ofm) {
                const bool full_it_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
                const bool full_sgl_f = OUTPUT_FEATURE_NUM % OSV_SIZE == 0 || fg * OSV_SIZE + 4 * lid + ofm < OUTPUT_FEATURE_NUM;
                if (full_it_x && full_sgl_f) {
#   if OUTPUT_DIMS == 5
                    const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + 4*lid + ofm, z, y, x+i);
#   elif OUTPUT_DIMS <= 4
                    const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV_SIZE + 4*lid + ofm, y, x+i);
#   endif
                    ((__global uchar*)output)[dst_index] = as_uchar4(dst[i])[ofm];
                }
            }
        }
#endif  // OUTPUT_FEATURE_NUM % 4 == 0
    }
#endif  // OUTPUT_IS_FP
}

#undef PACKED_TYPE_VEC
#undef ACCUMULATOR_TYPE_VEC
#undef TO_ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef TO_ACTIVATION_TYPE_VEC
#undef MMAD
