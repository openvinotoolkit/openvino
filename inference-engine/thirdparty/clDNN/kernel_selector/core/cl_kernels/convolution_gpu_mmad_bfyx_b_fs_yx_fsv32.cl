// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"

#include "include/data_types.cl"

#include "include/fetch.cl"
#include "include/imad.cl"

#define CEIL_DIV(x, y) (1 + ((x) - 1) / (y))
#define AS_TYPE(type, val) CAT(as_, type)(val)

#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif

#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif

#if QUANTIZATION_TERM
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
    #define BLOCK_WRITE(ptr, val) intel_sub_group_block_write_us8((__global ushort*)(ptr), as_ushort8(val));
#elif OUTPUT_X_BLOCK_SIZE == 4
    #define PACKED_TYPE_VEC MAKE_VECTOR_TYPE(PACKED_IN_TYPE, 4)
    #define ACCUMULATOR_TYPE_VEC int4
    #define TO_ACCUMULATOR_TYPE_VEC(x) convert_int4(x)
    #define ACTIVATION_TYPE_VEC float4
    #define TO_ACTIVATION_TYPE_VEC(x) convert_float4(x)
    #define BLOCK_WRITE(ptr, val) intel_sub_group_block_write_us4((__global ushort*)(ptr), as_ushort4(val));
#else
#error "convolution_gpu_mmad_bfyx_b_fs_yx_fsv32: Unsupported block size"
#endif

#else // QUANTIZATION_TERM
#error "convolution_gpu_mmad_bfyx_b_fs_yx_fsv32: invalid parameters: quantization term is expected to be true"
#endif

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(convolution_mmad_bfyx_b_fs_yx_fsv32)(
    __global INPUT0_TYPE* input,
    __global PACKED_OUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp,
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp,
    const __global COMPENSATION_TYPE *compensation,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const uint b = get_global_id(2);
    const uint fg = get_group_id(0);
    const uint x = ((uint)get_global_id(1) % CEIL_DIV(OUTPUT_SIZE_X, OUTPUT_X_BLOCK_SIZE)) * OUTPUT_X_BLOCK_SIZE;
    const uint y = (uint)get_global_id(1) / CEIL_DIV(OUTPUT_SIZE_X, OUTPUT_X_BLOCK_SIZE);

    const uint lid = get_sub_group_local_id();

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    ACCUMULATOR_TYPE_VEC acc[2] = { 0 }; // 2*8 packed channels * OUTPUT_X_BLOCK_SIZE
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    ACCUMULATOR_TYPE_VEC acc_assym_weights = 0;
#endif
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    uint filter_idx = fg * FILTER_SIZE_X * FILTER_SIZE_Y * 4 * OSV;

    int in_addr = input_offset + input_x * INPUT0_X_PITCH + input_y * INPUT0_Y_PITCH;
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    const char4 multiplier = (char4)(1, 1, 1, 0);
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
#if ASYMMETRIC_DATA_QUANTIZATION
                bool x_cross_fm = input_x + xb < 0 || input_x + xb >= INPUT0_SIZE_X;
                if (y_cross_fm || x_cross_fm) {
                    const int azp_idx = (4*lid) % ACTIVATIONS_ZERO_POINTS_FEATURE_NUM;
                    char4 zp = as_char4(((const __global uint*)(activations_zp + azp_idx))[0]);
                    zp[3] = 0;
                    line_cache[xb] = AS_PACKED_IN_TYPE(zp);
                }
                else
#endif
                {
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) src = 0;
                    src[0] = input[in_addr + 0 * INPUT0_FEATURE_PITCH
                                           + kh * DILATION_SIZE_Y * INPUT0_Y_PITCH
                                           + xb * INPUT0_X_PITCH];
                    src[1] = input[in_addr + 1 * INPUT0_FEATURE_PITCH
                                           + kh * DILATION_SIZE_Y * INPUT0_Y_PITCH
                                           + xb * INPUT0_X_PITCH];
                    src[2] = input[in_addr + 2 * INPUT0_FEATURE_PITCH
                                           + kh * DILATION_SIZE_Y * INPUT0_Y_PITCH
                                           + xb * INPUT0_X_PITCH];

                    line_cache[xb] = AS_PACKED_IN_TYPE(src);
                }
            }
        }

        __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
        for (uint kw = 0; kw < FILTER_SIZE_X ; ++kw) {
            const uint f_off = filter_idx
                             + kh * OSV * 4 * FILTER_SIZE_X
                             + kw * OSV * 4;

            int weights_data0 = as_int(intel_sub_group_block_read((const __global uint*)(weights + f_off)));
            int weights_data1 = as_int(intel_sub_group_block_read((const __global uint*)(weights + f_off + 16*4)));

            PACKED_TYPE_VEC src;

            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
                src[i] = line_cache[kw*DILATION_SIZE_X + STRIDE_SIZE_X*i];
                acc[0][i] = IMAD(acc[0][i], AS_INPUT0_TYPE_4(src[i]), as_char4(weights_data0));
                acc[1][i] = IMAD(acc[1][i], AS_INPUT0_TYPE_4(src[i]), as_char4(weights_data1));

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                acc_assym_weights[i] = IMAD(acc_assym_weights[i], AS_INPUT0_TYPE_4(src[i]), multiplier);
#endif
            }
        }
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = fg*OSV;
#endif
#endif

#if OUTPUT_IS_FP
    MAKE_VECTOR_TYPE(PACKED_OUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst[2];

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + 2*lid+0]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + 2*lid+1]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + 2 * lid + 0]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + 2 * lid + 1]);
#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV + 2*lid + 0];
        res1 += compensation[fg*OSV + 2*lid + 1];
#endif  // ASYMMETRIC_DATA_QUANTIZATION

#if HAS_FUSED_OPS
        { FUSED_OPS_0; dst[0][i] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; dst[1][i] = FUSED_OPS_RESULT_1; };
#else
        dst[0][i] = TO_OUTPUT_TYPE(res0);
        dst[1][i] = TO_OUTPUT_TYPE(res1);
#endif
    }

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
        for (int ofm = 0; ofm < 2; ofm++) {
            const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + ofm + 2*lid, y, x+i) + out_split_offset;
            if (x + i < OUTPUT_SIZE_X) {
                output[dst_index] = dst[ofm][i];
            }
        }
    }
#else  // OUTPUT_IS_FP
    MAKE_VECTOR_TYPE(PACKED_OUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst;

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + 2*lid+0]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + 2*lid+1]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + 2 * lid + 0]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + 2 * lid + 1]);
#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV + 2*lid + 0];
        res1 += compensation[fg*OSV + 2*lid + 1];
#endif  // ASYMMETRIC_DATA_QUANTIZATION

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) pack;
#if HAS_FUSED_OPS
        { FUSED_OPS_0; pack[0] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; pack[1] = FUSED_OPS_RESULT_1; };
#else
        pack[0] = TO_OUTPUT_TYPE(res0);
        pack[1] = TO_OUTPUT_TYPE(res1);
#endif
        dst[i] = AS_PACKED_OUT_TYPE(pack);
    }

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const bool full_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X;
    const bool full_f = OUTPUT_FEATURE_NUM % OSV == 0 || (fg + 1) * OSV <= OUTPUT_FEATURE_NUM;
    if (full_x && full_f) {
        const uint dst_index = (OUTPUT_GET_INDEX(b, fg*OSV, y, x) + out_split_offset) / 2;
        BLOCK_WRITE(output + dst_index, dst);
    } else {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
            const bool full_it_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
            const bool full_sgl_f = OUTPUT_FEATURE_NUM % OSV == 0 || fg * OSV + 2 * lid < OUTPUT_FEATURE_NUM;
            if (full_it_x && full_sgl_f) {
                const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + 2*lid, y, x+i) + out_split_offset;
                output[dst_index/2] = dst[i];
            }
        }
    }
#endif  // OUTPUT_IS_FP
}
#undef CEIL_DIV
#undef PACKED_TYPE_VEC
#undef ACCUMULATOR_TYPE_VEC
#undef TO_ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef TO_ACTIVATION_TYPE_VEC
#undef MMAD

#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef AS_INPUT0_TYPE_4
