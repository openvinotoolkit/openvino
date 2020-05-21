// Copyright (c) 2020 Intel Corporation
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
#include "include/fetch.cl"
#include "include/imad.cl"
#include "include/mmad.cl"
#include "include/data_types.cl"

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

#define SIMD 16
#define FSV 16

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(1, 1, FEATURE_SLM_SPLIT * SIMD)))
KERNEL(convolution_gpu_b_fs_yx_fsv16_imad)(
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights,
#if BIAS_TERM
    const __global BIAS_TYPE *biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx) {

    #define LUT_VALUE_CLAMP(x) (( (IN_BLOCK_WIDTH % SIMD == 0) || ((x) < IN_BLOCK_WIDTH % SIMD) ) ? (x) : 0)
    const int tmp = LUT_VALUE_CLAMP(get_sub_group_local_id());
    #undef LUT_VALUE_CLAMP

    const uint out_x = (uint)get_global_id(0) * OUT_BLOCK_WIDTH;
    const uint out_y = (uint)get_global_id(1) * OUT_BLOCK_HEIGHT;
    const uint out_b = (uint)(get_group_id(2) * OFM_SIZE_PER_SIMD) / ALIGN(OUTPUT_FEATURE_NUM, OFM_SIZE_PER_SIMD);
    uint out_fg = (uint)(get_group_id(2) * OFM_SIZE_PER_SIMD) % ALIGN(OUTPUT_FEATURE_NUM, OFM_SIZE_PER_SIMD);
    uint out_f = out_fg + get_sub_group_local_id();

    const int input_x = out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

#if FEATURE_SLM_SPLIT == 1
    const uint k_start = 0;
#else
    const uint k_start = get_sub_group_id() * FSV;
#endif

    uint filter_idx  = GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, out_f, k_start, 0, 0);
    const uint filter_idx_diff = (ALIGN(FILTER_IFM_NUM, 16) * FILTER_SIZE_X * FILTER_SIZE_Y * 16);

    uint input_start_idx = INPUT0_GET_INDEX(out_b, k_start, input_y, input_x);

    ACCUMULATOR_TYPE dotProd[OFM_BLOCKS_PER_SIMD][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH] = { };
    uint4 input_val[IN_BLOCK_HEIGHT][CEIL_DIV(IN_BLOCK_WIDTH, SIMD)];

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < CEIL_DIV(INPUT0_FEATURE_NUM, 16) / FEATURE_SLM_SPLIT; k++) {
        __attribute__((opencl_unroll_hint(1)))
        for (uint fyn = 0; fyn < FILTER_SIZE_Y / FILTER_SIZE_Y_UNROLL; fyn++) {
            // Load input block IN_BLOCK_HEIGHT x IN_BLOCK_WIDTH, scattering width along sub-group
            __attribute__((opencl_unroll_hint))
            for (uint iyb = 0; iyb < IN_BLOCK_HEIGHT; ++iyb) {
                __attribute__((opencl_unroll_hint))
                for (uint ixb = 0; ixb < CEIL_DIV(IN_BLOCK_WIDTH, SIMD); ++ixb) {
                    uint input_idx = input_start_idx + iyb * INPUT0_Y_PITCH * FSV + ixb * SIMD * FSV;
                    if (ixb != CEIL_DIV(IN_BLOCK_WIDTH, SIMD) - 1) {
                        input_val[iyb][ixb] = vload4(0, (__global uint *)(conv_input + input_idx + get_sub_group_local_id() * 16));
                    } else {
                        input_val[iyb][ixb] = vload4(0, (__global uint*)(conv_input + input_idx + tmp * 16));
                    }
                }
            }

            __attribute__((opencl_unroll_hint))
            for (uint fyu = 0; fyu < FILTER_SIZE_Y_UNROLL; ++fyu) {
                __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                for (uint fx = 0; fx < FILTER_SIZE_X; fx++) {

                    uint4 weights_val[OFM_BLOCKS_PER_SIMD];
                    __attribute__((opencl_unroll_hint))
                    for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                        weights_val[ofb] = vload4(0, (__global uint *)(weights + filter_idx + ofb * filter_idx_diff));
                    }

                    __attribute__((opencl_unroll_hint))
                    for (uint ive = 0; ive < 4; ive++) {
                        __attribute__((opencl_unroll_hint))
                        for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                            __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                            for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                                __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                                for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {
                                    const uint ow_offset = ow + OUT_BLOCK_WIDTH;
                                    const uint y_block_idx = oh * STRIDE_SIZE_Y + fyu * DILATION_SIZE_Y;
                                    const uint x_block_idx = ow * STRIDE_SIZE_X + fx * DILATION_SIZE_X;
                                    const uint shuffle_wi = x_block_idx % SIMD;
                                    const uint shuffle_idx = x_block_idx / SIMD;

                                    dotProd[ofb][oh][ow] = TO_ACCUMULATOR_TYPE(
                                        IMAD(dotProd[ofb][oh][ow],
                                        AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_val[y_block_idx][shuffle_idx][ive], shuffle_wi)),
                                        AS_FILTER_TYPE_4(weights_val[ofb][ive])));
                                }
                            }
                        }
                    }

                    filter_idx += FSV * FSV;
                }
            }
            input_start_idx += DILATION_SIZE_Y * INPUT0_Y_PITCH * FSV;
        }
        input_start_idx += INPUT0_FEATURE_PITCH * FSV * FEATURE_SLM_SPLIT - (FILTER_SIZE_Y / FILTER_SIZE_Y_UNROLL) * DILATION_SIZE_Y * INPUT0_Y_PITCH * FSV;

        filter_idx += FSV * FSV * FILTER_SIZE_X * FILTER_SIZE_Y * (FEATURE_SLM_SPLIT - 1);
    }

#if FEATURE_SLM_SPLIT != 1
    // Additional local memory reduction for feature split mode
#   if FEATURE_SLM_SPLIT < OFM_BLOCKS_PER_SIMD
#   error convolution_gpu_b_fs_yx_fsv16_imad.cl - OFM_BLOCKS_PER_SIMD must be less or equal to FEATURE_SLM_SPLIT
#   endif

    const uint partial_acc_size = (FEATURE_SLM_SPLIT - 1) * OFM_SIZE_PER_SIMD * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH;
    __local ACCUMULATOR_TYPE partial_acc[partial_acc_size];

    uint sgid_start_idx = get_sub_group_id();
    sgid_start_idx = sgid_start_idx == 0 ? 0 : sgid_start_idx - 1;
    __local ACCUMULATOR_TYPE* partial_acc_ptr = partial_acc + sgid_start_idx * OFM_SIZE_PER_SIMD * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH
                                                              + get_sub_group_local_id();

    if (get_sub_group_id() < OFM_BLOCKS_PER_SIMD) {
        __attribute__((opencl_unroll_hint))
        for (uint wg = 0; wg < OFM_BLOCKS_PER_SIMD; ++wg) {
            if (get_sub_group_id() == wg) {
                __attribute__((opencl_unroll_hint))
                for (uint ofb = 0; ofb < wg; ++ofb) {
                    __attribute__((opencl_unroll_hint))
                    for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                        __attribute__((opencl_unroll_hint))
                        for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                            const uint partial_acc_ptr_idx =
                                ofb * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                oh * OUT_BLOCK_WIDTH * SIMD +
                                ow * SIMD;
                            partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][oh][ow];
                        }
                    }
                }
                __attribute__((opencl_unroll_hint))
                for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                    __attribute__((opencl_unroll_hint))
                    for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                        dotProd[0][oh][ow] = dotProd[wg][oh][ow];
                    }
                }
                __attribute__((opencl_unroll_hint))
                for (uint ofb = wg + 1; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                    __attribute__((opencl_unroll_hint))
                    for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                        __attribute__((opencl_unroll_hint))
                        for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                            const uint partial_acc_ptr_idx =
                                ((wg != 0) ? OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * OFM_SIZE_PER_SIMD : 0) +
                                ofb * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                oh * OUT_BLOCK_WIDTH * SIMD +
                                ow * SIMD;
                            partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][oh][ow];
                        }
                    }
                }
            }
        }
    } else {
        __attribute__((opencl_unroll_hint))
        for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
            __attribute__((opencl_unroll_hint))
            for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                __attribute__((opencl_unroll_hint))
                for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                    const uint partial_acc_ptr_idx =
                        ofb * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                        oh * OUT_BLOCK_WIDTH * SIMD +
                        ow * SIMD;
                    partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][oh][ow];
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_id() >= OFM_BLOCKS_PER_SIMD)
        return;

    partial_acc_ptr = partial_acc + get_sub_group_id() * OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * SIMD + get_sub_group_local_id();
    __attribute__((opencl_unroll_hint))
    for (uint wg = 0; wg < FEATURE_SLM_SPLIT - 1; ++wg) {
        __attribute__((opencl_unroll_hint))
        for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
            __attribute__((opencl_unroll_hint))
            for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                const uint partial_acc_ptr_idx =
                    wg * OFM_SIZE_PER_SIMD * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH +
                    oh * OUT_BLOCK_WIDTH * SIMD +
                    ow * SIMD;
                dotProd[0][oh][ow] += partial_acc_ptr[partial_acc_ptr_idx];
            }
        }
    }
#endif

#if FEATURE_SLM_SPLIT == 1
#   define OFM_VALUES_PER_WI (OFM_BLOCKS_PER_SIMD)
#else
#   define OFM_VALUES_PER_WI 1
    out_f += get_sub_group_id() * SIMD;
    out_fg += get_sub_group_id() * SIMD;
#endif

#if BIAS_TERM
    BIAS_TYPE bias[OFM_VALUES_PER_WI];
    __attribute__((opencl_unroll_hint))
    for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
        bias[ofb] = biases[out_f + ofb * SIMD];
    }
#endif

    ACTIVATION_TYPE dequantized[OFM_VALUES_PER_WI][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH];
    __attribute__((opencl_unroll_hint))
    for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
        __attribute__((opencl_unroll_hint))
        for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
            __attribute__((opencl_unroll_hint))
            for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                dequantized[ofb][oh][ow] = TO_ACTIVATION_TYPE(dotProd[ofb][oh][ow]);
#if BIAS_TERM
                dequantized[ofb][oh][ow] += bias[ofb];
#endif
            }
        }
    }

    OUTPUT_TYPE result[OFM_VALUES_PER_WI][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH];
    __attribute__((opencl_unroll_hint))
    for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD_SCALAR
        FUSED_OPS_PRELOAD_SCALAR;
#endif
        __attribute__((opencl_unroll_hint))
        for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
            __attribute__((opencl_unroll_hint))
            for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                ACTIVATION_TYPE dequantized_val = dequantized[ofb][oh][ow];
#if HAS_FUSED_OPS
#   if FUSED_OPS_CAN_USE_PRELOAD_SCALAR
                FUSED_OPS_CALC_SCALAR;
#   else
                FUSED_OPS_SCALAR;
#   endif
                result[ofb][oh][ow] = FUSED_OPS_RESULT_SCALAR;
#else
                result[ofb][oh][ow] = TO_OUTPUT_TYPE(dequantized_val);
#endif
            }
        }
    }

    uint dst_index = OUTPUT_GET_INDEX(out_b, out_fg, out_y, out_x);

    if ((OUTPUT_SIZE_X % OUT_BLOCK_WIDTH == 0 || out_x + OUT_BLOCK_WIDTH <= OUTPUT_SIZE_X)
        && (OUTPUT_FEATURE_NUM % OFM_BLOCKS_PER_SIMD == 0) ) {
        __attribute__((opencl_unroll_hint(OFM_VALUES_PER_WI)))
        for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ofb++) {
            bool good_of_block = (CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % OFM_BLOCKS_PER_SIMD == 0) || (out_fg + ofb * SIMD <= OUTPUT_FEATURE_NUM);
            if (good_of_block) {
                __attribute__((opencl_unroll_hint))
                for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                    bool good_y = (OUTPUT_SIZE_Y % OUT_BLOCK_HEIGHT == 0) || (out_y + oh < OUTPUT_SIZE_Y);
                    if (good_y) {
                        uint ow = 0;
                    #if OUTPUT_TYPE_SIZE == 1
                        __attribute__((opencl_unroll_hint))
                        for (; ow + 8 <= OUT_BLOCK_WIDTH; ow += 8) {
                            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) result_val;
                            __attribute__((opencl_unroll_hint))
                            for (uint i = 0; i < 8; ++i) {
                                result_val[i] = result[ofb][oh][ow + i];
                            }
                            DT_OUTPUT_BLOCK_WRITE8(output, dst_index, result_val);
                            dst_index += 8 * SIMD;
                        }
                    #endif
                    #if OUTPUT_TYPE_SIZE <= 2
                        __attribute__((opencl_unroll_hint))
                        for (; ow + 4 <= OUT_BLOCK_WIDTH; ow += 4) {
                            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) result_val;
                            __attribute__((opencl_unroll_hint))
                            for (uint i = 0; i < 4; ++i) {
                                result_val[i] = result[ofb][oh][ow + i];
                            }
                            DT_OUTPUT_BLOCK_WRITE4(output, dst_index, result_val);
                            dst_index += 4 * SIMD;
                        }
                    #endif

                        __attribute__((opencl_unroll_hint))
                        for (; ow + 2 <= OUT_BLOCK_WIDTH; ow += 2) {
                            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) result_val;
                            __attribute__((opencl_unroll_hint))
                            for (uint i = 0; i < 2; ++i) {
                                result_val[i] = result[ofb][oh][ow + i];
                            }
                            DT_OUTPUT_BLOCK_WRITE2(output, dst_index, result_val);
                            dst_index += 2 * SIMD;
                        }

                        if (OUT_BLOCK_WIDTH % 2 == 1) {
                            OUTPUT_TYPE result_val = result[ofb][oh][ow];
                            DT_OUTPUT_BLOCK_WRITE(output, dst_index, result_val);
                            dst_index += 1 * SIMD;
                        }
                    }  // if (good_y)
                    dst_index += OUTPUT_Y_PITCH * FSV - OUT_BLOCK_WIDTH * FSV;
                }  // for (OUT_BLOCK_HEIGHT)
            }  // if (good_of_block)
            dst_index += OUTPUT_FEATURE_PITCH * FSV - OUTPUT_Y_PITCH * FSV * OUT_BLOCK_HEIGHT;
        }  // for (OFM_VALUES_PER_WI)
    } else {
        __attribute__((opencl_unroll_hint(OFM_VALUES_PER_WI)))
        for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ofb++) {
            bool good_of_block = (CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % OFM_BLOCKS_PER_SIMD == 0) || (out_fg + ofb * SIMD <= OUTPUT_FEATURE_NUM);
            if (good_of_block) {
                const uint dst_index = OUTPUT_GET_INDEX(out_b, out_f + ofb * SIMD, out_y, out_x);
                __attribute__((opencl_unroll_hint))
                for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                    bool good_y = (OUTPUT_SIZE_Y % OUT_BLOCK_HEIGHT == 0) || (out_y + oh < OUTPUT_SIZE_Y);
                    if (good_y) {
                        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                        for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {

#if OUTPUT_SIZE_X % OUT_BLOCK_WIDTH != 0
                            if (out_x + OUT_BLOCK_WIDTH > OUTPUT_SIZE_X && ow >= OUTPUT_SIZE_X % OUT_BLOCK_WIDTH)
                                break;
#endif

#if OUTPUT_FEATURE_NUM % SIMD != 0
                            if (out_fg + (ofb + 1) * SIMD >= OUTPUT_FEATURE_NUM && get_sub_group_local_id() >= OUTPUT_FEATURE_NUM % SIMD)
                                result[ofb][oh][ow] = (OUTPUT_TYPE)0;
#endif
                            output[dst_index + ow * FSV + oh * OUTPUT_Y_PITCH * FSV] = result[ofb][oh][ow];
                        }
                    }
                }
            }
        }
    }
}

#undef AS_INPUT0_TYPE_4
#undef AS_TYPE_N
#undef AS_TYPE_N_
#undef AS_FILTER_TYPE_4

#undef CEIL_DIV
#undef ALIGN

#undef SIMD
#undef FSV
#undef OFM_VALUES_PER_WI
