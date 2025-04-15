// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#define TYPE_N_(type, n) type##n
#define TYPE_N(type, n) TYPE_N_(type, n)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define INPUT0_TYPE_4 TYPE_N(INPUT0_TYPE, 4)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

#if INPUT0_PAD_BEFORE_SIZE_X != 0 || \
    INPUT0_PAD_BEFORE_SIZE_Y != 0 || \
    INPUT0_PAD_BEFORE_SIZE_Z != 0
    #define NON_ZERO_INPUT0_PAD_BEFORE
#endif

#if !defined COMPENSATION_TERM || \
    (defined COMPENSATION_TERM && defined NON_ZERO_INPUT0_PAD_BEFORE)
    #define SHOULD_BALANCE_COMPENSATION
#endif

#if defined ASYMMETRIC_DATA_QUANTIZATION && defined SHOULD_BALANCE_COMPENSATION
    #define SHOULD_USE_DATA_ZP
#endif

#if defined ASYMMETRIC_DATA_QUANTIZATION && \
    defined ASYMMETRIC_WEIGHTS_QUANTIZATION && \
    defined SHOULD_BALANCE_COMPENSATION
    #define SHOULD_USE_DATA_AND_WEIGHTS_ZP
#endif

#ifdef SHOULD_USE_DATA_AND_WEIGHTS_ZP
    #define ACCUMULATOR_TYPE_4 TYPE_N(ACCUMULATOR_TYPE, 4)
#endif

#ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
    #define FILTER_TYPE_16 TYPE_N(FILTER_TYPE, 16)
#endif

#define AS_FILTER_TYPE_4(x) AS_TYPE_N(FILTER_TYPE, 4, x)

#define SIMD 16
#define FSV 16

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(1, 1, FEATURE_SLM_SPLIT * SIMD)))
KERNEL(convolution_gpu_b_fs_zyx_fsv16_imad)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE *conv_input,
    __global OUTPUT_TYPE *output,
    const __global FILTER_TYPE *weights
#if BIAS_TERM
    , const __global BIAS_TYPE *biases
#endif
#ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#ifdef ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
#endif
#ifdef COMPENSATION_TERM
    , const __global COMPENSATION_TYPE *compensation
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {

    #define LUT_VALUE_CLAMP(x) (( (IN_BLOCK_WIDTH % SIMD == 0) || ((x) < IN_BLOCK_WIDTH % SIMD) ) ? (x) : 0)
    const int tmp = LUT_VALUE_CLAMP(get_sub_group_local_id());
    #undef LUT_VALUE_CLAMP

    const uint out_x = (uint)get_global_id(0) * OUT_BLOCK_WIDTH;
    const uint out_y = ((uint)get_global_id(1) / ALIGN(OUTPUT_SIZE_Z, OUT_BLOCK_DEPTH)) * OUT_BLOCK_HEIGHT;
#if INPUT0_DIMS == 4
    const uint out_z = 0;
#else
    const uint out_z = ((uint)get_global_id(1) % ALIGN(OUTPUT_SIZE_Z, OUT_BLOCK_DEPTH)) * OUT_BLOCK_DEPTH;
#endif
    const uint out_b = (uint)(get_group_id(2) / CEIL_DIV(FILTER_OFM_NUM, OFM_SIZE_PER_SIMD)) / FILTER_GROUPS_NUM;
    const uint g = (uint)(get_group_id(2) / CEIL_DIV(FILTER_OFM_NUM, OFM_SIZE_PER_SIMD)) % FILTER_GROUPS_NUM;
    uint out_f_sg = (uint)(get_group_id(2) * OFM_SIZE_PER_SIMD) % (ALIGN(FILTER_OFM_NUM, OFM_SIZE_PER_SIMD) * FILTER_GROUPS_NUM);
    uint out_f = out_f_sg + get_sub_group_local_id();
    uint out_f_g = (out_f % ALIGN(FILTER_OFM_NUM, OFM_SIZE_PER_SIMD));
#if FILTER_OFM_NUM % SIMD != 0
    out_f = out_f - (out_f / ALIGN(FILTER_OFM_NUM, SIMD)) * (SIMD - (FILTER_OFM_NUM % SIMD));
#endif

    const int input_x = out_x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = out_y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int input_z = out_z * STRIDE_SIZE_Z - PADDING_SIZE_Z;

#if FEATURE_SLM_SPLIT == 1
    const uint in_f_start = 0;
#else
    const uint in_f_start = get_sub_group_id() * FSV;
#endif

    uint filter_idx  = GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(FILTER, g, out_f_g, in_f_start, 0, 0, 0);
    const uint filter_idx_diff = (ALIGN(FILTER_IFM_NUM, FSV) * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z * FSV);

#if INPUT0_DIMS == 4
    uint input_start_idx = INPUT0_GET_INDEX(out_b, g * FILTER_IFM_NUM + in_f_start, input_y, input_x);
#else
    uint input_start_idx = INPUT0_GET_INDEX(out_b, g * FILTER_IFM_NUM + in_f_start, input_z, input_y, input_x);
#endif

    ACCUMULATOR_TYPE dotProd[OFM_BLOCKS_PER_SIMD][OUT_BLOCK_DEPTH][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH] = { };
#if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
    uint in_f_offset = (g * FILTER_IFM_NUM) % FSV;
#endif

    uint4 input_val[IN_BLOCK_DEPTH][IN_BLOCK_HEIGHT][CEIL_DIV(IN_BLOCK_WIDTH, SIMD)];

#ifdef SHOULD_USE_DATA_ZP
    uint data_zp_idx = g * FILTER_IFM_NUM + in_f_start;
    uint4 data_zp_val;
#endif

#ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
    uint4 weights_zp_val[OFM_BLOCKS_PER_SIMD];
    unroll_for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
        weights_zp_val[ofb] = as_uint4((FILTER_TYPE_16)weights_zp[out_f + ofb * FSV]);
    }
    #if FILTER_IFM_NUM % FSV != 0
        uint4 weights_zp_vec_partial[OFM_BLOCKS_PER_SIMD];
        unroll_for(uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
            weights_zp_vec_partial[ofb] = weights_zp_val[ofb];
            FILTER_TYPE* wzp_p = (FILTER_TYPE*)&weights_zp_vec_partial[ofb];
            unroll_for(uint f = FILTER_IFM_NUM % FSV; f < FSV; f++) {
                wzp_p[f] = 0;
            }
        }
    #endif
#endif

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < CEIL_DIV(FILTER_IFM_NUM, FSV) / FEATURE_SLM_SPLIT; k++) {
        #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
            #if FILTER_IFM_NUM % FSV != 0
                if (in_f_start + (k + 1) * FSV >= ALIGN(FILTER_IFM_NUM, FSV)) {
                    unroll_for(uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                        weights_zp_val[ofb] = weights_zp_vec_partial[ofb];
                    }
                }
            #endif
        #endif

        #ifdef SHOULD_USE_DATA_ZP
            #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
                data_zp_val = as_uint4(vload16(0, activations_zp + data_zp_idx));
            #else
                data_zp_val = vload4(0, (__global uint *)(activations_zp + data_zp_idx));
            #endif
        #endif

        #ifdef SHOULD_USE_DATA_AND_WEIGHTS_ZP
            ACCUMULATOR_TYPE_4 dotProdAZPxWZP[OFM_BLOCKS_PER_SIMD];
            unroll_for(uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                dotProdAZPxWZP[ofb] = 0;
                unroll_for(uint ive = 0; ive < 4; ive++) {
                    dotProdAZPxWZP[ofb][ive] = TO_ACCUMULATOR_TYPE(
                    IMAD(dotProdAZPxWZP[ofb][ive],
                    AS_INPUT0_TYPE_4(data_zp_val[ive]),
                    AS_FILTER_TYPE_4(weights_zp_val[ofb][ive])));
                }
            }
        #endif

        __attribute__((opencl_unroll_hint(1)))
        for (uint fzn = 0; fzn < FILTER_SIZE_Z / FILTER_SIZE_Z_UNROLL; fzn++) {
            __attribute__((opencl_unroll_hint(1)))
            for (uint fyn = 0; fyn < FILTER_SIZE_Y / FILTER_SIZE_Y_UNROLL; fyn++) {
                // Load input block IN_BLOCK_DEPTH x IN_BLOCK_HEIGHT x IN_BLOCK_WIDTH, scattering width along sub-group
                unroll_for(uint izb = 0; izb < IN_BLOCK_DEPTH; ++izb) {
                    unroll_for(uint iyb = 0; iyb < IN_BLOCK_HEIGHT; ++iyb) {
                        unroll_for(uint ixb = 0; ixb < CEIL_DIV(IN_BLOCK_WIDTH, SIMD); ++ixb) {
                            uint input_idx = input_start_idx + izb * INPUT0_Z_PITCH * FSV + iyb * INPUT0_Y_PITCH * FSV + ixb * SIMD * FSV;
                            #ifdef SHOULD_USE_DATA_ZP
                                const int y_idx = input_y + fyn * DILATION_SIZE_Y + iyb;
                                const int z_idx = input_z + fzn * DILATION_SIZE_Z + izb;
                            #endif
                            if (ixb != CEIL_DIV(IN_BLOCK_WIDTH, SIMD) - 1) {
                                #ifdef SHOULD_USE_DATA_ZP
                                    const int x_idx = input_x + ixb * SIMD + get_sub_group_local_id();
                                    const bool input_on_padding = (((x_idx < 0) || (x_idx >= INPUT0_SIZE_X)) ||
                                                                   ((y_idx < 0) || (y_idx >= INPUT0_SIZE_Y)) ||
                                                                   ((z_idx < 0) || (z_idx >= INPUT0_SIZE_Z)));
                                #endif

                                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
                                if (in_f_offset == 0) {
                                #endif
                                    #ifdef SHOULD_USE_DATA_ZP
                                        if (input_on_padding) {
                                            input_val[izb][iyb][ixb] = data_zp_val;
                                        } else {
                                    #endif
                                            input_val[izb][iyb][ixb] = vload4(0, (__global uint *)(conv_input + input_idx + get_sub_group_local_id() * FSV));
                                    #ifdef SHOULD_USE_DATA_ZP
                                        }
                                    #endif
                                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
                                } else {
                                    INPUT0_TYPE* input_int8_arr = (INPUT0_TYPE*) &input_val[izb][iyb][ixb];
                                    #ifdef SHOULD_USE_DATA_ZP
                                        INPUT0_TYPE* input_zp_int8_arr = (INPUT0_TYPE*) &data_zp_val;
                                    #endif
                                    __attribute__((opencl_unroll_hint(FSV)))
                                    for (uint v = 0; v < FSV; v++) {
                                        #ifdef SHOULD_USE_DATA_ZP
                                            if (input_on_padding) {
                                                input_int8_arr[v] = input_zp_int8_arr[v];
                                            } else {
                                        #endif
                                                if (v + in_f_offset < FSV) {
                                                    input_int8_arr[v] = conv_input[input_idx + get_sub_group_local_id() * FSV + v];
                                                } else {
                                                    const uint addr = input_idx + get_sub_group_local_id() * FSV + v +
                                                                ((INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X) *
                                                                 (INPUT0_SIZE_Y + INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y) *
                                                                 (INPUT0_SIZE_Z + INPUT0_PAD_BEFORE_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z) - 1) * FSV;
                                                    input_int8_arr[v] = conv_input[addr];
                                                }
                                        #ifdef SHOULD_USE_DATA_ZP
                                            }
                                        #endif
                                    }
                                }
                                #endif
                            } else {
                                #ifdef SHOULD_USE_DATA_ZP
                                    const int x_idx = input_x + ixb * SIMD + tmp;
                                    const bool input_on_padding = (((x_idx < 0) || (x_idx >= INPUT0_SIZE_X)) ||
                                                                   ((y_idx < 0) || (y_idx >= INPUT0_SIZE_Y)) ||
                                                                   ((z_idx < 0) || (z_idx >= INPUT0_SIZE_Z)));
                                #endif

                                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
                                if (in_f_offset == 0) {
                                #endif
                                    #ifdef SHOULD_USE_DATA_ZP
                                        if (input_on_padding) {
                                            input_val[izb][iyb][ixb] = data_zp_val;
                                        } else {
                                    #endif
                                            input_val[izb][iyb][ixb] = vload4(0, (__global uint *)(conv_input + input_idx + tmp * FSV));
                                    #ifdef SHOULD_USE_DATA_ZP
                                        }
                                    #endif
                                #if ((FILTER_GROUPS_NUM > 1) && (FILTER_IFM_NUM % FSV != 0))
                                } else {
                                    INPUT0_TYPE* input_int8_arr = (INPUT0_TYPE*) &input_val[izb][iyb][ixb];
                                    #ifdef SHOULD_USE_DATA_ZP
                                        INPUT0_TYPE* input_zp_int8_arr = (INPUT0_TYPE*) &data_zp_val;
                                    #endif
                                    __attribute__((opencl_unroll_hint(FSV)))
                                    for (uint v = 0; v < FSV; v++) {
                                        #ifdef SHOULD_USE_DATA_ZP
                                            if (input_on_padding) {
                                                input_int8_arr[v] = input_zp_int8_arr[v];
                                            } else {
                                            #endif
                                                if (v + in_f_offset < FSV) {
                                                    input_int8_arr[v] = conv_input[input_idx + tmp * FSV + v];
                                                } else {
                                                    const uint addr = input_idx + tmp * FSV + v +
                                                                ((INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X) *
                                                                    (INPUT0_SIZE_Y + INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y) *
                                                                    (INPUT0_SIZE_Z + INPUT0_PAD_BEFORE_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z) - 1) * FSV;
                                                    input_int8_arr[v] = conv_input[addr];
                                                }
                                        #ifdef SHOULD_USE_DATA_ZP
                                            }
                                        #endif
                                    }
                                }
                                #endif
                            }
                        }
                    }
                }

                sub_group_barrier(CLK_LOCAL_MEM_FENCE);

                unroll_for(uint fzu = 0; fzu < FILTER_SIZE_Z_UNROLL; ++fzu) {
                    unroll_for(uint fyu = 0; fyu < FILTER_SIZE_Y_UNROLL; ++fyu) {
                        unroll_for (uint fx = 0; fx < FILTER_SIZE_X; fx++) {

                            uint4 weights_val[OFM_BLOCKS_PER_SIMD];
                            unroll_for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                                weights_val[ofb] = vload4(0, (__global uint *)(weights + filter_idx + ofb * filter_idx_diff));
                            }

                            unroll_for (uint ive = 0; ive < 4; ive++) {
                                unroll_for (uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                                    #ifdef SHOULD_USE_DATA_ZP
                                        ACCUMULATOR_TYPE dotProdAZPxW = 0;
                                        dotProdAZPxW = TO_ACCUMULATOR_TYPE(
                                        IMAD(dotProdAZPxW,
                                        AS_INPUT0_TYPE_4(data_zp_val[ive]),
                                        AS_FILTER_TYPE_4(weights_val[ofb][ive])));
                                    #endif

                                    unroll_for (uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                                        unroll_for (uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                                            unroll_for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {
                                                const uint z_block_idx = od * STRIDE_SIZE_Z + fzu * DILATION_SIZE_Z;
                                                const uint y_block_idx = oh * STRIDE_SIZE_Y + fyu * DILATION_SIZE_Y;
                                                const uint x_block_idx = ow * STRIDE_SIZE_X + fx * DILATION_SIZE_X;
                                                const uint shuffle_wi = x_block_idx % SIMD;
                                                const uint shuffle_idx = x_block_idx / SIMD;

                                                INPUT0_TYPE_4 inputs = AS_INPUT0_TYPE_4(_sub_group_shuffle(input_val[z_block_idx][y_block_idx][shuffle_idx][ive],
                                                    shuffle_wi));

                                                dotProd[ofb][od][oh][ow] = TO_ACCUMULATOR_TYPE(
                                                    IMAD(dotProd[ofb][od][oh][ow],
                                                    inputs,
                                                    AS_FILTER_TYPE_4(weights_val[ofb][ive])));

                                                #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
                                                    ACCUMULATOR_TYPE dotProdAxWZP = 0;
                                                    dotProdAxWZP = TO_ACCUMULATOR_TYPE(
                                                    IMAD(dotProdAxWZP,
                                                    inputs,
                                                    AS_FILTER_TYPE_4(weights_zp_val[ofb][ive])));
                                                    dotProd[ofb][od][oh][ow] -= dotProdAxWZP;
                                                #endif

                                                #if !defined COMPENSATION_TERM && defined ASYMMETRIC_DATA_QUANTIZATION
                                                    dotProd[ofb][od][oh][ow] -= dotProdAZPxW;
                                                #endif

                                                #if (!defined COMPENSATION_TERM && \
                                                        defined ASYMMETRIC_DATA_QUANTIZATION && \
                                                        defined ASYMMETRIC_WEIGHTS_QUANTIZATION)
                                                    dotProd[ofb][od][oh][ow] += dotProdAZPxWZP[ofb][ive];
                                                #endif
                                            }
                                        }
                                    }
                                }
                            }

                            filter_idx += FSV * FSV;
                        }
                    }
                }
                input_start_idx += DILATION_SIZE_Y * INPUT0_Y_PITCH * FSV;
            }
            input_start_idx += DILATION_SIZE_Z * INPUT0_Z_PITCH * FSV - (FILTER_SIZE_Y / FILTER_SIZE_Y_UNROLL) * DILATION_SIZE_Y * INPUT0_Y_PITCH * FSV;
        }
        input_start_idx += INPUT0_FEATURE_PITCH * FSV * FEATURE_SLM_SPLIT - (FILTER_SIZE_Z / FILTER_SIZE_Z_UNROLL) * DILATION_SIZE_Z * INPUT0_Z_PITCH * FSV;

        filter_idx += FSV * FSV * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z * (FEATURE_SLM_SPLIT - 1);

        #ifdef SHOULD_USE_DATA_ZP
            data_zp_idx += FSV;
        #endif
    }

#if FEATURE_SLM_SPLIT != 1
    // Additional local memory reduction for feature split mode
#   if FEATURE_SLM_SPLIT < OFM_BLOCKS_PER_SIMD
#   error convolution_gpu_b_fs_zyx_fsv16_imad.cl - OFM_BLOCKS_PER_SIMD must be less or equal to FEATURE_SLM_SPLIT
#   endif

    const uint partial_acc_size = (FEATURE_SLM_SPLIT - 1) * OFM_SIZE_PER_SIMD * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH;
    __local ACCUMULATOR_TYPE partial_acc[partial_acc_size];

    uint sgid_start_idx = get_sub_group_id();
    sgid_start_idx = sgid_start_idx == 0 ? 0 : sgid_start_idx - 1;
    __local ACCUMULATOR_TYPE* partial_acc_ptr = partial_acc + sgid_start_idx * OFM_SIZE_PER_SIMD * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH +
                                                get_sub_group_local_id();

    if (get_sub_group_id() < OFM_BLOCKS_PER_SIMD) {
        unroll_for(uint wg = 0; wg < OFM_BLOCKS_PER_SIMD; ++wg) {
            if (get_sub_group_id() == wg) {
                unroll_for(uint ofb = 0; ofb < wg; ++ofb) {
                    unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                        unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                            unroll_for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                                const uint partial_acc_ptr_idx =
                                    ofb * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                    od * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                    oh * OUT_BLOCK_WIDTH * SIMD +
                                    ow * SIMD;
                                partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][od][oh][ow];
                            }
                        }
                    }
                }
                unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                    unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                        unroll_for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                            dotProd[0][od][oh][ow] = dotProd[wg][od][oh][ow];
                        }
                    }
                }
                unroll_for(uint ofb = wg + 1; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
                    unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                        unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                            unroll_for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                                const uint partial_acc_ptr_idx =
                                    ((wg != 0) ? OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_DEPTH * OFM_SIZE_PER_SIMD : 0) +
                                    ofb * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                    od * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                                    oh * OUT_BLOCK_WIDTH * SIMD +
                                    ow * SIMD;
                                partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][od][oh][ow];
                            }
                        }
                    }
                }
            }
        }
    } else {
        unroll_for(uint ofb = 0; ofb < OFM_BLOCKS_PER_SIMD; ++ofb) {
            unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                    unroll_for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                        const uint partial_acc_ptr_idx =
                            ofb * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                            od * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                            oh * OUT_BLOCK_WIDTH * SIMD +
                            ow * SIMD;
                        partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][od][oh][ow];
                    }
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_id() >= OFM_BLOCKS_PER_SIMD)
        return;

    partial_acc_ptr = partial_acc + get_sub_group_id() * OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_DEPTH * SIMD + get_sub_group_local_id();
    unroll_for (uint wg = 0; wg < FEATURE_SLM_SPLIT - 1; ++wg) {
        unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
            unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                unroll_for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                    const uint partial_acc_ptr_idx =
                        wg * OFM_SIZE_PER_SIMD * OUT_BLOCK_DEPTH * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH +
                        od * OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH * SIMD +
                        oh * OUT_BLOCK_WIDTH * SIMD +
                        ow * SIMD;
                    dotProd[0][od][oh][ow] += partial_acc_ptr[partial_acc_ptr_idx];
                }
            }
        }
    }
#endif

#if FEATURE_SLM_SPLIT == 1
#   define OFM_VALUES_PER_WI (OFM_BLOCKS_PER_SIMD)
#else
#   define OFM_VALUES_PER_WI 1
    out_f += get_sub_group_id() * SIMD;
    out_f_sg += get_sub_group_id() * SIMD;
#endif

#if BIAS_TERM
    BIAS_TYPE bias[OFM_VALUES_PER_WI];
    unroll_for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
        bias[ofb] = biases[out_f + ofb * SIMD];
    }
#endif

#ifdef COMPENSATION_TERM
    COMPENSATION_TYPE comp[OFM_VALUES_PER_WI];
    unroll_for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
        comp[ofb] = compensation[out_f + ofb * SIMD];
    }
#endif

    ACTIVATION_TYPE dequantized[OFM_VALUES_PER_WI][OUT_BLOCK_DEPTH][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH];
    unroll_for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
        unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
            unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                unroll_for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                    dequantized[ofb][od][oh][ow] = TO_ACTIVATION_TYPE(dotProd[ofb][od][oh][ow]);
#if BIAS_TERM
                    dequantized[ofb][od][oh][ow] += bias[ofb];
#endif
#ifdef COMPENSATION_TERM
                    dequantized[ofb][od][oh][ow] += comp[ofb];
#endif
                }
            }
        }
    }

    OUTPUT_TYPE result[OFM_VALUES_PER_WI][OUT_BLOCK_DEPTH][OUT_BLOCK_HEIGHT][OUT_BLOCK_WIDTH];
    unroll_for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ++ofb) {
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD_SCALAR
        FUSED_OPS_PRELOAD_SCALAR;
#endif
        unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
            unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                unroll_for(uint ow = 0; ow < OUT_BLOCK_WIDTH; ++ow) {
                    ACTIVATION_TYPE dequantized_val = dequantized[ofb][od][oh][ow];
#if HAS_FUSED_OPS
#   if FUSED_OPS_CAN_USE_PRELOAD_SCALAR
                    FUSED_OPS_CALC_SCALAR;
#   else
                    FUSED_OPS_SCALAR;
#   endif
                    result[ofb][od][oh][ow] = FUSED_OPS_RESULT_SCALAR;
#else
                    result[ofb][od][oh][ow] = TO_OUTPUT_TYPE(dequantized_val);
#endif
                }
            }
        }
    }

#if OUTPUT_DIMS == 4
    uint dst_index = OUTPUT_GET_INDEX(out_b, out_f_sg, out_y, out_x);
#else
    uint dst_index = OUTPUT_GET_INDEX(out_b, out_f_sg, out_z, out_y, out_x);
#endif

#if ((FILTER_OFM_NUM % OFM_BLOCKS_PER_SIMD == 0) && ((FILTER_GROUPS_NUM == 1) || (FILTER_OFM_NUM % SIMD == 0)))
    if ((OUTPUT_SIZE_X % OUT_BLOCK_WIDTH == 0 || out_x + OUT_BLOCK_WIDTH <= OUTPUT_SIZE_X)) {
        __attribute__((opencl_unroll_hint(OFM_VALUES_PER_WI)))
        for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ofb++) {
            bool good_of_block = (CEIL_DIV(FILTER_OFM_NUM, SIMD) % OFM_BLOCKS_PER_SIMD == 0) || (out_f_sg + ofb * SIMD <= FILTER_OFM_NUM);
            if (good_of_block) {
                unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                    bool good_z = (OUTPUT_SIZE_Z % OUT_BLOCK_DEPTH == 0) || (out_z + od < OUTPUT_SIZE_Z);
                    if (good_z) {
                        unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                            bool good_y = (OUTPUT_SIZE_Y % OUT_BLOCK_HEIGHT == 0) || (out_y + oh < OUTPUT_SIZE_Y);
                            if (good_y) {
                                uint ow = 0;
                            #if OUTPUT_TYPE_SIZE == 1
                                unroll_for (; ow + 8 <= OUT_BLOCK_WIDTH; ow += 8) {
                                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) result_val;
                                    unroll_for (uint i = 0; i < 8; ++i) {
                                        result_val[i] = result[ofb][od][oh][ow + i];
                                    }
                                    DT_OUTPUT_BLOCK_WRITE8(output, dst_index, result_val);
                                    dst_index += 8 * SIMD;
                                }
                            #endif
                            #if OUTPUT_TYPE_SIZE <= 2
                                unroll_for (; ow + 4 <= OUT_BLOCK_WIDTH; ow += 4) {
                                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) result_val;
                                    unroll_for (uint i = 0; i < 4; ++i) {
                                        result_val[i] = result[ofb][od][oh][ow + i];
                                    }
                                    DT_OUTPUT_BLOCK_WRITE4(output, dst_index, result_val);
                                    dst_index += 4 * SIMD;
                                }
                            #endif

                                unroll_for (; ow + 2 <= OUT_BLOCK_WIDTH; ow += 2) {
                                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) result_val;
                                    unroll_for (uint i = 0; i < 2; ++i) {
                                        result_val[i] = result[ofb][od][oh][ow + i];
                                    }
                                    DT_OUTPUT_BLOCK_WRITE2(output, dst_index, result_val);
                                    dst_index += 2 * SIMD;
                                }

                                if (OUT_BLOCK_WIDTH % 2 == 1) {
                                    OUTPUT_TYPE result_val = result[ofb][od][oh][ow];
                                    DT_OUTPUT_BLOCK_WRITE(output, dst_index, result_val);
                                    dst_index += 1 * SIMD;
                                }
                            }  // if (good_y)
                            dst_index += OUTPUT_Y_PITCH * FSV - OUT_BLOCK_WIDTH * FSV;
                        }  // for (OUT_BLOCK_HEIGHT)
                    }  // if (good_z)
                    dst_index += OUTPUT_Z_PITCH * FSV - OUTPUT_Y_PITCH * OUT_BLOCK_HEIGHT * FSV;
                }  // for (OUT_BLOCK_DEPTH)
            }  // if (good_of_block)
            dst_index += OUTPUT_FEATURE_PITCH * FSV - OUTPUT_Z_PITCH * OUT_BLOCK_DEPTH * FSV;
        }  // for (OFM_VALUES_PER_WI)
    } else {
#endif
        __attribute__((opencl_unroll_hint(OFM_VALUES_PER_WI)))
        for (uint ofb = 0; ofb < OFM_VALUES_PER_WI; ofb++) {
            bool good_of_block = (CEIL_DIV(FILTER_OFM_NUM, SIMD) % OFM_BLOCKS_PER_SIMD == 0) || (out_f_sg + ofb * SIMD <= FILTER_OFM_NUM);
            if (good_of_block) {
        #if OUTPUT_DIMS == 4
                const uint dst_index = OUTPUT_GET_INDEX(out_b, out_f + ofb * SIMD, out_y, out_x);
        #else
                const uint dst_index = OUTPUT_GET_INDEX(out_b, out_f + ofb * SIMD, out_z, out_y, out_x);
        #endif
                unroll_for(uint od = 0; od < OUT_BLOCK_DEPTH; ++od) {
                    bool good_z = (OUTPUT_SIZE_Z % OUT_BLOCK_DEPTH == 0) || (out_z + od < OUTPUT_SIZE_Z);
                    if (good_z) {
                        unroll_for(uint oh = 0; oh < OUT_BLOCK_HEIGHT; ++oh) {
                            bool good_y = (OUTPUT_SIZE_Y % OUT_BLOCK_HEIGHT == 0) || (out_y + oh < OUTPUT_SIZE_Y);
                            if (good_y) {
                                __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                                for (uint ow = 0; ow < OUT_BLOCK_WIDTH; ow++) {

    #if !IS_DYNAMIC
        #if OUTPUT_SIZE_X % OUT_BLOCK_WIDTH != 0
                                    if (out_x + OUT_BLOCK_WIDTH > OUTPUT_SIZE_X && ow >= OUTPUT_SIZE_X % OUT_BLOCK_WIDTH)
                                        break;
        #endif
    #else
                                    if (OUTPUT_SIZE_X % OUT_BLOCK_WIDTH != 0 && out_x + OUT_BLOCK_WIDTH > OUTPUT_SIZE_X && ow >= OUTPUT_SIZE_X % OUT_BLOCK_WIDTH)
                                        break;
    #endif
                                    if (out_f_g < FILTER_OFM_NUM) {
                                        output[dst_index + ow * FSV + oh * OUTPUT_Y_PITCH * FSV + od * OUTPUT_Z_PITCH * FSV] = result[ofb][od][oh][ow];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#if ((FILTER_OFM_NUM % OFM_BLOCKS_PER_SIMD == 0) && ((FILTER_GROUPS_NUM == 1) || (FILTER_OFM_NUM % SIMD == 0)))
    }
#endif
}

#undef TYPE_N_
#undef TYPE_N
#undef AS_TYPE_N
#undef AS_TYPE_N_

#undef INPUT0_TYPE_4
#undef AS_INPUT0_TYPE_4

#ifdef NON_ZERO_INPUT0_PAD_BEFORE
    #undef NON_ZERO_INPUT0_PAD_BEFORE
#endif

#ifdef SHOULD_BALANCE_COMPENSATION
    #undef SHOULD_BALANCE_COMPENSATION
#endif

#ifdef SHOULD_USE_DATA_ZP
    #undef SHOULD_USE_DATA_ZP
#endif

#ifdef SHOULD_USE_DATA_AND_WEIGHTS_ZP
    #undef SHOULD_USE_DATA_AND_WEIGHTS_ZP
#endif

#ifdef ACCUMULATOR_TYPE_4
#undef ACCUMULATOR_TYPE_4
#endif

#ifdef FILTER_TYPE_16
#undef FILTER_TYPE_16
#endif

#undef AS_FILTER_TYPE_4

#undef SIMD
#undef FSV
#undef OFM_VALUES_PER_WI
