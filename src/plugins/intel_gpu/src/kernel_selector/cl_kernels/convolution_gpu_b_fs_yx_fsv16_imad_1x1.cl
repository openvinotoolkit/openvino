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

#if INPUT0_PAD_BEFORE_SIZE_X != 0 || INPUT0_PAD_BEFORE_SIZE_Y != 0
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

#if FILTER_LAYOUT_OS_IS_YX_OSV16_ISV16
#   define GET_WEIGHTS_INDEX(o, i, z, y, x)     GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(FILTER, o, i, y, x)
#   define WEIGHTS_FEATURE_BLOCK_PITCH          (ALIGN(FILTER_IFM_NUM, FSV) * FILTER_SIZE_X * FILTER_SIZE_Y * FSV)
#   define WEIGHTS_IS_PITCH                     (FSV * FSV * FILTER_SIZE_X * FILTER_SIZE_Y)

#elif FILTER_LAYOUT_OS_IS_ZYX_OSV32_ISV16
#   define GET_WEIGHTS_INDEX(o, i, z, y, x)     GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(FILTER, o, i, z, y, x)
#   define WEIGHTS_FEATURE_BLOCK_PITCH          (FSV * FSV)
#   define WEIGHTS_IS_PITCH                     (2 * FSV * FSV * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z)

#elif FILTER_LAYOUT_OS_IS_ZYX_OSV64_ISV16
#   define GET_WEIGHTS_INDEX(o, i, z, y, x)     GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(FILTER, o, i, z, y, x)
#   define WEIGHTS_FEATURE_BLOCK_PITCH          (FSV * FSV)
#   define WEIGHTS_IS_PITCH                     (4 * FSV * FSV * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z)

#endif

#define FSV  16
#define SIMD 16

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(1, SIMD * FEATURE_SLM_SPLIT, 1)))
KERNEL(convolution_gpu_b_fs_yx_fsv16_imad_1x1)(
    const __global INPUT0_TYPE   *conv_input,
    __global OUTPUT_TYPE         *output,
    const __global FILTER_TYPE    *weights
#if BIAS_TERM
    , const __global BIAS_TYPE     *biases
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
)
{
    // Use group ids to ease sub-group uniform variables optimization for compiler
    const uint out_yx_sg = (uint)get_group_id(0) * OUT_BLOCK_SPATIAL;
    uint out_fg = (uint)get_group_id(1) * OUT_BLOCK_FEATURES * SIMD;
    const uint out_b = (uint)get_group_id(2);
    uint out_f = out_fg + get_sub_group_local_id();

    const uint sglid = get_sub_group_local_id();

    uint out_x_shuffle[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
    uint out_y_shuffle[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };

    const uint max_out_yx = OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
    uint max_local_yx = min(max_out_yx, out_yx_sg + OUT_BLOCK_SPATIAL);
    unroll_for (uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
        uint out_yx_shuffle = out_yx_sg + sglid + os * SIMD;
        uint out_yx_clamp = max_out_yx % OUT_BLOCK_SPATIAL == 0
                          ? out_yx_shuffle
                          : min(out_yx_shuffle, max_local_yx - 1);
        out_x_shuffle[os] = out_yx_clamp % OUTPUT_SIZE_X;
        out_y_shuffle[os] = out_yx_clamp / OUTPUT_SIZE_X;
    }

    const uint ifm_blocks = CEIL_DIV(INPUT0_FEATURE_NUM, FSV);
    const uint ifm_blocks_per_sg = ifm_blocks / FEATURE_SLM_SPLIT;
    const uint ifm_per_sg = ifm_blocks_per_sg * FSV;

    uint feature_offset = 0;
    uint feature_blocks = ifm_blocks_per_sg;
#if FEATURE_SLM_SPLIT != 1
    feature_offset = get_sub_group_id() * ifm_per_sg;

    if (ifm_blocks % FEATURE_SLM_SPLIT != 0) {
        bool bigger_sg = get_sub_group_id() < ifm_blocks % FEATURE_SLM_SPLIT;
        feature_blocks = bigger_sg ? ifm_blocks_per_sg + 1 : ifm_blocks_per_sg;
        feature_offset += bigger_sg ? get_sub_group_id() * FSV : ifm_blocks % FEATURE_SLM_SPLIT * FSV;
    }
#endif

    uint filter_idx = GET_WEIGHTS_INDEX(out_f, feature_offset, 0, 0, 0);

    uint input_idx[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
    #ifdef SHOULD_USE_DATA_ZP
        uint input_x[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
        uint input_y[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
    #endif

    unroll_for (uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
        #ifdef SHOULD_USE_DATA_ZP
            input_x[os] = out_x_shuffle[os] * STRIDE_SIZE_X - PADDING_SIZE_X;
            input_y[os] = out_y_shuffle[os] * STRIDE_SIZE_Y - PADDING_SIZE_Y;
            input_idx[os] = INPUT0_GET_INDEX(out_b, feature_offset, input_y[os], input_x[os]);
        #else
            uint input_x = out_x_shuffle[os] * STRIDE_SIZE_X - PADDING_SIZE_X;
            uint input_y = out_y_shuffle[os] * STRIDE_SIZE_Y - PADDING_SIZE_Y;
            input_idx[os] = INPUT0_GET_INDEX(out_b, feature_offset, input_y, input_x);
        #endif
    }

    ACCUMULATOR_TYPE dotProd[OUT_BLOCK_FEATURES][OUT_BLOCK_SPATIAL] = { };

    #ifdef SHOULD_USE_DATA_ZP
        uint data_zp_idx = feature_offset;
        uint4 data_zp_val;
    #endif

    #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
        uint4 weights_zp_val[OUT_BLOCK_FEATURES];
        unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
            weights_zp_val[ofb] = as_uint4((FILTER_TYPE_16)weights_zp[out_f + ofb * FSV]);
        }
        #if INPUT0_FEATURE_NUM % FSV != 0
            uint4 weights_zp_vec_partial[OUT_BLOCK_FEATURES];
            unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                weights_zp_vec_partial[ofb] = weights_zp_val[ofb];
                FILTER_TYPE* wzp_p = (FILTER_TYPE*)&weights_zp_vec_partial[ofb];
                unroll_for(uint f = INPUT0_FEATURE_NUM % FSV; f < FSV; f++) {
                    wzp_p[f] = 0;
                }
            }
        #endif
    #endif

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < feature_blocks; ++k) {
        #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
            #if INPUT0_FEATURE_NUM % FSV != 0
                if (feature_offset + (k + 1) * FSV >= ALIGN(INPUT0_FEATURE_NUM, FSV)) {
                    unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                        weights_zp_val[ofb] = weights_zp_vec_partial[ofb];
                    }
                }
            #endif
        #endif

        #ifdef SHOULD_USE_DATA_ZP
            #if (INPUT0_FEATURE_NUM % FSV != 0)
                data_zp_val = as_uint4(vload16(0, activations_zp + data_zp_idx));
            #else
                data_zp_val = vload4(0, (__global uint *)(activations_zp + data_zp_idx));
            #endif
        #endif

        #ifdef SHOULD_USE_DATA_AND_WEIGHTS_ZP
            ACCUMULATOR_TYPE_4 dotProdAZPxWZP[OUT_BLOCK_FEATURES];
            unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                dotProdAZPxWZP[ofb] = 0;
                unroll_for(uint ive = 0; ive < 4; ive++) {
                    dotProdAZPxWZP[ofb][ive] = TO_ACCUMULATOR_TYPE(
                    IMAD(dotProdAZPxWZP[ofb][ive],
                    AS_INPUT0_TYPE_4(data_zp_val[ive]),
                    AS_FILTER_TYPE_4(weights_zp_val[ofb][ive])));
                }
            }
        #endif

        uint4 weights_val[OUT_BLOCK_FEATURES] = { };
        unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
            weights_val[ofb] = vload4(0, (__global uint*)(weights + filter_idx + ofb * WEIGHTS_FEATURE_BLOCK_PITCH));
        }

        uint4 input_val[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
        unroll_for(uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
            #if defined ASYMMETRIC_DATA_QUANTIZATION && defined NON_ZERO_INPUT0_PAD_BEFORE
                if (((input_x[os] < 0) || (input_x[os] >= INPUT0_SIZE_X)) ||
                    ((input_y[os] < 0) || (input_y[os] >= INPUT0_SIZE_Y))) {
                    input_val[os] = data_zp_val;
                } else {
            #endif
                    input_val[os] = vload4(0, (__global uint *)(conv_input + input_idx[os]));
            #if defined ASYMMETRIC_DATA_QUANTIZATION && defined NON_ZERO_INPUT0_PAD_BEFORE
                }
            #endif
        }

#if OUT_BLOCK_FEATURES > 1 && FEATURE_SLM_SPLIT != 1 && OUT_BLOCK_SPATIAL > 14
        // For some cases compiler spills here due to loop order
        // Use suboptimal order to avoid this at cost of instruction dispatch delays.
        unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
            unroll_for(uint ive = 0; ive < 4; ++ive) {
                unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                    #ifdef SHOULD_USE_DATA_ZP
                        ACCUMULATOR_TYPE dotProdAZPxW = 0;
                        dotProdAZPxW = TO_ACCUMULATOR_TYPE(
                        IMAD(dotProdAZPxW,
                        AS_INPUT0_TYPE_4(data_zp_val[ive]),
                        AS_FILTER_TYPE_4(weights_val[ofb][ive])));
                    #endif
#else
        unroll_for(uint ive = 0; ive < 4; ++ive) {
            unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                #ifdef SHOULD_USE_DATA_ZP
                    ACCUMULATOR_TYPE dotProdAZPxW = 0;
                    dotProdAZPxW = TO_ACCUMULATOR_TYPE(
                    IMAD(dotProdAZPxW,
                    AS_INPUT0_TYPE_4(data_zp_val[ive]),
                    AS_FILTER_TYPE_4(weights_val[ofb][ive])));
                #endif
                unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
#endif
                        INPUT0_TYPE_4 inputs = AS_INPUT0_TYPE_4(_sub_group_shuffle(input_val[os / SIMD][ive], os % SIMD));

                        dotProd[ofb][os] = IMAD(dotProd[ofb][os],
                                                inputs,
                                                AS_FILTER_TYPE_4(weights_val[ofb][ive]));

                        #ifdef ASYMMETRIC_WEIGHTS_QUANTIZATION
                            ACCUMULATOR_TYPE dotProdAxWZP = 0;
                            dotProdAxWZP = TO_ACCUMULATOR_TYPE(
                            IMAD(dotProdAxWZP,
                            inputs,
                            AS_FILTER_TYPE_4(weights_zp_val[ofb][ive])));
                            dotProd[ofb][os] -= dotProdAxWZP;
                        #endif

                        #if !defined COMPENSATION_TERM && defined ASYMMETRIC_DATA_QUANTIZATION
                            dotProd[ofb][os] -= dotProdAZPxW;
                        #endif

                        #if (!defined COMPENSATION_TERM && \
                                defined ASYMMETRIC_DATA_QUANTIZATION && \
                                defined ASYMMETRIC_WEIGHTS_QUANTIZATION)
                            dotProd[ofb][os] += dotProdAZPxWZP[ofb][ive];
                        #endif
                }
            }
        }

        filter_idx += WEIGHTS_IS_PITCH;
        unroll_for(uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
            input_idx[os] += INPUT0_FEATURE_PITCH * FSV;
        }

        #ifdef SHOULD_USE_DATA_ZP
            data_zp_idx += FSV;
        #endif
    }

#if FEATURE_SLM_SPLIT != 1
    // Additional local memory reduction for feature split mode
#   if FEATURE_SLM_SPLIT < OUT_BLOCK_FEATURES
#   error convolution_gpu_b_fs_yx_fsv16_imad_1x1.cl - OUT_BLOCK_FEATURES must be less or equal to FEATURE_SLM_SPLIT
#   endif

    const uint partial_acc_size = (FEATURE_SLM_SPLIT - 1) * OUT_BLOCK_FEATURES * SIMD * OUT_BLOCK_SPATIAL;
    __local ACCUMULATOR_TYPE partial_acc[partial_acc_size];

    uint sgid_start_idx = get_sub_group_id();
    sgid_start_idx = sgid_start_idx == 0 ? 0 : sgid_start_idx - 1;
    __local ACCUMULATOR_TYPE* partial_acc_ptr = partial_acc + sgid_start_idx * OUT_BLOCK_FEATURES * SIMD * OUT_BLOCK_SPATIAL + sglid;

    if (get_sub_group_id() < OUT_BLOCK_FEATURES) {
        unroll_for(uint wg = 0; wg < OUT_BLOCK_FEATURES; ++wg) {
            if (get_sub_group_id() == wg) {
                unroll_for(uint ofb = 0; ofb < wg; ++ofb) {
                    unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
                        const uint partial_acc_ptr_idx =
                            ofb * OUT_BLOCK_SPATIAL * SIMD +
                            os * SIMD;
                        partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][os];
                    }
                }
                unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
                    dotProd[0][os] = dotProd[wg][os];
                }
                unroll_for(uint ofb = wg + 1; ofb < OUT_BLOCK_FEATURES; ++ofb) {
                    unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
                        const uint partial_acc_ptr_idx =
                            ((wg != 0) ? OUT_BLOCK_SPATIAL * OUT_BLOCK_FEATURES * SIMD : 0) +
                            ofb * OUT_BLOCK_SPATIAL * SIMD +
                            os * SIMD;
                        partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][os];
                    }
                }
            }
        }
    } else {
        unroll_for(uint ofb = 0; ofb < OUT_BLOCK_FEATURES; ++ofb) {
            unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
                const uint partial_acc_ptr_idx =
                    ofb * OUT_BLOCK_SPATIAL * SIMD +
                    os * SIMD;
                partial_acc_ptr[partial_acc_ptr_idx] = dotProd[ofb][os];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_id() >= OUT_BLOCK_FEATURES)
        return;

    partial_acc_ptr = partial_acc + get_sub_group_id() * OUT_BLOCK_SPATIAL * SIMD + sglid;
    unroll_for (uint wg = 0; wg < FEATURE_SLM_SPLIT - 1; ++wg) {
        unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
            const uint partial_acc_ptr_idx =
                wg * OUT_BLOCK_FEATURES * SIMD * OUT_BLOCK_SPATIAL +
                os * SIMD;
            dotProd[0][os] += partial_acc_ptr[partial_acc_ptr_idx];
        }
    }
#endif

#if FEATURE_SLM_SPLIT == 1
#   define FINAL_OUT_BLOCK_FEATURES (OUT_BLOCK_FEATURES)
#else
#   define FINAL_OUT_BLOCK_FEATURES 1
    out_f += get_sub_group_id() * SIMD;
    out_fg += get_sub_group_id() * SIMD;

    if (CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % OUT_BLOCK_FEATURES != 0 && out_fg >= OUTPUT_FEATURE_NUM)
        return;
#endif

#if BIAS_TERM
    // Preload bias
    BIAS_TYPE bias_val[FINAL_OUT_BLOCK_FEATURES];
    for (uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
        bias_val[ofb] = biases[out_f + ofb * SIMD];
    }
#endif

#ifdef COMPENSATION_TERM
    COMPENSATION_TYPE comp[FINAL_OUT_BLOCK_FEATURES];
    unroll_for (uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
        comp[ofb] = compensation[out_f + ofb * SIMD];
    }
#endif

    // Convert accumulator type to activation type
    ACTIVATION_TYPE dequantized[FINAL_OUT_BLOCK_FEATURES][OUT_BLOCK_SPATIAL];
    unroll_for (uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
        unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
            dequantized[ofb][os] = TO_ACTIVATION_TYPE(dotProd[ofb][os]);

#if BIAS_TERM
            dequantized[ofb][os] += TO_ACTIVATION_TYPE(bias_val[ofb]);
#endif
#ifdef COMPENSATION_TERM
            dequantized[ofb][os] += TO_ACTIVATION_TYPE(comp[ofb]);
#endif
        }
    }

    // Fused ops/activation
    OUTPUT_TYPE result[FINAL_OUT_BLOCK_FEATURES][OUT_BLOCK_SPATIAL];
    unroll_for (uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD_SCALAR
        FUSED_OPS_PRELOAD_SCALAR;
#endif
        unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
#if HAS_FUSED_OPS
    #if FUSED_OPS_CAN_USE_PRELOAD_SCALAR
            FUSED_OPS_CALC_SCALAR;
    #else
            FUSED_OPS_SCALAR;
    #endif
            result[ofb][os] = FUSED_OPS_RESULT_SCALAR;
#else
            result[ofb][os] = TO_OUTPUT_TYPE(ACTIVATION(dequantized[ofb][os], ACTIVATION_PARAMS));
#endif
        }
    }

    // Store output
    // Check if can use block writes
    bool only_x_block = OUTPUT_SIZE_X % OUT_BLOCK_SPATIAL == 0;
    bool at_least_one_x_block = OUTPUT_SIZE_X >= OUT_BLOCK_SPATIAL;
    bool full_x = out_yx_sg % OUTPUT_SIZE_X <= OUTPUT_SIZE_X - OUT_BLOCK_SPATIAL;
    bool can_write_x = only_x_block || (at_least_one_x_block && full_x);

    bool no_x_pad = OUTPUT_PAD_BEFORE_SIZE_X == 0 && OUTPUT_PAD_AFTER_SIZE_X == 0;
    bool exact_spatial = max_out_yx % OUT_BLOCK_SPATIAL == 0;
    bool full_spatial = out_yx_sg <= max_out_yx - OUT_BLOCK_SPATIAL;
    bool can_write_spatial = no_x_pad && (exact_spatial || full_spatial);

    bool full_feature_block = (OUTPUT_FEATURE_NUM % SIMD == 0) || (out_fg + FINAL_OUT_BLOCK_FEATURES * SIMD <= OUTPUT_FEATURE_NUM);

    bool can_use_full_block_write =  full_feature_block && (can_write_x || can_write_spatial);
    if (can_use_full_block_write) {
        uint output_idx = OUTPUT_GET_INDEX(out_b,
                                           out_fg,
                                           _sub_group_shuffle(out_y_shuffle[0], 0),
                                           _sub_group_shuffle(out_x_shuffle[0], 0));
        unroll_for(uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
            bool good_of_block = (CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % FINAL_OUT_BLOCK_FEATURES == 0)
                               || (out_fg + FINAL_OUT_BLOCK_FEATURES * SIMD <= OUTPUT_FEATURE_NUM)
                               || (ofb < CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % FINAL_OUT_BLOCK_FEATURES);
            if (good_of_block) {
                uint os = 0;
#if OUTPUT_TYPE_SIZE == 1
                for (; os + 8 <= OUT_BLOCK_SPATIAL; os += 8) {
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) result_val;
                    unroll_for(uint i = 0; i < 8; ++i) {
                        result_val[i] = result[ofb][os + i];
                    }
                    DT_OUTPUT_BLOCK_WRITE8(output, output_idx, result_val);
                    output_idx += 8 * SIMD;
                }
#endif
#if OUTPUT_TYPE_SIZE <= 2
                for (; os + 4 <= OUT_BLOCK_SPATIAL; os += 4) {
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) result_val;
                    unroll_for(uint i = 0; i < 4; ++i) {
                        result_val[i] = result[ofb][os + i];
                    }
                    DT_OUTPUT_BLOCK_WRITE4(output, output_idx, result_val);
                    output_idx += 4 * SIMD;
                }
#endif
                for (; os + 2 <= OUT_BLOCK_SPATIAL; os += 2) {
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) result_val;
                    unroll_for(uint i = 0; i < 2; ++i) {
                        result_val[i] = result[ofb][os + i];
                    }
                    DT_OUTPUT_BLOCK_WRITE2(output, output_idx, result_val);
                    output_idx += 2 * SIMD;
                }
                if (OUT_BLOCK_SPATIAL % 2 == 1) {
                    OUTPUT_TYPE result_val = result[ofb][os];
                    DT_OUTPUT_BLOCK_WRITE(output, output_idx, result_val);
                    output_idx += 1 * SIMD;
                }
            }
            output_idx += OUTPUT_FEATURE_PITCH * FSV - OUT_BLOCK_SPATIAL * SIMD;
        }
    } else {
        uint output_idx_shuffle[CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD)] = { };
        unroll_for(uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
            output_idx_shuffle[os] = OUTPUT_GET_INDEX(out_b, out_fg, out_y_shuffle[os], out_x_shuffle[os]);
        }
        unroll_for(uint ofb = 0; ofb < FINAL_OUT_BLOCK_FEATURES; ++ofb) {
            bool good_of_block = (CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % FINAL_OUT_BLOCK_FEATURES == 0)
                               || (out_fg + FINAL_OUT_BLOCK_FEATURES * SIMD <= OUTPUT_FEATURE_NUM)
                               || (ofb < CEIL_DIV(OUTPUT_FEATURE_NUM, SIMD) % FINAL_OUT_BLOCK_FEATURES);
            if (good_of_block) {
                unroll_for(uint os = 0; os < OUT_BLOCK_SPATIAL; ++os) {
                    bool good_os = (max_out_yx % OUT_BLOCK_SPATIAL == 0) || (out_yx_sg <= max_out_yx - OUT_BLOCK_SPATIAL) || (os < max_out_yx % OUT_BLOCK_SPATIAL);
                    if (!good_os)
                        break;

                    uint output_idx = _sub_group_shuffle(output_idx_shuffle[os / SIMD], os % SIMD);
                    bool good_of = (OUTPUT_FEATURE_NUM % SIMD == 0) || (out_f + ofb * SIMD < OUTPUT_FEATURE_NUM);

                    if (!good_of)
                        result[ofb][os] = (OUTPUT_TYPE)0;

                    output[output_idx + sglid] = result[ofb][os];
                }
            }

            unroll_for(uint os = 0; os < CEIL_DIV(OUT_BLOCK_SPATIAL, SIMD); ++os) {
                output_idx_shuffle[os] += OUTPUT_FEATURE_PITCH * FSV;
            }
        }
    }

#undef FINAL_OUT_BLOCK_FEATURES
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
