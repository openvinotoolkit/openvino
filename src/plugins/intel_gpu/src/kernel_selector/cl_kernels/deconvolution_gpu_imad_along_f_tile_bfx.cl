// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#include "deconvolution_gpu_imad_common.cl"

DECLARE_LOAD_CONTINOUS_4(load_input_ui, uint)
DECLARE_LOAD_CONTINOUS_4(load_weights_ui, uint)

#if OUTPUT_TYPE_SIZE == 1
DECLARE_STORE_BLOCK_16(store_output, OUTPUT_TYPE)
#elif OUTPUT_TYPE_SIZE == 2
DECLARE_STORE_BLOCK_8(store_output, OUTPUT_TYPE)
#else
DECLARE_STORE_BLOCK_4(store_output, OUTPUT_TYPE)
#endif

#define FILTER_TYPE4 MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)

#define AS_FILTER_TYPE4 CAT(as_, FILTER_TYPE4)
#define AS_INPUT_TYPE4 CAT(as_, INPUT_TYPE4)

#define WEIGHTS_GET_INDEX(g, o, i, z, y, x)     GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(FILTER, g, o, i, z, y, x, (SIMD * TILE_OFM), TILE_IFM)
#define WEIGHTS_TILE_IFM_PITCH                  (TILE_IFM * SIMD * TILE_OFM)
#define WEIGHTS_IN_TILE_OFM_PITCH               (TILE_IFM * SIMD)

__attribute__((reqd_work_group_size(1, SIMD, 1)))
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(deconvolution_gpu_imad_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif

) {
    const __global uint* input_ui = (const __global uint*)input;
    const __global uint* weights_ui = (const __global uint*)weights;

    uint out_b = get_global_id(2) * TILE_B;
    uint out_fg = get_group_id(1) * SIMD * TILE_OFM;
    uint out_f = out_fg + get_sub_group_local_id();
    uint out_x = (uint)get_global_id(0) * TILE_X % ALIGN(OUTPUT_SIZE_X, TILE_X);
#if OUTPUT_DIMS <= 4
    uint out_y = (uint)get_global_id(0) / CEIL_DIV(OUTPUT_SIZE_X, TILE_X);
    uint out_z = 0;
#elif OUTPUT_DIMS == 5
    uint out_y = (uint)get_global_id(0) / CEIL_DIV(OUTPUT_SIZE_X, TILE_X) % OUTPUT_SIZE_Y;
    uint out_z = (uint)get_global_id(0) / (CEIL_DIV(OUTPUT_SIZE_X, TILE_X) * OUTPUT_SIZE_Y);
#endif
    const uint sglid = get_sub_group_local_id();

#if GROUPED
    uint group = out_fg / FILTER_OFM_NUM;
    uint ofm = out_fg % FILTER_OFM_NUM + sglid;
#else
    uint group = 0;
    uint ofm = out_f;
#endif
    uint if_start = group * FILTER_IFM_NUM;

    int in_x_start = (int)out_x + (PADDING_SIZE_X - FILTER_SIZE_X + 1);
    int in_y_start = (int)out_y + (PADDING_SIZE_Y - FILTER_SIZE_Y + 1);
    int in_z_start = (int)out_z + (PADDING_SIZE_Z - FILTER_SIZE_Z + 1);

    uint fy_start = 0;
    uint fy_end = FILTER_SIZE_Y;
    uint fy_inc = STRIDE_SIZE_Y;
    if (in_y_start < 0)
        fy_start = -in_y_start;
    else if (in_y_start % STRIDE_SIZE_Y != 0)
        fy_start = STRIDE_SIZE_Y - in_y_start % STRIDE_SIZE_Y;
    if (in_y_start + FILTER_SIZE_Y - 1 >= INPUT0_SIZE_Y * STRIDE_SIZE_Y)
        fy_end = INPUT0_SIZE_Y * STRIDE_SIZE_Y - in_y_start;

    uint fz_start = 0;
    uint fz_end = FILTER_SIZE_Z;
    uint fz_inc = STRIDE_SIZE_Z;
    if (in_z_start < 0)
        fz_start = -in_z_start;
    else if (in_z_start % STRIDE_SIZE_Z != 0)
        fz_start = STRIDE_SIZE_Z - in_z_start % STRIDE_SIZE_Z;
    if (in_z_start + FILTER_SIZE_Z - 1 >= INPUT0_SIZE_Z * STRIDE_SIZE_Z)
        fz_end = INPUT0_SIZE_Z * STRIDE_SIZE_Z - in_z_start;

    ACCUMULATOR_TYPE acc[TILE_B][TILE_OFM][TILE_X] = { };
    uint in[TILE_B][TILE_IFM / 4];
    uint wei[TILE_OFM][TILE_IFM / 4];

    for (uint fz = fz_start; fz < fz_end; fz += fz_inc) {
        int in_z = in_z_start + fz;
        uint fixed_in_z = in_z / STRIDE_SIZE_Z;

        for (uint fy = fy_start; fy < fy_end; fy += fy_inc) {
            int in_y = in_y_start + fy;
            uint fixed_in_y = in_y / STRIDE_SIZE_Y;

            for (uint fx = 0; fx < FILTER_SIZE_X; fx += 1) {
                int in_x = in_x_start + fx + ((TILE_X == SIMD || sglid < TILE_X) ? sglid : 0);
                bool zero_x = false;
                zero_x |= in_x < 0;
                zero_x |= in_x >= INPUT0_SIZE_X * STRIDE_SIZE_X;
                zero_x |= in_x % STRIDE_SIZE_X != 0;
                in_x = max(in_x, 0);
                in_x = min(in_x, INPUT0_SIZE_X * STRIDE_SIZE_X);
                uint fixed_in_x = in_x / STRIDE_SIZE_X;

                uint weights_offset = WEIGHTS_GET_INDEX(group, ofm, 0, FILTER_SIZE_Z - fz - 1, FILTER_SIZE_Y - fy - 1, FILTER_SIZE_X - fx - 1) / 4;

#if INPUT_VALID_TILE_IFM_PITCH
#   if OUTPUT_DIMS <= 4
                uint input_offset = INPUT0_GET_INDEX(out_b, if_start, fixed_in_y, fixed_in_x) / 4;
#   elif OUTPUT_DIMS == 5
                uint input_offset = INPUT0_GET_INDEX(out_b, if_start, fixed_in_z, fixed_in_y, fixed_in_x) / 4;
#   endif
#endif

                for (uint fi = 0; fi < FILTER_IFM_NUM; fi += TILE_IFM) {
                    // Load weights [TILE_OFM, TILE_IFM, 1, 1]
                    unroll_for (uint of = 0; of < TILE_OFM; ++of) {
                        uint weights_idx = weights_offset + of * WEIGHTS_IN_TILE_OFM_PITCH / 4;
                        FUNC_CALL(load_weights_ui)(weights_ui, weights_idx, TILE_IFM / 4, wei[of]);
                    }
                    weights_offset += WEIGHTS_TILE_IFM_PITCH / 4;

                    // Load input [TILE_B, TILE_IFM, 1, 1]
#if !INPUT_VALID_TILE_IFM_PITCH
#   if OUTPUT_DIMS <= 4
                    uint input_offset = INPUT0_GET_INDEX(out_b, if_start + fi, fixed_in_y, fixed_in_x) / 4;
#   elif OUTPUT_DIMS == 5
                    uint input_offset = INPUT0_GET_INDEX(out_b, if_start + fi, fixed_in_z, fixed_in_y, fixed_in_x) / 4;
#   endif
#endif
                    unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
                        uint input_idx = input_offset + ob * INPUT_IN_TILE_B_PITCH / 4;
                        FUNC_CALL(load_input_ui)(input_ui, input_idx, TILE_IFM / 4, in[ob]);
                    }
#if INPUT_VALID_TILE_IFM_PITCH
                    input_offset += INPUT_TILE_IFM_PITCH / 4;
#endif
                    if (zero_x) {
                        unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
                            unroll_for(uint ifp = 0; ifp < TILE_IFM / 4; ++ifp) {
                                in[ob][ifp] = 0;
                            }
                        }
                    }

                    unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
                        unroll_for (uint of = 0; of < TILE_OFM; ++of) {
                            unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
                                unroll_for (uint imad_it = 0; imad_it < TILE_IFM / 4; ++imad_it) {
                                    uint in_val = _sub_group_shuffle(in[ob][imad_it], tx);
                                    acc[ob][of][tx] = IMAD(acc[ob][of][tx], AS_INPUT_TYPE4(in_val), AS_FILTER_TYPE4(wei[of][imad_it]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ACTIVATION_TYPE dequantized[TILE_B][TILE_OFM][TILE_X];
    unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
                dequantized[ob][of][tx] = TO_ACTIVATION_TYPE(acc[ob][of][tx]);
            }
        }
    }

#if BIAS_TERM
    unroll_for (uint of = 0; of < TILE_OFM; ++of) {
        BIAS_TYPE bias_val = bias[out_f + of * SIMD];
        unroll_for(uint ob = 0; ob < TILE_B; ++ob) {
            unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
                dequantized[ob][of][tx] += TO_ACTIVATION_TYPE(bias_val);
            }
        }
    }
#endif

    OUTPUT_TYPE result[TILE_B][TILE_OFM][TILE_X];
    unroll_for (uint of = 0; of < TILE_OFM; ++of) {
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_PRELOAD;
#endif
        unroll_for(uint ob = 0; ob < TILE_B; ++ob) {
            unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
#if HAS_FUSED_OPS
#   if FUSED_OPS_CAN_USE_PRELOAD
                FUSED_OPS_CALC;
#   else
                FUSED_OPS;
#   endif
                result[ob][of][tx] = FUSED_OPS_RESULT;
#else
                result[ob][of][tx] = TO_OUTPUT_TYPE(dequantized[ob][of][tx]);
#endif
            }
        }
    }

    bool leftovers_x = OUTPUT_SIZE_X % TILE_X != 0 && out_x + TILE_X >= OUTPUT_SIZE_X;
    bool leftovers_f = OUTPUT_FEATURE_NUM % SIMD != 0 && out_f + SIMD >= OUTPUT_FEATURE_NUM;

#if OUTPUT_NAIVE_STORE
    unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
            unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
                if ((leftovers_x && tx >= OUTPUT_SIZE_X % TILE_X) ||
                    (leftovers_f && out_f + of * SIMD >= OUTPUT_FEATURE_NUM))
                    break;
#if OUTPUT_DIMS <= 4
                uint output_idx = OUTPUT_GET_INDEX(out_b + ob, out_f + of * SIMD, out_y, out_x + tx);
#elif OUTPUT_DIMS == 5
                uint output_idx = OUTPUT_GET_INDEX(out_b + ob, out_f + of * SIMD, out_z, out_y, out_x + tx);
#endif
                output[output_idx] = result[ob][of][tx];
            }
        }
    }
#elif OUTPUT_BLOCK_X_STORE
    unroll_for (uint ob = 0; ob < TILE_B; ++ob) {
        unroll_for(uint of = 0; of < TILE_OFM; ++of) {
#if OUTPUT_DIMS <= 4
            uint output_idx = OUTPUT_GET_INDEX(out_b + ob, out_fg + of * SIMD, out_y, out_x);
#elif OUTPUT_DIMS == 5
            uint output_idx = OUTPUT_GET_INDEX(out_b + ob, out_fg + of * SIMD, out_z, out_y, out_x);
#endif
            if (!leftovers_x && !leftovers_f) {
                FUNC_CALL(store_output)(output, output_idx, TILE_X, result[ob][of]);
            } else if (!leftovers_f) {
                FUNC_CALL(store_output)(output, output_idx, OUTPUT_SIZE_X % TILE_X, result[ob][of]);
            } else {
                unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
                    if (out_f + of * SIMD < OUTPUT_FEATURE_NUM && out_x + tx < OUTPUT_SIZE_X) {
                        output[output_idx + sglid + tx * SIMD] = result[ob][of][tx];
                    }
                }
            }
        }
    }
#endif
}

#undef FILTER_TYPE4
#undef INPUT_TYPE4
#undef AS_FILTER_TYPE4
#undef AS_INPUT_TYPE4

#undef WEIGHTS_GET_INDEX
#undef WEIGHTS_TILE_IFM_PITCH
#undef WEIGHTS_IN_TILE_OFM_PITCH
