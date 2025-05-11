// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"

#include "deconvolution_gpu_imad_common.cl"

DECLARE_LOAD_CONTINOUS_4(load_input_ui, uint)
DECLARE_LOAD_CONTINOUS_4(load_weights_ui, uint)

#define FILTER_TYPE4 MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)

#define AS_FILTER_TYPE4 CAT(as_, FILTER_TYPE4)
#define AS_INPUT_TYPE4 CAT(as_, INPUT_TYPE4)

#define WEIGHTS_GET_INDEX(g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(FILTER, g, o, i, z, y, x)

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

    uint out_b = get_global_id(2);
    uint out_f = get_global_id(0);
    uint out_x = (uint)get_global_id(1) % OUTPUT_SIZE_X;
#if OUTPUT_DIMS <= 4
    uint out_y = (uint)get_global_id(1) / OUTPUT_SIZE_X;
    uint out_z = 0;
#elif OUTPUT_DIMS == 5
    uint out_y = (uint)get_global_id(1) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    uint out_z = (uint)get_global_id(1) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
#endif

#if GROUPED
    uint group = out_f / FILTER_OFM_NUM;
    uint ofm = out_f % FILTER_OFM_NUM;
#else
    uint group = 0;
    uint ofm = out_f;
#endif
    uint if_start = group * FILTER_IFM_NUM;

    int in_x_start = (int)out_x + (PADDING_SIZE_X - FILTER_SIZE_X + 1);
    int in_y_start = (int)out_y + (PADDING_SIZE_Y - FILTER_SIZE_Y + 1);
    int in_z_start = (int)out_z + (PADDING_SIZE_Z - FILTER_SIZE_Z + 1);

    uint fx_start = 0;
    uint fx_end = FILTER_SIZE_X;
    uint fx_inc = STRIDE_SIZE_X;
    if (in_x_start < 0)
        fx_start = -in_x_start;
    else if (in_x_start % STRIDE_SIZE_X != 0)
        fx_start = STRIDE_SIZE_X - in_x_start % STRIDE_SIZE_X;
    if (in_x_start + FILTER_SIZE_X - 1 >= INPUT0_SIZE_X * STRIDE_SIZE_X)
        fx_end = INPUT0_SIZE_X * STRIDE_SIZE_X - in_x_start;

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

    ACCUMULATOR_TYPE acc = 0;
    uint in[TILE_IFM / 4];
    uint wei[TILE_IFM / 4];

    for (uint fz = fz_start; fz < fz_end; fz += fz_inc) {
        int in_z = in_z_start + fz;
        uint fixed_in_z = in_z / STRIDE_SIZE_Z;

        for (uint fy = fy_start; fy < fy_end; fy += fy_inc) {
            int in_y = in_y_start + fy;
            uint fixed_in_y = in_y / STRIDE_SIZE_Y;

            for (uint fx = fx_start; fx < fx_end; fx += fx_inc) {
                int in_x = in_x_start + fx;
                uint fixed_in_x = in_x / STRIDE_SIZE_X;

                for (uint fi = 0; fi < FILTER_IFM_NUM; fi += TILE_IFM) {
                    // Load weights [1, TILE_IFM, 1, 1]
                    uint weights_idx = WEIGHTS_GET_INDEX(group, ofm, fi, FILTER_SIZE_Z - fz - 1, FILTER_SIZE_Y - fy - 1, FILTER_SIZE_X - fx - 1);
                    FUNC_CALL(load_weights_ui)(weights_ui, weights_idx / 4, TILE_IFM / 4, wei);

                    // Load input [1, TILE_IFM, 1, 1]
#if FILTER_GROUPS_NUM == 1 || FILTER_IFM_NUM % TILE_IFM == 0
#   if OUTPUT_DIMS <= 4
                    uint input_idx = INPUT0_GET_INDEX(out_b, fi + if_start, fixed_in_y, fixed_in_x);
#   elif OUTPUT_DIMS == 5
                    uint input_idx = INPUT0_GET_INDEX(out_b, fi + if_start, fixed_in_z, fixed_in_y, fixed_in_x);
#   endif
                    FUNC_CALL(load_input_ui)(input_ui, input_idx / 4, TILE_IFM / 4, in);
#else
                    for (uint tifm = 0; tifm < TILE_IFM; ++tifm) {
#   if OUTPUT_DIMS <= 4
                        uint input_idx = INPUT0_GET_INDEX(out_b, fi + if_start + tifm, fixed_in_y, fixed_in_x);
#   elif OUTPUT_DIMS == 5
                        uint input_idx = INPUT0_GET_INDEX(out_b, fi + if_start + tifm, fixed_in_z, fixed_in_y, fixed_in_x);
#   endif
                        ((INPUT0_TYPE*)(in))[tifm] = input[input_idx];
                    }
#endif

                    __attribute__((opencl_unroll_hint))
                    for (uint imad_it = 0; imad_it < TILE_IFM / 4; ++imad_it) {
                        acc = IMAD(acc, AS_INPUT_TYPE4(in[imad_it]), AS_FILTER_TYPE4(wei[imad_it]));
                    }
                }
            }
        }
    }

    ACTIVATION_TYPE dequantized;
    dequantized = TO_ACTIVATION_TYPE(acc);

#if BIAS_TERM
    BIAS_TYPE bias_val = bias[out_f];
    dequantized += TO_ACTIVATION_TYPE(bias_val);
#endif

    OUTPUT_TYPE result;
#if HAS_FUSED_OPS
    FUSED_OPS;
    result = FUSED_OPS_RESULT;
#else
    result = TO_OUTPUT_TYPE(dequantized);
#endif

#if OUTPUT_DIMS <= 4
    uint output_idx = OUTPUT_GET_INDEX(out_b, out_f, out_y, out_x);
#elif OUTPUT_DIMS == 5
    uint output_idx = OUTPUT_GET_INDEX(out_b, out_f, out_z, out_y, out_x);
#endif
    output[output_idx] = result;
}

#undef FILTER_TYPE4
#undef INPUT_TYPE4
#undef AS_FILTER_TYPE4
#undef AS_INPUT_TYPE4

#undef WEIGHTS_GET_INDEX
