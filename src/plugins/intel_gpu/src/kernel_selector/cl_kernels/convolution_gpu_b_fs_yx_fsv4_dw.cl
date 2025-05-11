// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/imad.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// ======================================================================================
// Host side jit-constants:
// ======================================================================================
// FILTER_BLOCKED    [uint] Number of spatial filter elements that will be used
//                          with IMAD.
// TILED             [bool] Use tiled mode, load input line across simd.
// TILE_X            [uint] Size of output tile in x dimension for tiled mode.
// TILE_Y            [uint] Size of output tile in y dimension.
// SIMD              [uint] Simd size for tiled mode.
// PRELOAD_INPUT     [bool] Flag to enable preloading of input in non-tiled mode.
// INPUT_LINE_SIZE   [uint] Size of preloaded input line in x dimension
//                          when PRELOAD_INPUT is enabled. Should be enough to
//                          calculate OUTPUT_BLOCK_X values in x dimension.
// OUTPUT_BLOCK_X    [uint] Output block size for non-tiled mode.
// PRELOAD_WEIGHTS   [bool] Flag to enable preloading of weights.
// ======================================================================================

#define FSV 4

#define DEQUANTIZED_TYPE float

#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define FILTER_TYPE4      MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define OUTPUT_TYPE4      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define BIAS_TYPE4        MAKE_VECTOR_TYPE(BIAS_TYPE, 4)
#define DEQUANTIZED_TYPE4 MAKE_VECTOR_TYPE(DEQUANTIZED_TYPE, 4)

#define AS_INPUT_TYPE4(val)       CAT(as_, INPUT_TYPE4)(val)
#define TO_DEQUANTIZED_TYPE(val)  CAT(convert_, DEQUANTIZED_TYPE)(val)
#define TO_DEQUANTIZED_TYPE4(val) CAT(convert_, DEQUANTIZED_TYPE4)(val)
#define TO_OUTPUT_TYPE4(val)      CAT(convert_, OUTPUT_TYPE4)(val)

#define GET_INPUT_INDEX(b, f, y, x)    GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, y, x)
#define GET_WEIGHTS_INDEX(g, o, i, y, x)  GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(FILTER, g, o, i, y, x)
#define GET_OUTPUT_INDEX(b, f, y, x)   GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x)
#define GET_BIAS_INDEX(b, f, y, x)     GET_DATA_INDEX(BIAS, b, f, y, x)

#define INPUT_X_PITCH FSV
#define INPUT_Y_PITCH (FSV * (INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X))

#define WEIGHTS_I_PITCH 1
#define WEIGHTS_YXS_PITCH 4

#define FILTER_SPATIAL_SIZE (FILTER_SIZE_X * FILTER_SIZE_Y)

#if FILTER_BLOCKED < FILTER_SPATIAL_SIZE && FILTER_BLOCKED % 4 != 0
#   error convolution_gpu_b_fs_yx_fsv4_dw.cl - filter blocks must either cover whole spatial filter or be multiple of 4.
#endif

#if TILED
#   if INPUT_LINE_SIZE != 1
#       error convolution_gpu_b_fs_yx_fsv4_dw.cl - for tiled mode input line size must be equal 1.
#   endif
#   if OUTPUT_BLOCK_X != 1
#       error convolution_gpu_b_fs_yx_fsv4_dw.cl - for tiled mode output block size must be equal 1.
#   endif
#endif

#if TILE_Y != 1
#   if STRIDE_SIZE_Y != DILATION_SIZE_Y
#       error convolution_gpu_b_fs_yx_fsv4_dw.cl - for y-tiling stride in y must be equal dilation in y.
#   endif
#   if !PRELOAD_WEIGHTS
#       error convolution_gpu_b_fs_yx_fsv4_dw.cl - for y-tiling the weights must be preloaded.
#   endif
#endif

#if TILED
REQD_SUB_GROUP_SIZE(SIMD)
#endif
KERNEL(convolution_gpu_b_fs_yx_fsv4_dw)(
    const __global INPUT_TYPE4   *input,
    __global OUTPUT_TYPE4        *output,
    const __global FILTER_TYPE4  *weights
#if BIAS_TERM
    , const __global BIAS_TYPE   *biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if TILED
    uint x = get_group_id(0) * TILE_X;
    uint y = get_global_id(1) * TILE_Y;
    uint tile_x = get_sub_group_local_id();
#else
    uint x = get_global_id(0) * OUTPUT_BLOCK_X;
    uint y = get_global_id(1) * TILE_Y;
    uint tile_x = 0;
#endif
    uint bf = get_global_id(2);
    uint b = bf % OUTPUT_BATCH_NUM;
    uint f = bf / OUTPUT_BATCH_NUM * FSV;
    uint g = f;

    uint input_offset = GET_INPUT_INDEX(b, f, (int)y * STRIDE_SIZE_Y - PADDING_SIZE_Y, (int)x * STRIDE_SIZE_X - PADDING_SIZE_X + (int)tile_x) / FSV;
    uint weights_offset = GET_WEIGHTS_INDEX(g, 0, 0, 0, 0) / FSV;
#if TILED
    // sub-group uniform
    weights_offset = sub_group_broadcast(weights_offset, 0);
#endif

#if PRELOAD_INPUT || TILED
    INPUT_TYPE4 in[FILTER_SIZE_Y * INPUT_LINE_SIZE];

    unroll_for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi) {
        // TODO Try to avoid loading last input line in padded situations
        unroll_for(uint xi = 0; xi < INPUT_LINE_SIZE; ++xi) {
            uint preload_offset = yi * INPUT_LINE_SIZE + xi;
            uint input_x_offset = xi * (INPUT_X_PITCH / FSV);
            uint input_y_offset = yi * (DILATION_SIZE_Y * INPUT_Y_PITCH / FSV);
            uint input_spatial_offset = input_x_offset + input_y_offset;
            uint input_idx = input_spatial_offset + input_offset;
            in[preload_offset] = input[input_idx];
        }
    }
#endif
    INPUT_TYPE4 in0;
    INPUT_TYPE4 in1;
    INPUT_TYPE4 in2;
    INPUT_TYPE4 in3;

#if PRELOAD_WEIGHTS
    FILTER_TYPE4 wei[CEIL_DIV(FILTER_SPATIAL_SIZE, 4) * 4];
    unroll_for (uint fsi = 0; fsi < FILTER_SPATIAL_SIZE; fsi += 4) {
        unroll_for(uint ofi = 0; ofi < 4; ++ofi) {
            uint preload_offset = (fsi / 4) * 4 + ofi;
            uint weights_idx = weights_offset + ofi * WEIGHTS_I_PITCH + (fsi / 4) * WEIGHTS_YXS_PITCH;
            wei[preload_offset] = weights[weights_idx];
        }
    }
#endif
    FILTER_TYPE4 wei0;
    FILTER_TYPE4 wei1;
    FILTER_TYPE4 wei2;
    FILTER_TYPE4 wei3;

#if TILE_Y != 1
uint tile_y_end = min(y + TILE_Y, (uint)OUTPUT_SIZE_Y);
input_offset += DILATION_SIZE_Y * FILTER_SIZE_Y * INPUT_Y_PITCH / FSV;

for (; y < tile_y_end; ++y) {
#endif

    int acc[OUTPUT_BLOCK_X][4] = { };

    unroll_for (uint fi = 0; fi < FILTER_BLOCKED / 4 * 4; fi += 4) {
        uint4 fis = (uint4)(fi, fi + 1, fi + 2, fi + 3);

        uint4 fx = fis % FILTER_SIZE_X;
        uint4 fy = fis / FILTER_SIZE_X;

#if PRELOAD_WEIGHTS
        wei0 = wei[fi / 4 * 4 + 0];
        wei1 = wei[fi / 4 * 4 + 1];
        wei2 = wei[fi / 4 * 4 + 2];
        wei3 = wei[fi / 4 * 4 + 3];
#else
        wei0 = weights[weights_offset];
        wei1 = weights[weights_offset + 1 * WEIGHTS_I_PITCH];
        wei2 = weights[weights_offset + 2 * WEIGHTS_I_PITCH];
        wei3 = weights[weights_offset + 3 * WEIGHTS_I_PITCH];
#endif

        unroll_for(uint oxi = 0; oxi < OUTPUT_BLOCK_X; ++oxi) {
            INPUT_TYPE4 in_trans0;
            INPUT_TYPE4 in_trans1;
            INPUT_TYPE4 in_trans2;
            INPUT_TYPE4 in_trans3;
#if TILED
            in_trans0 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s0]), (fx.s0 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
            in_trans1 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s1]), (fx.s1 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
            in_trans2 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s2]), (fx.s2 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
            in_trans3 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s3]), (fx.s3 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
#elif PRELOAD_INPUT
            uint4 input_x_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X);
            uint4 input_y_offset = fy * INPUT_LINE_SIZE;
            uint4 input_idx = input_x_offset + input_y_offset;
            in_trans0 = in[input_idx.s0];
            in_trans1 = in[input_idx.s1];
            in_trans2 = in[input_idx.s2];
            in_trans3 = in[input_idx.s3];
#else
            uint4 input_x_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X) * (INPUT_X_PITCH / FSV);
            uint4 input_y_offset = fy * (DILATION_SIZE_Y * INPUT_Y_PITCH / FSV);
            uint4 input_spatial_offset = input_x_offset + input_y_offset;
            uint4 input_idx = input_spatial_offset + input_offset;

            in_trans0 = input[input_idx.s0];
            in_trans1 = input[input_idx.s1];
            in_trans2 = input[input_idx.s2];
            in_trans3 = input[input_idx.s3];
#endif

            in0 = (INPUT_TYPE4)(in_trans0.s0, in_trans1.s0, in_trans2.s0, in_trans3.s0);
            in1 = (INPUT_TYPE4)(in_trans0.s1, in_trans1.s1, in_trans2.s1, in_trans3.s1);
            in2 = (INPUT_TYPE4)(in_trans0.s2, in_trans1.s2, in_trans2.s2, in_trans3.s2);
            in3 = (INPUT_TYPE4)(in_trans0.s3, in_trans1.s3, in_trans2.s3, in_trans3.s3);

            acc[oxi][0] = IMAD(acc[oxi][0], in0, wei0);
            acc[oxi][1] = IMAD(acc[oxi][1], in1, wei1);
            acc[oxi][2] = IMAD(acc[oxi][2], in2, wei2);
            acc[oxi][3] = IMAD(acc[oxi][3], in3, wei3);
        }

        weights_offset += WEIGHTS_YXS_PITCH;
    }

#if FILTER_BLOCKED % 4 != 0
    {
        uint fi = FILTER_BLOCKED / 4 * 4;
        uint4 fis = (uint4)(fi, fi + 1, fi + 2, fi + 3);

        uint4 fx = fis % FILTER_SIZE_X;
        uint4 fy = fis / FILTER_SIZE_X;

#   if PRELOAD_WEIGHTS
        wei0 = wei[fi / 4 * 4 + 0];
        wei1 = wei[fi / 4 * 4 + 1];
        wei2 = wei[fi / 4 * 4 + 2];
        wei3 = wei[fi / 4 * 4 + 3];
#   else
        wei0 = weights[weights_offset];
        wei1 = weights[weights_offset + 1 * WEIGHTS_I_PITCH];
        wei2 = weights[weights_offset + 2 * WEIGHTS_I_PITCH];
        wei3 = weights[weights_offset + 3 * WEIGHTS_I_PITCH];
#   endif

        unroll_for(uint oxi = 0; oxi < OUTPUT_BLOCK_X; ++oxi) {
            INPUT_TYPE4 in_trans0;
            INPUT_TYPE4 in_trans1;
            INPUT_TYPE4 in_trans2;
            INPUT_TYPE4 in_trans3;
#if TILED
            in_trans0 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s0]), (fx.s0 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
#   if FILTER_BLOCKED % 4 > 1
            in_trans1 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s1]), (fx.s1 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
#   endif
#   if FILTER_BLOCKED % 4 > 2
            in_trans2 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy.s2]), (fx.s2 * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
#   endif
#elif PRELOAD_INPUT
            uint4 input_x_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X);
            uint4 input_y_offset = fy * INPUT_LINE_SIZE;
            uint4 input_idx = input_x_offset + input_y_offset;
            in_trans0 = in[input_idx.s0];
#   if FILTER_BLOCKED % 4 > 1
            in_trans1 = in[input_idx.s1];
#   endif
#   if FILTER_BLOCKED % 4 > 2
            in_trans2 = in[input_idx.s2];
#   endif
#else  // Not tiled and no input preload
            uint4 input_x_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X) * (INPUT_X_PITCH / FSV);
            uint4 input_y_offset = fy * (DILATION_SIZE_Y * INPUT_Y_PITCH / FSV);
            uint4 input_spatial_offset = input_x_offset + input_y_offset;
            uint4 input_idx = input_spatial_offset + input_offset;

            in_trans0 = input[input_idx.s0];
#   if FILTER_BLOCKED % 4 > 1
            in_trans1 = input[input_idx.s1];
#   endif
#   if FILTER_BLOCKED % 4 > 2
            in_trans2 = input[input_idx.s2];
#   endif
#endif

#if FILTER_BLOCKED % 4 < 2
            in_trans1 = (INPUT_TYPE4)(0);
#endif
#if FILTER_BLOCKED % 4 < 3
            in_trans2 = (INPUT_TYPE4)(0);
#endif
            in_trans3 = (INPUT_TYPE4)(0);

            in0 = (INPUT_TYPE4)(in_trans0.s0, in_trans1.s0, in_trans2.s0, in_trans3.s0);
            in1 = (INPUT_TYPE4)(in_trans0.s1, in_trans1.s1, in_trans2.s1, in_trans3.s1);
            in2 = (INPUT_TYPE4)(in_trans0.s2, in_trans1.s2, in_trans2.s2, in_trans3.s2);
            in3 = (INPUT_TYPE4)(in_trans0.s3, in_trans1.s3, in_trans2.s3, in_trans3.s3);

            acc[oxi][0] = IMAD(acc[oxi][0], in0, wei0);
            acc[oxi][1] = IMAD(acc[oxi][1], in1, wei1);
            acc[oxi][2] = IMAD(acc[oxi][2], in2, wei2);
            acc[oxi][3] = IMAD(acc[oxi][3], in3, wei3);
        }
    }
#endif

#if FILTER_BLOCKED < FILTER_SPATIAL_SIZE
#   if PRELOAD_WEIGHTS
        wei0 = wei[FILTER_BLOCKED / 4 * 4 + 0];
        wei1 = wei[FILTER_BLOCKED / 4 * 4 + 1];
        wei2 = wei[FILTER_BLOCKED / 4 * 4 + 2];
        wei3 = wei[FILTER_BLOCKED / 4 * 4 + 3];
#   else
        wei0 = weights[weights_offset];
        wei1 = weights[weights_offset + 1 * WEIGHTS_I_PITCH];
        wei2 = weights[weights_offset + 2 * WEIGHTS_I_PITCH];
        wei3 = weights[weights_offset + 3 * WEIGHTS_I_PITCH];
#   endif

    unroll_for (uint fi = 0; fi < FILTER_SPATIAL_SIZE - FILTER_BLOCKED; ++fi) {
        uint fx = (fi + FILTER_BLOCKED) % FILTER_SIZE_X;
        uint fy = (fi + FILTER_BLOCKED) / FILTER_SIZE_X;

        unroll_for(uint oxi = 0; oxi < OUTPUT_BLOCK_X; ++oxi) {

#   if TILED
            in0 = AS_INPUT_TYPE4(_sub_group_shuffle(as_uint(in[fy]), (fx * DILATION_SIZE_X + tile_x * STRIDE_SIZE_X) % SIMD));
#   elif PRELOAD_INPUT
            uint input_x_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X);
            uint input_y_offset = fy * INPUT_LINE_SIZE;
            uint input_idx = input_x_offset + input_y_offset;
            in0 = in[input_idx];
#   else
            uint input_spatial_offset = (fx * DILATION_SIZE_X + oxi * STRIDE_SIZE_X) * (INPUT_X_PITCH / FSV)
                                        + fy * (DILATION_SIZE_Y * INPUT_Y_PITCH / FSV);
            uint input_idx = input_spatial_offset + input_offset;

            in0 = input[input_idx];
#   endif

            acc[oxi][0] += (int)in0.s0 * (int)wei0[fi];
            acc[oxi][1] += (int)in0.s1 * (int)wei1[fi];
            acc[oxi][2] += (int)in0.s2 * (int)wei2[fi];
            acc[oxi][3] += (int)in0.s3 * (int)wei3[fi];
        }
    }
#endif

#if TILE_Y != 1
    unroll_for (uint yi = 0; yi < FILTER_SIZE_Y - 1; ++yi) {
        unroll_for(uint xi = 0; xi < INPUT_LINE_SIZE; ++xi) {
            in[yi * INPUT_LINE_SIZE + xi] = in[(yi + 1) * INPUT_LINE_SIZE + xi];
        }
    }
    {
        uint yi = FILTER_SIZE_Y - 1;
        unroll_for(uint xi = 0; xi < INPUT_LINE_SIZE; ++xi) {
            in[yi * INPUT_LINE_SIZE + xi] = input[input_offset + xi * (INPUT_X_PITCH / FSV)];
        }
        input_offset += DILATION_SIZE_Y * INPUT_Y_PITCH / FSV;
    }
#endif

    for (uint oxi = 0; oxi < OUTPUT_BLOCK_X; ++oxi) {
        DEQUANTIZED_TYPE4 dequantized = (DEQUANTIZED_TYPE4)(
            TO_DEQUANTIZED_TYPE(acc[oxi][0]),
            TO_DEQUANTIZED_TYPE(acc[oxi][1]),
            TO_DEQUANTIZED_TYPE(acc[oxi][2]),
            TO_DEQUANTIZED_TYPE(acc[oxi][3]));

#if BIAS_TERM
        BIAS_TYPE4 bias;
#   if BIAS_PER_OUTPUT
        // TODO After adding static bias reorder we could use b_fs_yx_fsv4 format and single load
        uint bias_offset0 = GET_BIAS_INDEX(b, f + 0, y, x + oxi);
        uint bias_offset1 = GET_BIAS_INDEX(b, f + 1, y, x + oxi);
        uint bias_offset2 = GET_BIAS_INDEX(b, f + 2, y, x + oxi);
        uint bias_offset3 = GET_BIAS_INDEX(b, f + 3, y, x + oxi);
        bias.s0 = bias[bias_offset0];
        bias.s1 = bias[bias_offset1];
        bias.s2 = bias[bias_offset2];
        bias.s3 = bias[bias_offset3];
#   elif BIAS_PER_OFM
        uint bias_offset = f;
        bias = ((const __global BIAS_TYPE4*)(biases + bias_offset))[0];
#   else
#       error convolution_gpu_b_fs_yx_fsv4_dw.cl - not supported bias mode.
#   endif
        dequantized += TO_DEQUANTIZED_TYPE4(bias);
#endif

        OUTPUT_TYPE4 out;

#if HAS_FUSED_OPS
        FUSED_OPS_PRELOAD;
        FUSED_OPS_CALC;
        out = TO_OUTPUT_TYPE4(FUSED_OPS_RESULT);
#else
        out = TO_OUTPUT_TYPE4(dequantized);
#endif

#if TILED
        if (tile_x >= TILE_X || x + tile_x >= OUTPUT_SIZE_X)
            continue;
#elif OUTPUT_SIZE_X % OUTPUT_BLOCK_X != 0
        if (x + oxi >= OUTPUT_SIZE_X)
            break;
#endif

        if (OUTPUT_FEATURE_NUM % FSV != 0 && f + FSV >= OUTPUT_FEATURE_NUM) {
            if (OUTPUT_FEATURE_NUM % FSV <= 1)
                out.s1 = (OUTPUT_TYPE)(0);
            if (OUTPUT_FEATURE_NUM % FSV <= 2)
                out.s2 = (OUTPUT_TYPE)(0);
            out.s3 = (OUTPUT_TYPE)(0);
        }

        uint output_offset = GET_OUTPUT_INDEX(b, f, y, x + oxi + tile_x) / FSV;
        output[output_offset] = out;
    }

#if TILE_Y != 1
}
#endif
}

#undef FSV

#undef DEQUANTIZED_TYPE

#undef INPUT_TYPE4
#undef FILTER_TYPE4
#undef OUTPUT_TYPE4
#undef BIAS_TYPE4
#undef DEQUANTIZED_TYPE4

#undef AS_INPUT_TYPE4
#undef TO_DEQUANTIZED_TYPE
#undef TO_DEQUANTIZED_TYPE4
#undef TO_OUTPUT_TYPE4

#undef GET_INPUT_INDEX
#undef GET_WEIGHTS_INDEX
#undef GET_OUTPUT_INDEX
#undef GET_BIAS_INDEX

#undef INPUT_X_PITCH
#undef INPUT_Y_PITCH

#undef WEIGHTS_I_PITCH
#undef WEIGHTS_YXS_PITCH

#undef FILTER_SPATIAL_SIZE
