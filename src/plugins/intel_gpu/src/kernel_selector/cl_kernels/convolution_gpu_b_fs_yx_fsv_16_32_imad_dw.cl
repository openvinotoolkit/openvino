// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/imad.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

// ======================================================================================
// Host side jit-constants:
// ======================================================================================
// SIMD   [{8, 16}] - Sub-group/simd size for the kernel. Used as third dimension of
//                    local work size.
// TILE_X [uint] - Number of output values along x dimension calculated by single
//                 work-item/sub-group.
// LWS0 [uint] - Local work size 0th dimension.
// LWS1 [uint] - Local work size 1st dimension.
// FILTER_BLOCKED - Number of filter spatial elements to process using IMAD. Must be less
//                  or equal to total filter spatial size.
//                  Currently only supported to be multiple of 4.
// PRELOAD_INPUT_TO_SLM - Flag indicating that input memory for entire work-group should
//                        first be loaded into slm. This preloaded input will be then
//                        used in computation loop to take advantage of overlapping regions.
// CHECK_BOUNDARY - Flag indicating that input has virtual padding and there is either no
//                  physical padding or it can't be used.
// ======================================================================================
// Supported operations:
// input/output format: any b_fs_yx_fsv<k> - where <k> >= SIMD,
//                      input and output formats must be the same
// weights format:      os_i_yxs_oxv<k>_yxsv4 - where <k> same as in input format
// input data types:   uchar8, char8
// weights data types: uchar8, char8
// output data types:  uchar8, char8, half, float
// asymetric quantization: weights zero points, compensation term, activation zero points
// ======================================================================================

// ======================================================================================
// Definitions not exposed to host:
// ======================================================================================
// Performs bounds checking when loading to SLM and doesn't load input padding
// Valid only together with PRELOAD_INPUT_TO_SLM
#define CHECK_BOUNDARY_IN_SLM (CHECK_BOUNDARY && PRELOAD_INPUT_TO_SLM)
// ======================================================================================

#if OUTPUT_LAYOUT_B_FS_YX_FSV16
#   define FSV 16
#elif OUTPUT_LAYOUT_B_FS_YX_FSV32
#   define FSV 32
#else
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - unsupported output layout.
#endif

#if FSV == SIMD
#   define F_PER_WI 1
#elif FSV == (2 * SIMD)
#   define F_PER_WI 2
#elif FSV == (4 * SIMD)
#   define F_PER_WI 4
#else
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - unsupported layout x simd combination.
#endif

#define DEQUANTIZED_TYPE  float
#define DEQUANTIZED_TYPE2 MAKE_VECTOR_TYPE(DEQUANTIZED_TYPE, 2)
#define DEQUANTIZED_TYPE4 MAKE_VECTOR_TYPE(DEQUANTIZED_TYPE, 4)

#define INPUT_TYPE        INPUT0_TYPE
#define INPUT_TYPE2       MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8       MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define INPUT_TYPE16      MAKE_VECTOR_TYPE(INPUT0_TYPE, 16)

#define FILTER_TYPE4      MAKE_VECTOR_TYPE(FILTER_TYPE, 4)

#define OUTPUT_TYPE2      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2)
#define OUTPUT_TYPE4      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define OUTPUT_TYPE8      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)
#define OUTPUT_TYPE16     MAKE_VECTOR_TYPE(OUTPUT_TYPE, 16)

#define AS_FILTER_TYPE4(val)      CAT(as_, FILTER_TYPE4)(val)

#define TO_DEQUANTIZED_TYPE(val)  CAT(convert_, DEQUANTIZED_TYPE)(val)

#define GET_INPUT_INDEX(b, f, y, x)    INPUT0_GET_INDEX(b, f, y, x)
#if FSV == 16
#   define GET_WEIGHTS_INDEX(g, o, i, y, x)  GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(FILTER, g, 0, 0, y, x)
#else
#   define GET_WEIGHTS_INDEX(g, o, i, y, x)  GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(FILTER, g, 0, 0, y, x)
#endif
#define GET_OUTPUT_INDEX(b, f, y, x)   OUTPUT_GET_INDEX(b, f, y, x)
#define GET_BIAS_INDEX(b, f, y, x)     BIAS_GET_INDEX(b, f, y, x)

#define INPUT_X_PITCH FSV
#define INPUT_Y_PITCH (FSV * (INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X))

#define WEIGHTS_YXS_PITCH (4 * FSV)

#define FILTER_SPATIAL_SIZE (FILTER_SIZE_X * FILTER_SIZE_Y)

#define MAX_OPT_BLOCK_WRITE_BYTES ((16 * 2 * 4) / SIMD)

#define VEC_TO_ARRAY_2(arr, vec, offset)                \
    (arr)[(offset) + 0] = (vec).s0;                     \
    (arr)[(offset) + 1] = (vec).s1
#define VEC_TO_ARRAY_4(arr, vec, offset)                \
    VEC_TO_ARRAY_2(arr, (vec).s01, offset);             \
    VEC_TO_ARRAY_2(arr, (vec).s23, (offset) + 2)
#define VEC_TO_ARRAY_8(arr, vec, offset)                \
    VEC_TO_ARRAY_4(arr, (vec).s0123, offset);           \
    VEC_TO_ARRAY_4(arr, (vec).s4567, (offset) + 4)
#define VEC_TO_ARRAY_16(arr, vec, offset)               \
    VEC_TO_ARRAY_8(arr, (vec).s01234567, offset);       \
    VEC_TO_ARRAY_8(arr, (vec).s89abcdef, (offset) + 8)

#define ARRAY_TO_VEC_2(vec, arr, offset)                \
    (vec).s0 = (arr)[(offset)];                         \
    (vec).s1 = (arr)[(offset) + 1]
#define ARRAY_TO_VEC_4(vec, arr, offset)                \
    ARRAY_TO_VEC_2((vec).s01, arr, offset);             \
    ARRAY_TO_VEC_2((vec).s23, arr, (offset) + 2)
#define ARRAY_TO_VEC_8(vec, arr, offset)                \
    ARRAY_TO_VEC_4((vec).s0123, arr, offset);           \
    ARRAY_TO_VEC_4((vec).s4567, arr, (offset) + 4)
#define ARRAY_TO_VEC_16(vec, arr, offset)               \
    ARRAY_TO_VEC_8((vec).s01234567, arr, offset);       \
    ARRAY_TO_VEC_8((vec).s89abcdef, arr, (offset) + 8)

#ifndef OUTPUT_PAD_VALUE
#   define OUTPUT_PAD_VALUE (OUTPUT_TYPE)(0)
#   define OUTPUT_PAD_VALUE_undef
#endif

#if PRELOAD_INPUT_TO_SLM
#   define INPUT_BLOCK_READN(vec_size, ptr, offset)   BLOCK_READN_SLM(INPUT0_TYPE, vec_size, ptr, offset)
#else
#   define INPUT_BLOCK_READN(vec_size, ptr, offset)   BLOCK_READN(INPUT0_TYPE, vec_size, ptr, offset)
#endif

// Defines validation
#if FILTER_BLOCKED % 4 != 0
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - FILTER_BLOCKED must be multiple of 4.
#endif

#if !PRELOAD_INPUT_TO_SLM && CHECK_BOUNDARY_IN_SLM
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - internal error, CHECK_BOUNDARY_IN_SLM enabled without PRELOAD_INPUT_TO_SLM.
#endif


REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS0, LWS1, SIMD)))
KERNEL(convolution)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global  INPUT0_TYPE  *input,
    __global        OUTPUT_TYPE  *output,
    const __global  FILTER_TYPE  *weights
#if BIAS_TERM
    , const __global BIAS_TYPE     *biases
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
#endif
#if COMPENSATION_TERM
    , const __global COMPENSATION_TYPE *compensation
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
#if LWS0 == 1
    uint x = (uint)get_group_id(0) * TILE_X;
#else
    uint x = ((uint)get_group_id(0) * LWS0 + (uint)get_sub_group_id() % LWS0) * TILE_X;
    x = sub_group_broadcast(x, 0);  // Why in the world get_sub_group_id() is not sub-group uniform?
#endif
#if LWS1 == 1
    uint y = get_group_id(1);
#else
    uint y = (uint)get_group_id(1) * LWS1 + (uint)get_sub_group_id() / LWS0;
    y = sub_group_broadcast(y, 0);  // Why in the world get_sub_group_id() is not sub-group uniform?
#endif
    uint bf = get_group_id(2);
    uint b = bf % OUTPUT_BATCH_NUM;
    uint f = bf / OUTPUT_BATCH_NUM * FSV;

    // For asymmetric data hints can save some comparisions (ex. 3x3 kernel with 1 padding -> no need to check padding correction at fx=1 fy=1)
    ASSUME_HINT(x < OUTPUT_SIZE_X);
    ASSUME_HINT(y < OUTPUT_SIZE_Y);

#if PRELOAD_INPUT_TO_SLM
    const uint wg_tile_x = TILE_X * LWS0;
    const uint wg_tile_y = LWS1;
    const uint wg_in_tile_x = (wg_tile_x - 1) * STRIDE_SIZE_X + (FILTER_SIZE_X - 1) * DILATION_SIZE_X + 1;
    const uint wg_in_tile_y = LWS1 == 1 ? FILTER_SIZE_Y : (wg_tile_y - 1) * STRIDE_SIZE_Y + (FILTER_SIZE_Y - 1) * DILATION_SIZE_Y + 1;
    const uint slm_total_size = wg_in_tile_x * wg_in_tile_y * FSV;
    __local INPUT0_TYPE input_slm[slm_total_size];
    // Preload input to slm
    {
        const int wg_x = (uint)get_group_id(0) * LWS0 * TILE_X;
        const int wg_y = (uint)get_group_id(1) * LWS1;

        uint input_offset_base = GET_INPUT_INDEX(b, f, wg_y * STRIDE_SIZE_Y - PADDING_SIZE_Y, wg_x * STRIDE_SIZE_X - PADDING_SIZE_X);
        const uint iteration_preload_bytes = 16;  // Load 16 bytes into slm for work-item for one loop iteration

    #if ASYMMETRIC_DATA_QUANTIZATION && CHECK_BOUNDARY_IN_SLM
        uint4 azp_uniform[FSV / iteration_preload_bytes];
        unroll_for(uint i = 0; i < FSV / iteration_preload_bytes; ++i) {
            azp_uniform[i] = ((const __global uint4*)(activations_zp + (f + i * iteration_preload_bytes)))[0];
        }
    #endif

        uint in_s = get_sub_group_id() * SIMD * iteration_preload_bytes + get_sub_group_local_id() * iteration_preload_bytes;
        ASSUME_HINT(in_s < (LWS0 * LWS1 * SIMD * iteration_preload_bytes + SIMD * iteration_preload_bytes));  // Can get rid of unnecessary if (3 instructions)
        for (; in_s < slm_total_size; in_s += LWS0 * LWS1 * SIMD * iteration_preload_bytes) {
            uint input_f = in_s % FSV;
            uint input_x = in_s / FSV % wg_in_tile_x;
            uint input_y = in_s / FSV / wg_in_tile_x;
            if (LWS1 == 1)
                input_y *= DILATION_SIZE_Y;
            uint input_idx = input_offset_base + input_f + input_x * INPUT_X_PITCH + input_y * INPUT_Y_PITCH;

            uint4 tmp = 0;
        #if CHECK_BOUNDARY_IN_SLM
            int pad_x = wg_x * STRIDE_SIZE_X + (int)input_x < PADDING_SIZE_X || wg_x * STRIDE_SIZE_X + (int)input_x >= INPUT0_SIZE_X + PADDING_SIZE_X;
            int pad_y = wg_y * STRIDE_SIZE_Y + (int)input_y < PADDING_SIZE_Y || wg_y * STRIDE_SIZE_Y + (int)input_y >= INPUT0_SIZE_Y + PADDING_SIZE_Y;

            if (pad_x || pad_y) {
            #if ASYMMETRIC_DATA_QUANTIZATION
                #if FSV == 16
                    tmp = azp_uniform[0];
                #else // -> FSV == 32
                    tmp = (input_f == 0) ? azp_uniform[0] : azp_uniform[1];
                #endif
            #endif
            } else
        #endif
            {
                tmp = ((const __global uint4*)(input + input_idx))[0];
            }

            ((__local uint4*)(input_slm + in_s))[0] = tmp;
        }
    }
    const uint input_x_pitch = FSV;
    const uint input_y_pitch = wg_in_tile_x * FSV;
    const uint dilation_size_y = (LWS1 == 1) ? 1 : DILATION_SIZE_Y;

    uint input_offset = 0;
    if (LWS0 != 1)
        input_offset += ((uint)get_sub_group_id() % LWS0) * TILE_X * STRIDE_SIZE_X * input_x_pitch;
    if (LWS1 != 1)
        input_offset += ((uint)get_sub_group_id() / LWS0) * STRIDE_SIZE_Y * input_y_pitch;
    const __local INPUT0_TYPE* input_ptr = input_slm;

    barrier(CLK_LOCAL_MEM_FENCE);
#else
    const uint input_x_pitch = INPUT_X_PITCH;
    const uint input_y_pitch = INPUT_Y_PITCH;
    const uint dilation_size_y = DILATION_SIZE_Y;
    uint input_offset = GET_INPUT_INDEX(b, f, (int)y * STRIDE_SIZE_Y - PADDING_SIZE_Y, (int)x * STRIDE_SIZE_X - PADDING_SIZE_X);
    const __global INPUT0_TYPE* input_ptr = input;
#endif

#if ASYMMETRIC_DATA_QUANTIZATION && CHECK_BOUNDARY && !CHECK_BOUNDARY_IN_SLM
    MAKE_VECTOR_TYPE(ACTIVATIONS_ZERO_POINTS_TYPE, F_PER_WI) azp =
        BLOCK_READN(ACTIVATIONS_ZERO_POINTS_TYPE, F_PER_WI, activations_zp, f);
#endif

    uint weights_offset = GET_WEIGHTS_INDEX(f, 0, 0, 0, 0);

    int acc[TILE_X * F_PER_WI] = { };
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    int src_sum[TILE_X * F_PER_WI] = { };
#endif

    bool early_return = false;
    early_return |= CEIL_DIV(OUTPUT_SIZE_X, TILE_X) % LWS0 != 0 && x >= OUTPUT_SIZE_X;
    early_return |= OUTPUT_SIZE_Y % LWS1 != 0 && y >= OUTPUT_SIZE_Y;
    if (early_return)
        return;

    unroll_for (uint fi = 0; fi < FILTER_BLOCKED / 4 * 4; fi += 4) {
        // Loop over 4 filter spatials that match imad case
        uint4 fis = (uint4)(fi, fi + 1, fi + 2, fi + 3);

        uint4 fx = fis % FILTER_SIZE_X;
        uint4 fy = fis / FILTER_SIZE_X;

        // Input loading:
        INPUT_TYPE in_trans0[TILE_X * F_PER_WI];
        INPUT_TYPE in_trans1[TILE_X * F_PER_WI];
        INPUT_TYPE in_trans2[TILE_X * F_PER_WI];
        INPUT_TYPE in_trans3[TILE_X * F_PER_WI];
#if STRIDE_SIZE_X == 1
        // Without strides block reads can be used to load whole TILE_X inputs
        // Block read ladder to select optimal combination of block reads for TILE_X
        uint4 input_x_offset = fx * (DILATION_SIZE_X * input_x_pitch);
        uint4 input_y_offset = fy * (dilation_size_y * input_y_pitch);
        uint4 input_spatial_offset = input_x_offset + input_y_offset;
        uint4 input_idx = input_spatial_offset + input_offset;

        uint tx = 0;
        unroll_for(; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            INPUT_TYPE16 tmp_in0 = INPUT_BLOCK_READN(16, input_ptr, input_idx.s0);
            INPUT_TYPE16 tmp_in1 = INPUT_BLOCK_READN(16, input_ptr, input_idx.s1);
            INPUT_TYPE16 tmp_in2 = INPUT_BLOCK_READN(16, input_ptr, input_idx.s2);
            INPUT_TYPE16 tmp_in3 = INPUT_BLOCK_READN(16, input_ptr, input_idx.s3);

            VEC_TO_ARRAY_16(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_16(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_16(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_16(in_trans3, tmp_in3, tx);

            input_idx += 16 * SIMD;
        }
        if (TILE_X * F_PER_WI % 16 >= 8) {
            INPUT_TYPE8 tmp_in0 = INPUT_BLOCK_READN(8, input_ptr, input_idx.s0);
            INPUT_TYPE8 tmp_in1 = INPUT_BLOCK_READN(8, input_ptr, input_idx.s1);
            INPUT_TYPE8 tmp_in2 = INPUT_BLOCK_READN(8, input_ptr, input_idx.s2);
            INPUT_TYPE8 tmp_in3 = INPUT_BLOCK_READN(8, input_ptr, input_idx.s3);

            VEC_TO_ARRAY_8(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_8(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_8(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_8(in_trans3, tmp_in3, tx);

            input_idx += 8 * SIMD;
            tx += 8;
        }
        if (TILE_X * F_PER_WI % 8 >= 4) {
            INPUT_TYPE4 tmp_in0 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s0);
            INPUT_TYPE4 tmp_in1 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s1);
            INPUT_TYPE4 tmp_in2 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s2);
            INPUT_TYPE4 tmp_in3 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s3);

            VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_4(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_4(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_4(in_trans3, tmp_in3, tx);

            input_idx += 4 * SIMD;
            tx += 4;
        }
        if (TILE_X * F_PER_WI % 4 >= 2) {
            INPUT_TYPE2 tmp_in0 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s0);
            INPUT_TYPE2 tmp_in1 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s1);
            INPUT_TYPE2 tmp_in2 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s2);
            INPUT_TYPE2 tmp_in3 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s3);

            VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_2(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_2(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_2(in_trans3, tmp_in3, tx);

            input_idx += 2 * SIMD;
            tx += 2;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            in_trans0[tx] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s0);
            in_trans1[tx] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s1);
            in_trans2[tx] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s2);
            in_trans3[tx] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s3);
        }
#else
        uint4 input_x_offset = fx * DILATION_SIZE_X * input_x_pitch;
        uint4 input_y_offset = fy * dilation_size_y * input_y_pitch;
        uint4 input_spatial_offset = input_x_offset + input_y_offset;
        uint4 input_start_offset = input_spatial_offset + input_offset;
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            uint4 input_idx = input_start_offset + tx * STRIDE_SIZE_X * input_x_pitch;
            // Block reads along feature slice
            uint fw = 0;
            unroll_for(; fw + 4 <= F_PER_WI; fw += 4) {
                INPUT_TYPE4 tmp_in0 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s0);
                INPUT_TYPE4 tmp_in1 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s1);
                INPUT_TYPE4 tmp_in2 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s2);
                INPUT_TYPE4 tmp_in3 = INPUT_BLOCK_READN(4, input_ptr, input_idx.s3);

                VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans1, tmp_in1, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans2, tmp_in2, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans3, tmp_in3, tx * F_PER_WI + fw);

                input_idx += 4 * SIMD;
            }
            if (F_PER_WI % 4 >= 2) {
                INPUT_TYPE2 tmp_in0 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s0);
                INPUT_TYPE2 tmp_in1 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s1);
                INPUT_TYPE2 tmp_in2 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s2);
                INPUT_TYPE2 tmp_in3 = INPUT_BLOCK_READN(2, input_ptr, input_idx.s3);

                VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans1, tmp_in1, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans2, tmp_in2, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans3, tmp_in3, tx * F_PER_WI + fw);

                input_idx += 2 * SIMD;
                fw += 2;
            }
            if (F_PER_WI % 2 == 1) {
                in_trans0[tx * F_PER_WI + fw] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s0);
                in_trans1[tx * F_PER_WI + fw] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s1);
                in_trans2[tx * F_PER_WI + fw] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s2);
                in_trans3[tx * F_PER_WI + fw] = INPUT_BLOCK_READN(1, input_ptr, input_idx.s3);
            }
        }
#endif
        // Weights loading:
        FILTER_TYPE4 wei[F_PER_WI];
        unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
            wei[fw] = AS_FILTER_TYPE4(_sub_group_block_read((const __global uint*)(weights + weights_offset) + fw * SIMD));
        }

    #if CHECK_BOUNDARY && !CHECK_BOUNDARY_IN_SLM
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            int4 input_x = convert_int4(x * STRIDE_SIZE_X + tx * STRIDE_SIZE_X + fx * DILATION_SIZE_X) - PADDING_SIZE_X;
            int4 input_y = convert_int4(y * STRIDE_SIZE_Y + fy * dilation_size_y) - PADDING_SIZE_Y;
            int4 input_pad = input_x < 0 || input_x >= INPUT0_SIZE_X || input_y < 0 || input_y >= INPUT0_SIZE_Y;
        #if ASYMMETRIC_DATA_QUANTIZATION
            #define padding_value(fw) (((ACTIVATIONS_ZERO_POINTS_TYPE*)&azp)[fw])
        #else
            #define padding_value(fw) ((INPUT0_TYPE)0)
        #endif
            unroll_for(uint fwp = 0; fwp < F_PER_WI; ++fwp) {
                in_trans0[tx * F_PER_WI + fwp] = input_pad.s0 ? padding_value(fwp) : in_trans0[tx * F_PER_WI + fwp];
            }
            unroll_for(uint fwp = 0; fwp < F_PER_WI; ++fwp) {
                in_trans1[tx * F_PER_WI + fwp] = input_pad.s1 ? padding_value(fwp) : in_trans1[tx * F_PER_WI + fwp];
            }
            unroll_for(uint fwp = 0; fwp < F_PER_WI; ++fwp) {
                in_trans2[tx * F_PER_WI + fwp] = input_pad.s2 ? padding_value(fwp) : in_trans2[tx * F_PER_WI + fwp];
            }
            unroll_for(uint fwp = 0; fwp < F_PER_WI; ++fwp) {
                in_trans3[tx * F_PER_WI + fwp] = input_pad.s3 ? padding_value(fwp) : in_trans3[tx * F_PER_WI + fwp];
            }
        #undef padding_value
        }
    #endif

        // Transpose input:
        INPUT_TYPE4 in[TILE_X * F_PER_WI];
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                uint in_offset = tx * F_PER_WI + fw;
                in[in_offset] = (INPUT_TYPE4)(in_trans0[in_offset], in_trans1[in_offset], in_trans2[in_offset], in_trans3[in_offset]);
            }
        }

        // IMAD:
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                acc[tx * F_PER_WI + fw] = IMAD(acc[tx * F_PER_WI + fw], in[tx * F_PER_WI + fw], wei[fw]);
            }
        }

    #if ASYMMETRIC_WEIGHTS_QUANTIZATION
        // Accumulate for input values for asymmetric weights:
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                src_sum[tx * F_PER_WI + fw] = IMAD(src_sum[tx * F_PER_WI + fw], in[tx * F_PER_WI + fw], (char4)(1, 1, 1, 1));
            }
        }
    #endif

        weights_offset += WEIGHTS_YXS_PITCH;
    }


#if FILTER_BLOCKED < FILTER_SPATIAL_SIZE
    // Leftovers in filters spatial - use raw multiplication instead of imad
    // Load inputs before loop to avoid byte scattered reads + there are at most 3 leftovers
    FILTER_TYPE4 wei[F_PER_WI];
    unroll_for (uint fw = 0; fw < F_PER_WI; ++fw) {
        wei[fw] = AS_FILTER_TYPE4(_sub_group_block_read((const __global uint*)(weights + weights_offset) + fw * SIMD));
    }

    unroll_for (uint fi = 0; fi < FILTER_SPATIAL_SIZE - FILTER_BLOCKED; ++fi) {
        // Input loading:
        uint fx = (fi + FILTER_BLOCKED) % FILTER_SIZE_X;
        uint fy = (fi + FILTER_BLOCKED) / FILTER_SIZE_X;

        INPUT_TYPE in_trans0[TILE_X * F_PER_WI];
#   if STRIDE_SIZE_X == 1
        uint input_x_offset = fx * (DILATION_SIZE_X * input_x_pitch);
        uint input_y_offset = fy * (dilation_size_y * input_y_pitch);
        uint input_spatial_offset = input_x_offset + input_y_offset;
        uint input_idx = input_spatial_offset + input_offset;

        uint tx = 0;
        unroll_for(; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            INPUT_TYPE16 tmp_in0 = INPUT_BLOCK_READN(16, input_ptr, input_idx);
            VEC_TO_ARRAY_16(in_trans0, tmp_in0, tx);
            input_idx += 16 * SIMD;
        }
        if (TILE_X * F_PER_WI % 16 >= 8) {
            INPUT_TYPE8 tmp_in0 = INPUT_BLOCK_READN(8, input_ptr, input_idx);
            VEC_TO_ARRAY_8(in_trans0, tmp_in0, tx);
            input_idx += 8 * SIMD;
            tx += 8;
        }
        if (TILE_X * F_PER_WI % 8 >= 4) {
            INPUT_TYPE4 tmp_in0 = INPUT_BLOCK_READN(4, input_ptr, input_idx);
            VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx);
            input_idx += 4 * SIMD;
            tx += 4;
        }
        if (TILE_X * F_PER_WI % 4 >= 2) {
            INPUT_TYPE2 tmp_in0 = INPUT_BLOCK_READN(2, input_ptr, input_idx);
            VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx);
            input_idx += 2 * SIMD;
            tx += 2;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            in_trans0[tx] = INPUT_BLOCK_READN(1, input_ptr, input_idx);
        }
#   else
        uint input_x_offset = fx * DILATION_SIZE_X * input_x_pitch;
        uint input_y_offset = fy * dilation_size_y * input_y_pitch;
        uint input_spatial_offset = input_x_offset + input_y_offset;
        uint input_start_offset = input_spatial_offset + input_offset;
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            uint input_idx = input_start_offset + tx * STRIDE_SIZE_X * input_x_pitch;
            uint fw = 0;
            unroll_for(; fw + 4 <= F_PER_WI; fw += 4) {
                INPUT_TYPE4 tmp_in0 = INPUT_BLOCK_READN(4, input_ptr, input_idx);
                VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                input_idx += 4 * SIMD;
            }
            if (F_PER_WI % 4 >= 2) {
                INPUT_TYPE2 tmp_in0 = INPUT_BLOCK_READN(2, input_ptr, input_idx);
                VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                input_idx += 2 * SIMD;
                fw += 2;
            }
            if (F_PER_WI % 2 == 1) {
                in_trans0[tx * F_PER_WI + fw] = INPUT_BLOCK_READN(1, input_ptr, input_idx);
            }
        }
#   endif

    #if CHECK_BOUNDARY && !CHECK_BOUNDARY_IN_SLM
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            int input_x = (x + tx) * STRIDE_SIZE_X + fx * DILATION_SIZE_X - PADDING_SIZE_X;
            int input_y = y * STRIDE_SIZE_Y + fy * dilation_size_y - PADDING_SIZE_Y;
            int input_pad = input_x < 0 || input_x >= INPUT0_SIZE_X || input_y < 0 || input_y >= INPUT0_SIZE_Y;
        #if ASYMMETRIC_DATA_QUANTIZATION
            #define padding_value(fw) (((ACTIVATIONS_ZERO_POINTS_TYPE*)&azp)[fw])
        #else
            #define padding_value(fw) ((INPUT0_TYPE)0)
        #endif
            unroll_for(uint fwp = 0; fwp < F_PER_WI; ++fwp) {
                in_trans0[tx * F_PER_WI + fwp] = input_pad ? padding_value(fwp) : in_trans0[tx * F_PER_WI + fwp];
            }
        #undef padding_value
        }
    #endif

        // Raw multiply accumulate:
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                acc[tx * F_PER_WI + fw] += (int)in_trans0[tx * F_PER_WI + fw] * (int)wei[fw][fi];
            }
        }

    #if ASYMMETRIC_WEIGHTS_QUANTIZATION
        // Accumulate input values for asymmetric weights:
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                src_sum[tx * F_PER_WI + fw] += (int)in_trans0[tx * F_PER_WI + fw];
            }
        }
    #endif
    }
#endif

    DEQUANTIZED_TYPE dequantized[TILE_X * F_PER_WI];
    for (uint tx = 0; tx < TILE_X * F_PER_WI; ++tx) {
        dequantized[tx] = TO_DEQUANTIZED_TYPE(acc[tx]);
    }

#if BIAS_TERM
#   if BIAS_PER_OFM
    MAKE_VECTOR_TYPE(BIAS_TYPE, F_PER_WI) bias_val = BLOCK_READN(BIAS_TYPE, F_PER_WI, biases, f);
    unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
        unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
            dequantized[tx * F_PER_WI + fw] += TO_DEQUANTIZED_TYPE(((BIAS_TYPE*)&bias_val)[fw]);
        }
    }
#   elif BIAS_PER_OUTPUT
    unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
        unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
            uint bias_offset = GET_BIAS_INDEX(b, f + fw * SIMD + get_sub_group_local_id(), y, x + tx);
            BIAS_TYPE bias = biases[bias_offset];
            dequantized[tx * F_PER_WI + fw] += TO_DEQUANTIZED_TYPE(bias);
        }
    }
#   else
#       error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - unsupported bias mode.
#   endif
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    {
        MAKE_VECTOR_TYPE(WEIGHTS_ZERO_POINTS_TYPE, F_PER_WI) wzp = BLOCK_READN(WEIGHTS_ZERO_POINTS_TYPE, F_PER_WI, weights_zp, f);
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                dequantized[tx * F_PER_WI + fw] -= TO_DEQUANTIZED_TYPE(src_sum[tx * F_PER_WI + fw]) * TO_DEQUANTIZED_TYPE(((WEIGHTS_ZERO_POINTS_TYPE*)&wzp)[fw]);
            }
        }
    }
#endif

#if COMPENSATION_TERM
    {
        MAKE_VECTOR_TYPE(COMPENSATION_TYPE, F_PER_WI) comp = BLOCK_READN(COMPENSATION_TYPE, F_PER_WI, compensation, f);
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                dequantized[tx * F_PER_WI + fw] += TO_DEQUANTIZED_TYPE(((COMPENSATION_TYPE*)&comp)[fw]);
            }
        }
    }
#endif

    OUTPUT_TYPE out[TILE_X * F_PER_WI];
    // Fused ops and conversion to output type
    unroll_for (uint tx = 0; tx < TILE_X; ++tx) {
#if HAS_FUSED_OPS
        uint fused_ops_x = x + tx;
        uint fused_ops_f = f;
        uint fw = 0;
        unroll_for(; fw + 4 <= F_PER_WI; fw += 4) {
            DEQUANTIZED_TYPE4 fused_ops_in;
            ARRAY_TO_VEC_4(fused_ops_in, dequantized, tx * F_PER_WI + fw);
            FUSED_OPS_4;
            VEC_TO_ARRAY_4(out, FUSED_OPS_RESULT_4, tx * F_PER_WI + fw);
            fused_ops_f += 4 * SIMD;
        }
        if (F_PER_WI % 4 >= 2) {
            DEQUANTIZED_TYPE2 fused_ops_in;
            ARRAY_TO_VEC_2(fused_ops_in, dequantized, tx * F_PER_WI + fw);
            FUSED_OPS_2;
            VEC_TO_ARRAY_2(out, FUSED_OPS_RESULT_2, tx * F_PER_WI + fw);
            fw += 2;
            fused_ops_f += 2 * SIMD;
        }
        if (F_PER_WI % 2 == 1) {
            DEQUANTIZED_TYPE fused_ops_in;
            fused_ops_in = dequantized[tx * F_PER_WI + fw];
            FUSED_OPS_1;
            out[tx * F_PER_WI + fw] = FUSED_OPS_RESULT_1;
        }
#else
        unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
            out[tx * F_PER_WI + fw] = TO_OUTPUT_TYPE(dequantized[tx * F_PER_WI + fw]);
        }
#endif
    }

    // Fill results outside output in features with OUTPUT_PAD_VALUE.
    if (OUTPUT_FEATURE_NUM % FSV != 0 && f + FSV > OUTPUT_FEATURE_NUM) {
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            unroll_for(uint fw = 0; fw < F_PER_WI; ++fw) {
                const uint sglid = get_sub_group_local_id();
                // Hint here can save some movs if features are divisible by SIMD and not by FSV
                ASSUME_HINT(sglid < SIMD);
                bool outside = fw * SIMD + get_sub_group_local_id() >= OUTPUT_FEATURE_NUM % FSV;
                out[tx * F_PER_WI + fw] = outside ? OUTPUT_PAD_VALUE : out[tx * F_PER_WI + fw];
            }
        }
    }

    uint output_offset = GET_OUTPUT_INDEX(b, f, y, x);

    if (OUTPUT_SIZE_X % TILE_X == 0 || x + TILE_X <= OUTPUT_SIZE_X) {
        // Full output tile x write using block write ladder
        uint tx = 0;
    #if OUTPUT_TYPE_SIZE * 16 <= MAX_OPT_BLOCK_WRITE_BYTES
        unroll_for(; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            OUTPUT_TYPE16 tmp_write;
            ARRAY_TO_VEC_16(tmp_write, out, tx);
            DT_OUTPUT_BLOCK_WRITE16(output, output_offset, tmp_write);
            output_offset += 16 * SIMD;
        }
    #endif
    #if OUTPUT_TYPE_SIZE * 8 <= MAX_OPT_BLOCK_WRITE_BYTES
        unroll_for(; tx + 8 <= TILE_X * F_PER_WI; tx += 8) {
            OUTPUT_TYPE8 tmp_write;
            ARRAY_TO_VEC_8(tmp_write, out, tx);
            DT_OUTPUT_BLOCK_WRITE8(output, output_offset, tmp_write);
            output_offset += 8 * SIMD;
        }
    #endif
    #if OUTPUT_TYPE_SIZE * 4 <= MAX_OPT_BLOCK_WRITE_BYTES
        unroll_for(; tx + 4 <= TILE_X * F_PER_WI; tx += 4) {
            OUTPUT_TYPE4 tmp_write;
            ARRAY_TO_VEC_4(tmp_write, out, tx);
            DT_OUTPUT_BLOCK_WRITE4(output, output_offset, tmp_write);
            output_offset += 4 * SIMD;
        }
    #endif
        unroll_for(; tx + 2 <= TILE_X * F_PER_WI; tx += 2) {
            OUTPUT_TYPE2 tmp_write;
            ARRAY_TO_VEC_2(tmp_write, out, tx);
            DT_OUTPUT_BLOCK_WRITE2(output, output_offset, tmp_write);
            output_offset += 2 * SIMD;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, out[tx]);
        }
    } else {
        // Leftovers write, block writes in f dimension only
        unroll_for(uint tx = 0; tx < TILE_X; ++tx) {
            if (tx < OUTPUT_SIZE_X % TILE_X) {
                uint fw = 0;
            #if OUTPUT_TYPE_SIZE * 4 <= MAX_OPT_BLOCK_WRITE_BYTES
                unroll_for(; fw + 4 <= F_PER_WI; fw += 4) {
                    OUTPUT_TYPE4 tmp_write;
                    ARRAY_TO_VEC_4(tmp_write, out, tx * F_PER_WI + fw);
                    DT_OUTPUT_BLOCK_WRITE4(output, output_offset + fw * SIMD, tmp_write);
                }
            #endif
                unroll_for(; fw + 2 <= F_PER_WI; fw += 2) {
                    OUTPUT_TYPE2 tmp_write;
                    ARRAY_TO_VEC_2(tmp_write, out, tx * F_PER_WI + fw);
                    DT_OUTPUT_BLOCK_WRITE2(output, output_offset + fw * SIMD, tmp_write);
                }
                if (F_PER_WI % 2 == 1) {
                    DT_OUTPUT_BLOCK_WRITE(output, output_offset + fw * SIMD, out[tx * F_PER_WI + fw]);
                }
            }
            output_offset += FSV;
        }
    }
}

#undef CHECK_BOUNDARY_IN_SLM

#undef FSV

#undef F_PER_WI

#undef DEQUANTIZED_TYPE
#undef DEQUANTIZED_TYPE2
#undef DEQUANTIZED_TYPE4

#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8
#undef INPUT_TYPE16

#undef FILTER_TYPE4

#undef OUTPUT_TYPE2
#undef OUTPUT_TYPE4
#undef OUTPUT_TYPE8
#undef OUTPUT_TYPE16

#undef AS_FILTER_TYPE

#undef TO_DEQUANTIZED_TYPE

#undef GET_INPUT_INDEX
#undef GET_WEIGHTS_INDEX
#undef GET_OUTPUT_INDEX

#undef INPUT_X_PITCH
#undef INPUT_Y_PITCH

#undef WEIGHTS_YXS_PITCH

#undef FILTER_SPATIAL_SIZE

#undef MAX_OPT_BLOCK_WRITE_BYTES

#undef VEC_TO_ARRAY_2
#undef VEC_TO_ARRAY_4
#undef VEC_TO_ARRAY_8
#undef VEC_TO_ARRAY_16

#undef ARRAY_TO_VEC_2
#undef ARRAY_TO_VEC_4
#undef ARRAY_TO_VEC_8
#undef ARRAY_TO_VEC_16

#ifdef OUTPUT_PAD_VALUE_undef
#   undef OUTPUT_PAD_VALUE
#   undef OUTPUT_PAD_VALUE_undef
#endif
