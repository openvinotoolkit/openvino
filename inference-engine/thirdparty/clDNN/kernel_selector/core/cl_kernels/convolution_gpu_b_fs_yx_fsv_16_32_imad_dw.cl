/*
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
*/

#include "include/imad.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

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
// ======================================================================================
// Supported operations:
// input/output format: any b_fs_yx_fsv<k> - where <k> >= SIMD,
//                      input and output formats must be the same
// weights format:      os_i_yxs_oxv<k>_yxsv4 - where <k> same as in input format
// input data types:   uchar8, char8
// weights data types: uchar8, char8
// output data types:  uchar8, char8, half, float
// asymetric quantization: weights zero points, compensation term
// ======================================================================================

#if OUTPUT_LAYOUT_B_FS_YX_FSV16
#   define FSV 16
#elif OUTPUT_LAYOUT_B_FS_YX_FSV32
#   define FSV 32
#else
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - unsupported output layout.
#endif

#define F_PER_WI ((FSV) / (SIMD))

#define DEQUANTIZED_TYPE float
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

#define AS_INPUT_TYPE(val)        CAT(as_, INPUT_TYPE)(val)
#define AS_INPUT_TYPE2(val)       CAT(as_, INPUT_TYPE2)(val)
#define AS_INPUT_TYPE4(val)       CAT(as_, INPUT_TYPE4)(val)
#define AS_INPUT_TYPE8(val)       CAT(as_, INPUT_TYPE8)(val)
#define AS_INPUT_TYPE16(val)      CAT(as_, INPUT_TYPE16)(val)

#define AS_FILTER_TYPE4(val)      CAT(as_, FILTER_TYPE4)(val)

#define TO_DEQUANTIZED_TYPE(val)  CAT(convert_, DEQUANTIZED_TYPE)(val)

#define GET_INPUT_INDEX(b, f, y, x)    INPUT0_GET_INDEX(b, f, y, x)
#if FSV == 16
#   define GET_WEIGHTS_INDEX(g, o, i, y, x)  GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(FILTER, g, 0, 0, y, x)
#else
#   define GET_WEIGHTS_INDEX(g, o, i, y, x)  GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(FILTER, g, 0, 0, y, x)
#endif
#define GET_OUTPUT_INDEX(b, f, y, x)   OUTPUT_GET_INDEX(b, f, y, x)
#define GET_BIAS_INDEX(b, f, y, x)     BIAS_GET_INDEX(b, f, y, x)

#define INPUT_X_PITCH FSV
#define INPUT_Y_PITCH (FSV * (INPUT0_SIZE_X + INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X))

#define WEIGHTS_YXS_PITCH (4 * FSV)

#define FILTER_SPATIAL_SIZE (FILTER_SIZE_X * FILTER_SIZE_Y)

#if OUTPUT_TYPE_SIZE == 1
#   define OUTPUT_BLOCK_WRITE(ptr, val)    BLOCK_WRITE_UC_1((__global uchar*)(ptr), as_uchar(val));
#   define OUTPUT_BLOCK_WRITE2(ptr, val)   BLOCK_WRITE_UC_2((__global uchar*)(ptr), as_uchar2(val));
#   define OUTPUT_BLOCK_WRITE4(ptr, val)   BLOCK_WRITE_UC_4((__global uchar*)(ptr), as_uchar4(val));
#   define OUTPUT_BLOCK_WRITE8(ptr, val)   BLOCK_WRITE_UC_8((__global uchar*)(ptr), as_uchar8(val));
#   define OUTPUT_BLOCK_WRITE16(ptr, val)  BLOCK_WRITE_UC_16((__global uchar*)(ptr), as_uchar16(val));
#elif OUTPUT_TYPE_SIZE == 2
#   define OUTPUT_BLOCK_WRITE(ptr, val)    intel_sub_group_block_write_us((__global ushort*)(ptr), as_ushort(val));
#   define OUTPUT_BLOCK_WRITE2(ptr, val)   intel_sub_group_block_write_us2((__global ushort*)(ptr), as_ushort2(val));
#   define OUTPUT_BLOCK_WRITE4(ptr, val)   intel_sub_group_block_write_us4((__global ushort*)(ptr), as_ushort4(val));
#   define OUTPUT_BLOCK_WRITE8(ptr, val)   intel_sub_group_block_write_us8((__global ushort*)(ptr), as_ushort8(val));
#   define OUTPUT_BLOCK_WRITE16(ptr, val)                                               \
    OUTPUT_BLOCK_WRITE8(ptr, (val).lo)                                                  \
    OUTPUT_BLOCK_WRITE8((__global ushort*)(ptr) + 8 * get_max_sub_group_size(), (val).hi)
#elif OUTPUT_TYPE_SIZE == 4
#   define OUTPUT_BLOCK_WRITE(ptr, val)    intel_sub_group_block_write((__global uint*)(ptr), as_uint(val));
#   define OUTPUT_BLOCK_WRITE2(ptr, val)   intel_sub_group_block_write2((__global uint*)(ptr), as_uint2(val));
#   define OUTPUT_BLOCK_WRITE4(ptr, val)   intel_sub_group_block_write4((__global uint*)(ptr), as_uint4(val));
#   define OUTPUT_BLOCK_WRITE8(ptr, val)   intel_sub_group_block_write8((__global uint*)(ptr), as_uint8(val));
#   define OUTPUT_BLOCK_WRITE16(ptr, val)                                               \
    OUTPUT_BLOCK_WRITE8(ptr, (val).lo)                                                  \
    OUTPUT_BLOCK_WRITE8((__global uint*)(ptr) + 8 * get_max_sub_group_size(), (val).hi)
#else
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - unsupported output type.
#endif

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

#if FILTER_BLOCKED % 4 != 0
#   error convolution_gpu_b_fs_yx_fsv_16_32_imad_dw.cl - FILTER_BLOCKED must be multiple of 4.
#endif

#ifndef OUTPUT_PAD_VALUE
#   define OUTPUT_PAD_VALUE (OUTPUT_TYPE)(0)
#   define OUTPUT_PAD_VALUE_undef
#endif

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(LWS0, LWS1, SIMD)))
KERNEL(convolution)(
    const __global  INPUT0_TYPE  *input,
    __global        OUTPUT_TYPE  *output,
    const __global  FILTER_TYPE  *weights,
#if BIAS_TERM
    const __global BIAS_TYPE     *biases,
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp,
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp,
#endif
#if COMPENSATION_TERM
    const __global COMPENSATION_TYPE *compensation,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx
) {
    uint x = get_global_id(0) * TILE_X;
    uint y = get_global_id(1);
    uint bf = get_group_id(2);
    uint b = bf % OUTPUT_BATCH_NUM;
    uint f = bf / OUTPUT_BATCH_NUM * FSV;

    uint input_offset = GET_INPUT_INDEX(b, f, (int)y * STRIDE_SIZE_Y - PADDING_SIZE_Y, (int)x * STRIDE_SIZE_X - PADDING_SIZE_X);
    uint weights_offset = GET_WEIGHTS_INDEX(f, 0, 0, 0, 0);

    int acc[TILE_X * F_PER_WI] = { };
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    int src_sum[TILE_X * F_PER_WI] = { };
#endif

    __attribute__((opencl_unroll_hint))
    for (uint fi = 0; fi < FILTER_BLOCKED / 4 * 4; fi += 4) {
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
        uint4 input_x_offset = fx * (DILATION_SIZE_X * INPUT_X_PITCH);
        uint4 input_y_offset = fy * (DILATION_SIZE_Y * INPUT_Y_PITCH);
        uint4 input_spatial_offset = input_x_offset + input_y_offset;
        uint4 input_idx = input_spatial_offset + input_offset;

        uint tx = 0;
        __attribute__((opencl_unroll_hint))
        for (; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            INPUT_TYPE16 tmp_in0 = AS_INPUT_TYPE16(BLOCK_READ_UC_16((const __global uchar*)(input + input_idx.s0)));
            INPUT_TYPE16 tmp_in1 = AS_INPUT_TYPE16(BLOCK_READ_UC_16((const __global uchar*)(input + input_idx.s1)));
            INPUT_TYPE16 tmp_in2 = AS_INPUT_TYPE16(BLOCK_READ_UC_16((const __global uchar*)(input + input_idx.s2)));
            INPUT_TYPE16 tmp_in3 = AS_INPUT_TYPE16(BLOCK_READ_UC_16((const __global uchar*)(input + input_idx.s3)));

            VEC_TO_ARRAY_16(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_16(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_16(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_16(in_trans3, tmp_in3, tx);

            input_idx += 16 * SIMD;
        }
        if (TILE_X * F_PER_WI % 16 >= 8) {
            INPUT_TYPE8 tmp_in0 = AS_INPUT_TYPE8(BLOCK_READ_UC_8((const __global uchar*)(input + input_idx.s0)));
            INPUT_TYPE8 tmp_in1 = AS_INPUT_TYPE8(BLOCK_READ_UC_8((const __global uchar*)(input + input_idx.s1)));
            INPUT_TYPE8 tmp_in2 = AS_INPUT_TYPE8(BLOCK_READ_UC_8((const __global uchar*)(input + input_idx.s2)));
            INPUT_TYPE8 tmp_in3 = AS_INPUT_TYPE8(BLOCK_READ_UC_8((const __global uchar*)(input + input_idx.s3)));

            VEC_TO_ARRAY_8(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_8(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_8(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_8(in_trans3, tmp_in3, tx);

            input_idx += 8 * SIMD;
            tx += 8;
        }
        if (TILE_X * F_PER_WI % 8 >= 4) {
            INPUT_TYPE4 tmp_in0 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s0)));
            INPUT_TYPE4 tmp_in1 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s1)));
            INPUT_TYPE4 tmp_in2 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s2)));
            INPUT_TYPE4 tmp_in3 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s3)));

            VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_4(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_4(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_4(in_trans3, tmp_in3, tx);

            input_idx += 4 * SIMD;
            tx += 4;
        }
        if (TILE_X * F_PER_WI % 4 >= 2) {
            INPUT_TYPE2 tmp_in0 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s0)));
            INPUT_TYPE2 tmp_in1 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s1)));
            INPUT_TYPE2 tmp_in2 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s2)));
            INPUT_TYPE2 tmp_in3 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s3)));

            VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx);
            VEC_TO_ARRAY_2(in_trans1, tmp_in1, tx);
            VEC_TO_ARRAY_2(in_trans2, tmp_in2, tx);
            VEC_TO_ARRAY_2(in_trans3, tmp_in3, tx);

            input_idx += 2 * SIMD;
            tx += 2;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            in_trans0[tx] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s0)));
            in_trans1[tx] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s1)));
            in_trans2[tx] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s2)));
            in_trans3[tx] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s3)));
        }
#else
        uint4 input_x_offset = fx * DILATION_SIZE_X * INPUT_X_PITCH;
        uint4 input_y_offset = fy * DILATION_SIZE_Y * INPUT_Y_PITCH;
        uint4 input_spatial_offset = input_x_offset + input_y_offset;
        uint4 input_start_offset = input_spatial_offset + input_offset;
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            uint4 input_idx = input_start_offset + tx * STRIDE_SIZE_X * INPUT_X_PITCH;
            // Block reads along feature slice
            uint fw = 0;
            __attribute__((opencl_unroll_hint))
            for (; fw + 4 <= F_PER_WI; fw += 4) {
                INPUT_TYPE4 tmp_in0 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s0)));
                INPUT_TYPE4 tmp_in1 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s1)));
                INPUT_TYPE4 tmp_in2 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s2)));
                INPUT_TYPE4 tmp_in3 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx.s3)));

                VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans1, tmp_in1, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans2, tmp_in2, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_4(in_trans3, tmp_in3, tx * F_PER_WI + fw);

                input_idx += 4 * SIMD;
            }
            if (F_PER_WI % 4 >= 2) {
                INPUT_TYPE2 tmp_in0 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s0)));
                INPUT_TYPE2 tmp_in1 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s1)));
                INPUT_TYPE2 tmp_in2 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s2)));
                INPUT_TYPE2 tmp_in3 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx.s3)));

                VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans1, tmp_in1, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans2, tmp_in2, tx * F_PER_WI + fw);
                VEC_TO_ARRAY_2(in_trans3, tmp_in3, tx * F_PER_WI + fw);

                input_idx += 2 * SIMD;
                fw += 2;
            }
            if (F_PER_WI % 2 == 1) {
                in_trans0[tx * F_PER_WI + fw] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s0)));
                in_trans1[tx * F_PER_WI + fw] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s1)));
                in_trans2[tx * F_PER_WI + fw] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s2)));
                in_trans3[tx * F_PER_WI + fw] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx.s3)));
            }
        }
#endif
        // Weights loading:
        FILTER_TYPE4 wei[F_PER_WI];
        __attribute__((opencl_unroll_hint))
        for (uint fw = 0; fw < F_PER_WI; ++fw) {
            wei[fw] = AS_FILTER_TYPE4(intel_sub_group_block_read((const __global uint*)(weights + weights_offset) + fw * SIMD));
        }

        // Transpose input:
        INPUT_TYPE4 in[TILE_X * F_PER_WI];
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
                uint in_offset = tx * F_PER_WI + fw;
                in[in_offset] = (INPUT_TYPE4)(in_trans0[in_offset], in_trans1[in_offset], in_trans2[in_offset], in_trans3[in_offset]);
            }
        }

        // IMAD:
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
                acc[tx * F_PER_WI + fw] = IMAD(acc[tx * F_PER_WI + fw], in[tx * F_PER_WI + fw], wei[fw]);
            }
        }

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        // Accumulate for input values for asymmetric weights:
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
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
    __attribute__((opencl_unroll_hint))
    for (uint fw = 0; fw < F_PER_WI; ++fw) {
        wei[fw] = AS_FILTER_TYPE4(intel_sub_group_block_read((const __global uint*)(weights + weights_offset) + fw * SIMD));
    }

    __attribute__((opencl_unroll_hint))
    for (uint fi = 0; fi < FILTER_SPATIAL_SIZE - FILTER_BLOCKED; ++fi) {
        // Input loading:
        uint fx = (fi + FILTER_BLOCKED) % FILTER_SIZE_X;
        uint fy = (fi + FILTER_BLOCKED) / FILTER_SIZE_X;

        INPUT_TYPE in_trans0[TILE_X * F_PER_WI];
#   if STRIDE_SIZE_X == 1
        uint input_x_offset = fx * (DILATION_SIZE_X * INPUT_X_PITCH);
        uint input_y_offset = fy * (DILATION_SIZE_Y * INPUT_Y_PITCH);
        uint input_spatial_offset = input_x_offset + input_y_offset;
        uint input_idx = input_spatial_offset + input_offset;

        uint tx = 0;
        __attribute__((opencl_unroll_hint))
        for (; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            INPUT_TYPE16 tmp_in0 = AS_INPUT_TYPE16(BLOCK_READ_UC_16((const __global uchar*)(input + input_idx)));
            VEC_TO_ARRAY_16(in_trans0, tmp_in0, tx);
            input_idx += 16 * SIMD;
        }
        if (TILE_X * F_PER_WI % 16 >= 8) {
            INPUT_TYPE8 tmp_in0 = AS_INPUT_TYPE8(BLOCK_READ_UC_8((const __global uchar*)(input + input_idx)));
            VEC_TO_ARRAY_8(in_trans0, tmp_in0, tx);
            input_idx += 8 * SIMD;
            tx += 8;
        }
        if (TILE_X * F_PER_WI % 8 >= 4) {
            INPUT_TYPE4 tmp_in0 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx)));
            VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx);
            input_idx += 4 * SIMD;
            tx += 4;
        }
        if (TILE_X * F_PER_WI % 4 >= 2) {
            INPUT_TYPE2 tmp_in0 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx)));
            VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx);
            input_idx += 2 * SIMD;
            tx += 2;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            in_trans0[tx] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx)));
        }
#   else
        uint input_x_offset = fx * DILATION_SIZE_X * INPUT_X_PITCH;
        uint input_y_offset = fy * DILATION_SIZE_Y * INPUT_Y_PITCH;
        uint input_spatial_offset = input_x_offset + input_y_offset;
        uint input_start_offset = input_spatial_offset + input_offset;
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            uint input_idx = input_start_offset + tx * STRIDE_SIZE_X * INPUT_X_PITCH;
            uint fw = 0;
            __attribute__((opencl_unroll_hint))
            for (; fw + 4 <= F_PER_WI; fw += 4) {
                INPUT_TYPE4 tmp_in0 = AS_INPUT_TYPE4(BLOCK_READ_UC_4((const __global uchar*)(input + input_idx)));
                VEC_TO_ARRAY_4(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                input_idx += 4 * SIMD;
            }
            if (F_PER_WI % 4 >= 2) {
                INPUT_TYPE2 tmp_in0 = AS_INPUT_TYPE2(BLOCK_READ_UC_2((const __global uchar*)(input + input_idx)));
                VEC_TO_ARRAY_2(in_trans0, tmp_in0, tx * F_PER_WI + fw);
                input_idx += 2 * SIMD;
                fw += 2;
            }
            if (F_PER_WI % 2 == 1) {
                in_trans0[tx * F_PER_WI + fw] = AS_INPUT_TYPE(BLOCK_READ_UC_1((const __global uchar*)(input + input_idx)));
            }
        }
#   endif
        // Raw multiply accumulate:
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
                acc[tx * F_PER_WI + fw] += (int)in_trans0[tx * F_PER_WI + fw] * (int)wei[fw][fi];
            }
        }

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        // Accumulate input values for asymmetric weights:
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
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
    __attribute__((opencl_unroll_hint))
    for (uint fw = 0; fw < F_PER_WI; ++fw) {
        uint bias_offset = f + fw * SIMD + get_sub_group_local_id();
        BIAS_TYPE bias = biases[bias_offset];
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            dequantized[tx * F_PER_WI + fw] += TO_DEQUANTIZED_TYPE(bias);
        }
    }
#   elif BIAS_PER_OUTPUT
    __attribute__((opencl_unroll_hint))
    for (uint tx = 0; tx < TILE_X; ++tx) {
        __attribute__((opencl_unroll_hint))
        for (uint fw = 0; fw < F_PER_WI; ++fw) {
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
        __attribute__((opencl_unroll_hint))
        for (uint fw = 0; fw < F_PER_WI; ++fw) {
            WEIGHTS_ZERO_POINTS_TYPE wzp = weights_zp[f + fw * SIMD + get_sub_group_local_id()];
            __attribute__((opencl_unroll_hint))
            for (uint tx = 0; tx < TILE_X; ++tx) {
                dequantized[tx * F_PER_WI + fw] -= TO_DEQUANTIZED_TYPE(src_sum[tx * F_PER_WI + fw]) * TO_DEQUANTIZED_TYPE(wzp);
            }
        }
    }
#endif

#if COMPENSATION_TERM
    {
        __attribute__((opencl_unroll_hint))
        for (uint fw = 0; fw < F_PER_WI; ++fw) {
            COMPENSATION_TYPE comp = compensation[f + fw * SIMD + get_sub_group_local_id()];
            __attribute__((opencl_unroll_hint))
            for (uint tx = 0; tx < TILE_X; ++tx) {
                dequantized[tx * F_PER_WI + fw] += TO_DEQUANTIZED_TYPE(comp);
            }
        }
    }
#endif

    OUTPUT_TYPE out[TILE_X * F_PER_WI];
    // Fused ops and conversion to output type
    __attribute__((opencl_unroll_hint))
    for (uint tx = 0; tx < TILE_X; ++tx) {
#if HAS_FUSED_OPS
        uint fused_ops_x = x + tx;
        uint fused_ops_f = f;
        uint fw = 0;
        __attribute__((opencl_unroll_hint))
        for (; fw + 4 <= F_PER_WI; fw += 4) {
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
        __attribute__((opencl_unroll_hint))
        for (uint fw = 0; fw < F_PER_WI; ++fw) {
            out[tx * F_PER_WI + fw] = TO_OUTPUT_TYPE(dequantized[tx * F_PER_WI + fw]);
        }
#endif
    }

    // Fill results outside output in features with OUTPUT_PAD_VALUE.
    if (OUTPUT_FEATURE_NUM % FSV != 0 && f + FSV > OUTPUT_FEATURE_NUM) {
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            __attribute__((opencl_unroll_hint))
            for (uint fw = 0; fw < F_PER_WI; ++fw) {
                bool outside = fw * SIMD + get_sub_group_local_id() >= OUTPUT_FEATURE_NUM % FSV;
                out[tx * F_PER_WI + fw] = outside ? OUTPUT_PAD_VALUE : out[tx * F_PER_WI + fw];
            }
        }
    }

    uint output_offset = GET_OUTPUT_INDEX(b, f, y, x);

    if (OUTPUT_SIZE_X % TILE_X == 0 || x + TILE_X <= OUTPUT_SIZE_X) {
        // Full output tile x write using block write ladder
        uint tx = 0;
        __attribute__((opencl_unroll_hint))
        for (; tx + 16 <= TILE_X * F_PER_WI; tx += 16) {
            OUTPUT_TYPE16 tmp_write;
            ARRAY_TO_VEC_16(tmp_write, out, tx);
            OUTPUT_BLOCK_WRITE16(output + output_offset, tmp_write);
            output_offset += 16 * SIMD;
        }
        if (TILE_X * F_PER_WI % 16 >= 8) {
            OUTPUT_TYPE8 tmp_write;
            ARRAY_TO_VEC_8(tmp_write, out, tx);
            OUTPUT_BLOCK_WRITE8(output + output_offset, tmp_write);
            tx += 8;
            output_offset += 8 * SIMD;
        }
        if (TILE_X * F_PER_WI % 8 >= 4) {
            OUTPUT_TYPE4 tmp_write;
            ARRAY_TO_VEC_4(tmp_write, out, tx);
            OUTPUT_BLOCK_WRITE4(output + output_offset, tmp_write);
            tx += 4;
            output_offset += 4 * SIMD;
        }
        if (TILE_X * F_PER_WI % 4 >= 2) {
            OUTPUT_TYPE2 tmp_write;
            ARRAY_TO_VEC_2(tmp_write, out, tx);
            OUTPUT_BLOCK_WRITE2(output + output_offset, tmp_write);
            tx += 2;
            output_offset += 2 * SIMD;
        }
        if (TILE_X * F_PER_WI % 2 == 1) {
            OUTPUT_BLOCK_WRITE(output + output_offset, out[tx]);
        }
    } else {
        // Leftovers write, block writes in f dimension only
        __attribute__((opencl_unroll_hint))
        for (uint tx = 0; tx < TILE_X; ++tx) {
            if (tx < OUTPUT_SIZE_X % TILE_X) {
                uint fw = 0;
                __attribute__((opencl_unroll_hint))
                for (; fw + 4 <= F_PER_WI; fw += 4) {
                    OUTPUT_TYPE4 tmp_write;
                    ARRAY_TO_VEC_4(tmp_write, out, tx * F_PER_WI + fw);
                    OUTPUT_BLOCK_WRITE4(output + output_offset + fw * SIMD, tmp_write);
                }
                if (F_PER_WI % 4 >= 2) {
                    OUTPUT_TYPE2 tmp_write;
                    ARRAY_TO_VEC_2(tmp_write, out, tx * F_PER_WI + fw);
                    OUTPUT_BLOCK_WRITE2(output + output_offset + fw * SIMD, tmp_write);
                    fw += 2;
                }
                if (F_PER_WI % 2 == 1) {
                    OUTPUT_BLOCK_WRITE(output + output_offset + fw * SIMD, out[tx * F_PER_WI + fw]);
                }
            }
            output_offset += FSV;
        }
    }
}

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

#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE8
#undef AS_INPUT_TYPE16

#undef AS_FILTER_TYPE

#undef TO_DEQUANTIZED_TYPE

#undef GET_INPUT_INDEX
#undef GET_WEIGHTS_INDEX
#undef GET_OUTPUT_INDEX

#undef INPUT_X_PITCH
#undef INPUT_Y_PITCH

#undef WEIGHTS_YXS_PITCH 

#undef FILTER_SPATIAL_SIZE

#undef OUTPUT_BLOCK_WRITE
#undef OUTPUT_BLOCK_WRITE2
#undef OUTPUT_BLOCK_WRITE4
#undef OUTPUT_BLOCK_WRITE8
#undef OUTPUT_BLOCK_WRITE16

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
