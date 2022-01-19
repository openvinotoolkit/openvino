// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define OC_BLOCK_SIZE 32

#define ALIGNED_BLOCK_READ(ptr, byte_offset) as_uint(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ2(ptr, byte_offset) as_uint2(intel_sub_group_block_read2((const __global uint*)(ptr) + (byte_offset)))

#if BINARY_PACKED_OUTPUT
    #define BUFFER_TYPE UNIT_TYPE
#else
    #define BUFFER_TYPE OUTPUT_TYPE
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(binary_convolution_generic)(const __global INPUT0_TYPE* input,
                                         __global OUTPUT_TYPE* output,
                                   const __global FILTER_TYPE* weights,
#if HAS_FUSED_OPS_DECLS
                                   FUSED_OPS_DECLS,
#endif
                                   uint split_idx)
{
    const int f_block = get_global_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_group_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint input_offset = INPUT0_OFFSET
                            + b*INPUT0_FEATURE_NUM_PACKED*INPUT0_FEATURE_PITCH
                            + input_y*INPUT0_Y_PITCH
                            + input_x*INPUT0_X_PITCH;

    typedef MAKE_VECTOR_TYPE(FILTER_TYPE, 2) data_t;

#if BINARY_PACKED_OUTPUT
    const uint dst_index = OUTPUT_OFFSET
                         + b*OUTPUT_FEATURE_NUM_PACKED*OUTPUT_FEATURE_PITCH
                         + f_block*OUTPUT_FEATURE_PITCH
                         + y*OUTPUT_Y_PITCH
                         + x;
#else
    const uint dst_index = OUTPUT_OFFSET
                         + b*OUTPUT_BATCH_PITCH
                         + f_block*OC_BLOCK_SIZE*OUTPUT_FEATURE_PITCH
                         + y*OUTPUT_Y_PITCH
                         + x;
#endif
    const uint filter_offset = f_block*OC_BLOCK_SIZE*INPUT0_FEATURE_NUM_PACKED*FILTER_SIZE_Y*FILTER_SIZE_X;

    int dst_buf[SUB_GROUP_SIZE*2] = { 0 }; // 2 OC x 16 X

#if EXCLUDE_PAD
    int real_ks = 0;
    // calc real kernel size for out_x = x+lid
    for (int kh = 0; kh < FILTER_SIZE_Y; kh++)
    {
        for (int kw = 0; kw < FILTER_SIZE_X; kw++)
        {
            real_ks += ((input_x + kw + lid*STRIDE_SIZE_X >= 0) &&
                        (input_x + kw + lid*STRIDE_SIZE_X < INPUT0_SIZE_X) &&
                        (input_y + kh >= 0) &&
                        (input_y + kh < INPUT0_SIZE_Y)) ? 1 : 0;
        }
    }
#endif

    for (int k = 0; k < INPUT0_FEATURE_NUM_PACKED; ++k)
    {
        for (int kh = 0; kh < FILTER_SIZE_Y; kh++)
        {
            INPUT0_TYPE line_cache[INPUT_ELEMENTS_PER_WI];
            for (int i = 0; i < INPUT_ELEMENTS_PER_WI; i++)
            {
                line_cache[i] = PAD_VALUE;
            }

            if (input_y + kh >= 0 && input_y + kh < INPUT0_SIZE_Y)
            {
                for (int i = 0; i < INPUT_ELEMENTS_PER_WI; i++)
                {
                     if (input_x + i*SUB_GROUP_SIZE >= 0 && input_x + (i+1)*SUB_GROUP_SIZE < INPUT0_SIZE_X)
                         line_cache[i] = ALIGNED_BLOCK_READ(input, input_offset + kh*INPUT0_Y_PITCH + k*INPUT0_FEATURE_PITCH + i*SUB_GROUP_SIZE);
                     else if (input_x + i*SUB_GROUP_SIZE + lid >= 0 && input_x + i*SUB_GROUP_SIZE + lid < INPUT0_SIZE_X)
                         line_cache[i] = input[input_offset + kh*INPUT0_Y_PITCH + k*INPUT0_FEATURE_PITCH + i*SUB_GROUP_SIZE + lid];
                }
            }

            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (int kw = 0; kw < FILTER_SIZE_X; kw++)
            {
                // Load 32 OC x 32 ICP. Each WI has lid-th and (lid+16)-th channels
                data_t wei = ALIGNED_BLOCK_READ2(weights, filter_offset + OC_BLOCK_SIZE*(k*FILTER_SIZE_Y*FILTER_SIZE_X + kh*FILTER_SIZE_X + kw));

                // Single WI in subgroup calcs 2 OC x 16 X elements
                __attribute__((opencl_unroll_hint(SUB_GROUP_SIZE)))
                for (int i = 0; i < SUB_GROUP_SIZE; i++)
                {
                    INPUT0_TYPE src = intel_sub_group_shuffle(line_cache[(kw + i*STRIDE_SIZE_X) / SUB_GROUP_SIZE],
                                                                         (kw + i*STRIDE_SIZE_X) % SUB_GROUP_SIZE);
#if EXCLUDE_PAD
                    int compute = ((input_x + kw + i*STRIDE_SIZE_X >= 0) &&
                                   (input_x + kw + i*STRIDE_SIZE_X < INPUT0_SIZE_X) &&
                                   (input_y + kh >= 0) &&
                                   (input_y + kh < INPUT0_SIZE_Y)) ? 1 : 0;

                    if (!compute)
                        continue;
#endif

#if LEFTOVERS_IC
                    if (k == INPUT0_FEATURE_NUM_PACKED - 1)
                    {
                        dst_buf[0*SUB_GROUP_SIZE + i] += popcount((wei.s0 ^ src) & FILTER_MASK);
                        dst_buf[1*SUB_GROUP_SIZE + i] += popcount((wei.s1 ^ src) & FILTER_MASK);
                        continue;
                    }
#endif

                    dst_buf[0*SUB_GROUP_SIZE + i] += popcount(wei.s0 ^ src);
                    dst_buf[1*SUB_GROUP_SIZE + i] += popcount(wei.s1 ^ src);
                }
            }
        }
    }

#if EXCLUDE_PAD

#endif
    // Load data for fused operations (scales, biases, quantization thresholds, etc)
#if CUSTOM_FUSED_OPS
    FUSED_OPS_PREPARE_DATA;
#endif

    BUFFER_TYPE dst[SUB_GROUP_SIZE*2];

    __attribute__((opencl_unroll_hint(SUB_GROUP_SIZE*2)))
    for (int i = 0; i < SUB_GROUP_SIZE*2; i++)
    {
#if EXCLUDE_PAD
        CONV_RESULT_TYPE res = TO_CONV_RESULT_TYPE(INPUT0_FEATURE_NUM*intel_sub_group_shuffle(real_ks, i%SUB_GROUP_SIZE) - 2*dst_buf[i]);
#else
        CONV_RESULT_TYPE res = TO_CONV_RESULT_TYPE(INPUT0_FEATURE_NUM*FILTER_SIZE_Y*FILTER_SIZE_X - 2*dst_buf[i]);
#endif

#if CUSTOM_FUSED_OPS
        DO_ELTWISE_FUSED_OPS;
        dst[i] = res;
#elif HAS_FUSED_OPS
        FUSED_OPS;
        dst[i] = FUSED_OPS_RESULT;
#else
        dst[i] = res;
#endif

    }

#if BINARY_PACKED_OUTPUT
    int packed_out[SUB_GROUP_SIZE];

#if CUSTOM_FUSED_OPS
    DO_CHANNEL_PACK_OPS;
#else
    #error "BINARY_PACKED_OUTPUT should be true only if node has fused quantize with bin output"
#endif

    bool in_x = (x + lid) < OUTPUT_SIZE_X;
    bool in_y = y < OUTPUT_SIZE_Y;
    if (in_x && in_y)
        output[dst_index + lid] = packed_out[lid];

#else

    for (int oc = 0; oc < 2; oc++)
    {
        for (int ow = 0; ow < SUB_GROUP_SIZE; ow++)
        {
            bool in_x = (x + ow) < OUTPUT_SIZE_X;
            bool in_y = y < OUTPUT_SIZE_Y;
            bool in_fm = f_block*OC_BLOCK_SIZE + oc*SUB_GROUP_SIZE + lid < OUTPUT_FEATURE_NUM;
            if (in_x && in_y && in_fm)
            {
                output[dst_index + (oc*SUB_GROUP_SIZE + lid)*OUTPUT_FEATURE_PITCH + ow] = TO_OUTPUT_TYPE(dst[oc*SUB_GROUP_SIZE + ow]);
            }
        }
    }

#endif
}
