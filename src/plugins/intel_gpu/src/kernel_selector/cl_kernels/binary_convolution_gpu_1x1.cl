// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define OC_BLOCK_SIZE 32

#define GET_WEI(data, id) intel_sub_group_shuffle(data, id)
#define ALIGNED_BLOCK_READ(ptr, byte_offset) as_uint(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write((__global uint*)(ptr) + (byte_offset), as_uint(val))
#define ALIGNED_BLOCK_READ2(ptr, byte_offset) as_uint2(intel_sub_group_block_read2((const __global uint*)(ptr) + (byte_offset)))

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(binary_convolution_1x1)(const __global INPUT0_TYPE* input,
                                     __global OUTPUT_TYPE* output,
                               const __global FILTER_TYPE* weights,
#if HAS_FUSED_OPS_DECLS
                               FUSED_OPS_DECLS,
#endif
                               uint split_idx)
{
    const int xy = get_group_id(0);
    const int f_block = get_global_id(1);
    const int b = get_global_id(2);
    const int lid = get_sub_group_local_id();
#if PADDED_INPUT
    const int x = (xy * XY_BLOCK_SIZE + lid) % OUTPUT_SIZE_X;
    const int y = (xy * XY_BLOCK_SIZE + lid) / OUTPUT_SIZE_X;
    const uint input_offset = INPUT0_OFFSET
                            + b*INPUT0_FEATURE_NUM_PACKED*INPUT0_FEATURE_PITCH
                            + y*INPUT0_Y_PITCH;
#else
    const int x = (xy * XY_BLOCK_SIZE + lid) % OUTPUT_SIZE_X;
    const int y = (xy * XY_BLOCK_SIZE + lid) / OUTPUT_SIZE_X;
    const uint input_offset = INPUT0_OFFSET
                            + b*INPUT0_FEATURE_NUM_PACKED*INPUT0_FEATURE_PITCH
                            + xy*XY_BLOCK_SIZE;
#endif
    typedef MAKE_VECTOR_TYPE(FILTER_TYPE, 2) wei_t;

#if BINARY_PACKED_OUTPUT
    const uint dst_index = OUTPUT_OFFSET
                         + b*OUTPUT_FEATURE_NUM_PACKED*OUTPUT_FEATURE_PITCH
                         + f_block*OUTPUT_FEATURE_PITCH;
#else
    const uint dst_index = OUTPUT_OFFSET
                         + b*OUTPUT_BATCH_PITCH
                         + f_block*OC_BLOCK_SIZE*OUTPUT_FEATURE_PITCH;
#endif
    const uint filter_offset = f_block*OC_BLOCK_SIZE*INPUT0_FEATURE_NUM_PACKED;

    int dst_buf[OC_BLOCK_SIZE] = { 0 }; // 32 OC

    for (int k = 0; k < INPUT0_FEATURE_NUM_PACKED; ++k)
    {
        // Load 16 input elements from feature map by subgroup
#if PADDED_INPUT
        INPUT0_TYPE src = input[input_offset + k*INPUT0_FEATURE_PITCH + x];
#else
        INPUT0_TYPE src = ALIGNED_BLOCK_READ(input, input_offset + k*INPUT0_FEATURE_PITCH);
#endif

        // Load 32 OC x 32 ICP. Each WI has lid-th and (lid+16)-th channels
        wei_t wei = ALIGNED_BLOCK_READ2(weights, filter_offset + k * OC_BLOCK_SIZE);

        // Shuffle 32 OC x 32 ICP of weights in each WI
        const wei_t wei0  = GET_WEI(wei, 0);
        const wei_t wei1  = GET_WEI(wei, 1);
        const wei_t wei2  = GET_WEI(wei, 2);
        const wei_t wei3  = GET_WEI(wei, 3);
        const wei_t wei4  = GET_WEI(wei, 4);
        const wei_t wei5  = GET_WEI(wei, 5);
        const wei_t wei6  = GET_WEI(wei, 6);
        const wei_t wei7  = GET_WEI(wei, 7);
        const wei_t wei8  = GET_WEI(wei, 8);
        const wei_t wei9  = GET_WEI(wei, 9);
        const wei_t wei10 = GET_WEI(wei, 10);
        const wei_t wei11 = GET_WEI(wei, 11);
        const wei_t wei12 = GET_WEI(wei, 12);
        const wei_t wei13 = GET_WEI(wei, 13);
        const wei_t wei14 = GET_WEI(wei, 14);
        const wei_t wei15 = GET_WEI(wei, 15);

#if LEFTOVERS_IC
        if (k == INPUT0_FEATURE_NUM_PACKED - 1)
        {
            dst_buf[0]  += popcount((wei0.s0 ^ src) & FILTER_MASK);
            dst_buf[1]  += popcount((wei1.s0 ^ src) & FILTER_MASK);
            dst_buf[2]  += popcount((wei2.s0 ^ src) & FILTER_MASK);
            dst_buf[3]  += popcount((wei3.s0 ^ src) & FILTER_MASK);
            dst_buf[4]  += popcount((wei4.s0 ^ src) & FILTER_MASK);
            dst_buf[5]  += popcount((wei5.s0 ^ src) & FILTER_MASK);
            dst_buf[6]  += popcount((wei6.s0 ^ src) & FILTER_MASK);
            dst_buf[7]  += popcount((wei7.s0 ^ src) & FILTER_MASK);
            dst_buf[8]  += popcount((wei8.s0 ^ src) & FILTER_MASK);
            dst_buf[9]  += popcount((wei9.s0 ^ src) & FILTER_MASK);
            dst_buf[10] += popcount((wei10.s0 ^ src) & FILTER_MASK);
            dst_buf[11] += popcount((wei11.s0 ^ src) & FILTER_MASK);
            dst_buf[12] += popcount((wei12.s0 ^ src) & FILTER_MASK);
            dst_buf[13] += popcount((wei13.s0 ^ src) & FILTER_MASK);
            dst_buf[14] += popcount((wei14.s0 ^ src) & FILTER_MASK);
            dst_buf[15] += popcount((wei15.s0 ^ src) & FILTER_MASK);

#if OUTPUT_FEATURE_NUM > 16
            dst_buf[16] += popcount((wei0.s1 ^ src) & FILTER_MASK);
            dst_buf[17] += popcount((wei1.s1 ^ src) & FILTER_MASK);
            dst_buf[18] += popcount((wei2.s1 ^ src) & FILTER_MASK);
            dst_buf[19] += popcount((wei3.s1 ^ src) & FILTER_MASK);
            dst_buf[20] += popcount((wei4.s1 ^ src) & FILTER_MASK);
            dst_buf[21] += popcount((wei5.s1 ^ src) & FILTER_MASK);
            dst_buf[22] += popcount((wei6.s1 ^ src) & FILTER_MASK);
            dst_buf[23] += popcount((wei7.s1 ^ src) & FILTER_MASK);
            dst_buf[24] += popcount((wei8.s1 ^ src) & FILTER_MASK);
            dst_buf[25] += popcount((wei9.s1 ^ src) & FILTER_MASK);
            dst_buf[26] += popcount((wei10.s1 ^ src) & FILTER_MASK);
            dst_buf[27] += popcount((wei11.s1 ^ src) & FILTER_MASK);
            dst_buf[28] += popcount((wei12.s1 ^ src) & FILTER_MASK);
            dst_buf[29] += popcount((wei13.s1 ^ src) & FILTER_MASK);
            dst_buf[30] += popcount((wei14.s1 ^ src) & FILTER_MASK);
            dst_buf[31] += popcount((wei15.s1 ^ src) & FILTER_MASK);
#endif
            break;
        }
#endif
        dst_buf[0]  += popcount(wei0.s0 ^ src);
        dst_buf[1]  += popcount(wei1.s0 ^ src);
        dst_buf[2]  += popcount(wei2.s0 ^ src);
        dst_buf[3]  += popcount(wei3.s0 ^ src);
        dst_buf[4]  += popcount(wei4.s0 ^ src);
        dst_buf[5]  += popcount(wei5.s0 ^ src);
        dst_buf[6]  += popcount(wei6.s0 ^ src);
        dst_buf[7]  += popcount(wei7.s0 ^ src);
        dst_buf[8]  += popcount(wei8.s0 ^ src);
        dst_buf[9]  += popcount(wei9.s0 ^ src);
        dst_buf[10] += popcount(wei10.s0 ^ src);
        dst_buf[11] += popcount(wei11.s0 ^ src);
        dst_buf[12] += popcount(wei12.s0 ^ src);
        dst_buf[13] += popcount(wei13.s0 ^ src);
        dst_buf[14] += popcount(wei14.s0 ^ src);
        dst_buf[15] += popcount(wei15.s0 ^ src);

#if OUTPUT_FEATURE_NUM > 16
        dst_buf[16] += popcount(wei0.s1 ^ src);
        dst_buf[17] += popcount(wei1.s1 ^ src);
        dst_buf[18] += popcount(wei2.s1 ^ src);
        dst_buf[19] += popcount(wei3.s1 ^ src);
        dst_buf[20] += popcount(wei4.s1 ^ src);
        dst_buf[21] += popcount(wei5.s1 ^ src);
        dst_buf[22] += popcount(wei6.s1 ^ src);
        dst_buf[23] += popcount(wei7.s1 ^ src);
        dst_buf[24] += popcount(wei8.s1 ^ src);
        dst_buf[25] += popcount(wei9.s1 ^ src);
        dst_buf[26] += popcount(wei10.s1 ^ src);
        dst_buf[27] += popcount(wei11.s1 ^ src);
        dst_buf[28] += popcount(wei12.s1 ^ src);
        dst_buf[29] += popcount(wei13.s1 ^ src);
        dst_buf[30] += popcount(wei14.s1 ^ src);
        dst_buf[31] += popcount(wei15.s1 ^ src);
#endif
    }

    // Load data for fused operations (scales, biases, quantization thresholds, etc)
#if CUSTOM_FUSED_OPS
    FUSED_OPS_PREPARE_DATA;
#endif

    UNIT_TYPE dst[OC_BLOCK_SIZE];
    for (int oc = 0; oc < OC_BLOCK_SIZE; oc++)
    {
        CONV_RESULT_TYPE res = TO_CONV_RESULT_TYPE(INPUT0_FEATURE_NUM - 2*dst_buf[oc]);
#if CUSTOM_FUSED_OPS
        DO_ELTWISE_FUSED_OPS;
// Don't save floating-point intermediate result, since packed one is already computed
#if !BINARY_PACKED_OUTPUT
        dst[oc] = res;
#endif
#elif HAS_FUSED_OPS
        FUSED_OPS;
        dst[oc] = FUSED_OPS_RESULT;
#endif

    }

    bool in_x = x < OUTPUT_SIZE_X;
    bool in_y = y < OUTPUT_SIZE_Y;
#if BINARY_PACKED_OUTPUT

#if PADDED_OUTPUT
    if (in_x && in_y)
        output[dst_index + y*OUTPUT_Y_PITCH + x] = TO_OUTPUT_TYPE(packed_res);
#else
    if (xy * XY_BLOCK_SIZE < OUTPUT_SIZE_X*OUTPUT_SIZE_Y)
        ALIGNED_BLOCK_WRITE(output, dst_index + xy*XY_BLOCK_SIZE, TO_OUTPUT_TYPE(packed_res));
    else if (in_x && in_y)
        output[dst_index + y*OUTPUT_Y_PITCH + x] = TO_OUTPUT_TYPE(packed_res);

#endif

#else

    for (int oc = 0; oc < OC_BLOCK_SIZE; oc++)
    {
        bool in_fm = f_block*OC_BLOCK_SIZE + oc < OUTPUT_FEATURE_NUM;
        if (in_x && in_y && in_fm)
            output[dst_index + oc*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x] = TO_OUTPUT_TYPE(dst[oc]);
    }

#endif

}
