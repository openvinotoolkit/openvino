// Copyright (c) 2019 Intel Corporation
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


#include "include/include_all.cl"
#include "include/unit_type.cl"

#define OC_BLOCK_SIZE 16

#define GET_SRC(data, id) intel_sub_group_shuffle(data, id)
#define ALIGNED_BLOCK_READ(ptr, byte_offset) as_uint(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ2(ptr, byte_offset) as_uint2(intel_sub_group_block_read2((const __global uint*)(ptr) + (byte_offset)))

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(binary_convolution_1x1_b_fs_yx_fsv16)(const __global INPUT0_TYPE* input,
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
    const uint output_x_pitch = OC_BLOCK_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + OC_BLOCK_SIZE - 1) / OC_BLOCK_SIZE);
    const uint dst_index = OUTPUT_OFFSET*OC_BLOCK_SIZE
                         + b*output_b_pitch
                         + f_block*output_fs_pitch;

    const uint filter_offset = ((f_block/2)*2)*OC_BLOCK_SIZE*INPUT0_FEATURE_NUM_PACKED + (f_block%2)*16;

    int dst_buf[OC_BLOCK_SIZE] = { 0 }; // 16 X

    for (int k = 0; k < INPUT0_FEATURE_NUM_PACKED; ++k)
    {
        // Load 16 input elements from feature map by subgroup
#if PADDED_INPUT
        INPUT0_TYPE src = input[input_offset + k*INPUT0_FEATURE_PITCH + x];
#else
        INPUT0_TYPE src = ALIGNED_BLOCK_READ(input, input_offset + k*INPUT0_FEATURE_PITCH);
#endif

        // Load 32 OC x 32 ICP. Each WI has lid-th and (lid+16)-th channels
        FILTER_TYPE wei = ALIGNED_BLOCK_READ(weights, filter_offset + k * OC_BLOCK_SIZE*2);

        // Shuffle 2 OC x 32 ICP x 16 X of src
        const INPUT0_TYPE src0  = GET_SRC(src, 0);
        const INPUT0_TYPE src1  = GET_SRC(src, 1);
        const INPUT0_TYPE src2  = GET_SRC(src, 2);
        const INPUT0_TYPE src3  = GET_SRC(src, 3);
        const INPUT0_TYPE src4  = GET_SRC(src, 4);
        const INPUT0_TYPE src5  = GET_SRC(src, 5);
        const INPUT0_TYPE src6  = GET_SRC(src, 6);
        const INPUT0_TYPE src7  = GET_SRC(src, 7);
        const INPUT0_TYPE src8  = GET_SRC(src, 8);
        const INPUT0_TYPE src9  = GET_SRC(src, 9);
        const INPUT0_TYPE src10 = GET_SRC(src, 10);
        const INPUT0_TYPE src11 = GET_SRC(src, 11);
        const INPUT0_TYPE src12 = GET_SRC(src, 12);
        const INPUT0_TYPE src13 = GET_SRC(src, 13);
        const INPUT0_TYPE src14 = GET_SRC(src, 14);
        const INPUT0_TYPE src15 = GET_SRC(src, 15);

#if LEFTOVERS_IC
        if (k == INPUT0_FEATURE_NUM_PACKED - 1)
        {
            dst_buf[0]  += popcount((wei ^ src0) & FILTER_MASK);
            dst_buf[1]  += popcount((wei ^ src1) & FILTER_MASK);
            dst_buf[2]  += popcount((wei ^ src2) & FILTER_MASK);
            dst_buf[3]  += popcount((wei ^ src3) & FILTER_MASK);
            dst_buf[4]  += popcount((wei ^ src4) & FILTER_MASK);
            dst_buf[5]  += popcount((wei ^ src5) & FILTER_MASK);
            dst_buf[6]  += popcount((wei ^ src6) & FILTER_MASK);
            dst_buf[7]  += popcount((wei ^ src7) & FILTER_MASK);
            dst_buf[8]  += popcount((wei ^ src8) & FILTER_MASK);
            dst_buf[9]  += popcount((wei ^ src9) & FILTER_MASK);
            dst_buf[10] += popcount((wei ^ src10) & FILTER_MASK);
            dst_buf[11] += popcount((wei ^ src11) & FILTER_MASK);
            dst_buf[12] += popcount((wei ^ src12) & FILTER_MASK);
            dst_buf[13] += popcount((wei ^ src13) & FILTER_MASK);
            dst_buf[14] += popcount((wei ^ src14) & FILTER_MASK);
            dst_buf[15] += popcount((wei ^ src15) & FILTER_MASK);
            break;
        }
#endif
        dst_buf[0]  += popcount(wei ^ src0);
        dst_buf[1]  += popcount(wei ^ src1);
        dst_buf[2]  += popcount(wei ^ src2);
        dst_buf[3]  += popcount(wei ^ src3);
        dst_buf[4]  += popcount(wei ^ src4);
        dst_buf[5]  += popcount(wei ^ src5);
        dst_buf[6]  += popcount(wei ^ src6);
        dst_buf[7]  += popcount(wei ^ src7);
        dst_buf[8]  += popcount(wei ^ src8);
        dst_buf[9]  += popcount(wei ^ src9);
        dst_buf[10] += popcount(wei ^ src10);
        dst_buf[11] += popcount(wei ^ src11);
        dst_buf[12] += popcount(wei ^ src12);
        dst_buf[13] += popcount(wei ^ src13);
        dst_buf[14] += popcount(wei ^ src14);
        dst_buf[15] += popcount(wei ^ src15);
    }

    // Load data for fused operations (scales, biases, quantization thresholds, etc)
#if CUSTOM_FUSED_OPS
    FUSED_OPS_PREPARE_DATA;
#endif

    OUTPUT_TYPE dst[OC_BLOCK_SIZE];
    __attribute__((opencl_unroll_hint(OC_BLOCK_SIZE)))
    for (int oc = 0; oc < OC_BLOCK_SIZE; oc++)
    {
        CONV_RESULT_TYPE res = TO_CONV_RESULT_TYPE(INPUT0_FEATURE_NUM - 2*dst_buf[oc]);
#if CUSTOM_FUSED_OPS
        DO_ELTWISE_FUSED_OPS;
        dst[oc] = res;
#elif HAS_FUSED_OPS
        FUSED_OPS;
        dst[oc] = TO_OUTPUT_TYPE(FINAL_NAME);
#endif
    }

#if LEFTOVERS_OC
    bool in_fm = f_block*OC_BLOCK_SIZE + lid < OUTPUT_FEATURE_NUM;
    __attribute__((opencl_unroll_hint(SUB_GROUP_SIZE)))
    for (int ox = 0; ox < SUB_GROUP_SIZE; ox++) {
        int xi = (xy * XY_BLOCK_SIZE+ox) % OUTPUT_SIZE_X;
        int yi = (xy * XY_BLOCK_SIZE+ox) / OUTPUT_SIZE_X;
        bool in_x = xi < OUTPUT_SIZE_X;
        bool in_y = yi < OUTPUT_SIZE_Y;
        if (in_x && in_y && in_fm) {
            output[dst_index + yi*output_y_pitch + xi*output_x_pitch + lid] = dst[ox];
        }
    }
#else
    for (int ox = 0; ox < SUB_GROUP_SIZE; ox++) {
        int xi = (xy * XY_BLOCK_SIZE+ox) % OUTPUT_SIZE_X;
        int yi = (xy * XY_BLOCK_SIZE+ox) / OUTPUT_SIZE_X;
        bool in_x = xi < OUTPUT_SIZE_X;
        bool in_y = yi < OUTPUT_SIZE_Y;
        if (in_x && in_y)
            UNIT_BLOCK_WRITE(output, dst_index + yi*output_y_pitch + xi*output_x_pitch, dst[ox]);
    }
#endif
}
