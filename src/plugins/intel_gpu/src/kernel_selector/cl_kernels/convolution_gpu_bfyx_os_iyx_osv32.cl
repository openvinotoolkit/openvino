// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

#ifdef LEFTOVERS
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM + LEFTOVERS)
#else
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM)
#endif

#if FILTER_OFM_NUM <= 16
    #define NUM_CALC_UNIT_SIZE 1
    #define CALC_UNIT_TYPE UNIT_TYPE
    #define CALC_UNIT_BLOCK_READ UNIT_BLOCK_READ
#else
    #define NUM_CALC_UNIT_SIZE 2
    #define CALC_UNIT_TYPE UNIT_TYPE2
    #define CALC_UNIT_BLOCK_READ UNIT_BLOCK_READ2
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_os_iyx_osv32)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights
#if BIAS_TERM
    , const __global UNIT_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint oc  = (uint)get_global_id(0) * OUTPUT_BLOCK_WIDTH;  // oc = Output Column
    const uint or  = (uint)get_global_id(1) * OUTPUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm  = get_global_id(2);                             // fm = Feature Map = od = Output Depth
    const uint lid = get_sub_group_local_id();
    const uint batch_idx = (fm / SUB_GROUP_SIZE) % OUTPUT_BATCH_NUM;
    const uint fmg = (fm / SUB_GROUP_SIZE) / OUTPUT_BATCH_NUM;
    const uint feature_idx = fmg * OSV_SIZE + lid;

    UNIT_TYPE in[IN_BLOCK_ARRAY_SIZE];
    CALC_UNIT_TYPE out[OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT];

    unroll_for (int i = 0; i < (OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT); ++i) {
        out[i] = UNIT_VAL_ZERO;
    }

    uint in_addr = batch_idx * INPUT0_BATCH_PITCH + INPUT0_OFFSET_WITH_PADDING +
                   (or * STRIDE_SIZE_Y * INPUT0_Y_PITCH) + (oc * STRIDE_SIZE_X + lid) * INPUT0_X_PITCH;

    uint weight_addr = fmg * FILTER_IFM_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * OSV_SIZE;

    for (uint kd = 0; kd < FILTER_IFM_NUM; ++kd)
    {
        uint tmp_in_addr = in_addr;

#if IN_BLOCK_WIDTH % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + (in_block_pos % IN_BLOCK_WIDTH) * INPUT0_X_PITCH];

            // If we have row break, move to the next row.
            if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                tmp_in_addr += INPUT0_Y_PITCH;
        }
#elif (2 * IN_BLOCK_WIDTH) % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            if (in_block_next_x_pos <= IN_BLOCK_WIDTH) { //
                in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + (in_block_pos % IN_BLOCK_WIDTH) * INPUT0_X_PITCH];

                // If we have row break, move to the next row.
                if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT0_Y_PITCH;
            }
            else {
                // TODO: Generalize this step to relax IN_BLOCK_WIDTH restrictions.
                // Position in sub-group on which new row need to be read.
                const uint sg_br_pos = IN_BLOCK_WIDTH - in_block_pos % IN_BLOCK_WIDTH;

                if (lid < sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + (in_block_pos % IN_BLOCK_WIDTH) * INPUT0_X_PITCH];
                // We have row break inside sub-group. Need to move to next line.
                tmp_in_addr += INPUT0_Y_PITCH;
                if (lid >= sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr - (sg_br_pos * INPUT0_X_PITCH)];

                // If we have another row break, move to the next row.
                if (in_block_next_x_pos == 2 * IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT0_Y_PITCH;
            }
        }
#else
    #error IN_BLOCK_WIDTH must be multiple of SUB_GROUP_SIZE or half of SUB_GROUP_SIZE. Other scenarios are not currently implemented.
#endif

        //move to next filter
        in_addr += INPUT0_FEATURE_PITCH;

        unroll_for (uint kr = 0; kr < FILTER_SIZE_Y; ++kr) {
            unroll_for (uint kc = 0; kc < FILTER_SIZE_X; ++kc) {
                CALC_UNIT_TYPE w = CALC_UNIT_BLOCK_READ(weights, weight_addr);
                unroll_for (uint br=0; br<OUTPUT_BLOCK_HEIGHT; ++br) {
                    uint y_pos = br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y;
                    unroll_for (uint bc=0; bc<OUTPUT_BLOCK_WIDTH; ++bc) {
                        uint x_pos = bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X;
                        #if IN_BLOCK_WIDTH != SUB_GROUP_SIZE
                            UNIT_TYPE val = sub_group_broadcast(in[((y_pos * IN_BLOCK_WIDTH) + x_pos) / SUB_GROUP_SIZE],
                                                                ((y_pos * IN_BLOCK_WIDTH) + x_pos) % SUB_GROUP_SIZE);
                        #else
                            UNIT_TYPE val = sub_group_broadcast(in[y_pos], x_pos);
                        #endif

                        out[br * OUTPUT_BLOCK_WIDTH + bc] = mad(w, val, out[br * OUTPUT_BLOCK_WIDTH + bc]);
                    }
                }
                weight_addr += OSV_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
            }
        }
    }

#if BIAS_TERM
    for (uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for (uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
            #if BIAS_PER_OUTPUT
                unsigned bias_index = feature_idx * OUTPUT_SIZE_X * OUTPUT_SIZE_Y + or * OUTPUT_SIZE_X + oc;
                #if NUM_CALC_UNIT_SIZE == 1
                    out[r * OUTPUT_BLOCK_WIDTH + c] += bias[bias_index];
                #else
                    out[r * OUTPUT_BLOCK_WIDTH + c].s0 += bias[bias_index];
                    bias_index += SUB_GROUP_SIZE * OUTPUT_SIZE_X * OUTPUT_SIZE_Y;
                    out[r * OUTPUT_BLOCK_WIDTH + c].s1 += bias[bias_index];
                #endif
            #else
                unsigned bias_index = feature_idx - lid;
                CALC_UNIT_TYPE bias_read = CALC_UNIT_BLOCK_READ(bias, (feature_idx - lid));
                out[r * OUTPUT_BLOCK_WIDTH + c] += bias_read;
            #endif
        }
    }
#endif

//--------------------------------------------------------------------
// output phase
//--------------------------------------------------------------------

    for (uint fid = 0; fid < NUM_CALC_UNIT_SIZE; ++fid) {
        if ((feature_idx + SUB_GROUP_SIZE * fid) < FILTER_OFM_NUM) {
            uint out_addr = OUTPUT_OFFSET;
            out_addr += batch_idx * OUTPUT_BATCH_PITCH;
            out_addr += (feature_idx + SUB_GROUP_SIZE * fid) * OUTPUT_FEATURE_PITCH;
            out_addr += or * OUTPUT_Y_PITCH + oc;

            for (uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
                if (or + r < OUTPUT_SIZE_Y)
                {
                    for (uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
                        if (oc + c < OUTPUT_SIZE_X) {
                            #if NUM_CALC_UNIT_SIZE == 1
                                UNIT_TYPE dst = out[r * OUTPUT_BLOCK_WIDTH + c];
                            #else
                                UNIT_TYPE dst = out[r * OUTPUT_BLOCK_WIDTH + c][fid];
                            #endif
                            #if HAS_FUSED_OPS
                                uint feature_num = feature_idx + SUB_GROUP_SIZE * fid;
                                FUSED_OPS;
                                dst = FUSED_OPS_RESULT;
                            #else
                                dst = ACTIVATION(dst, ACTIVATION_PARAMS);
                            #endif

                            output[out_addr + r * OUTPUT_Y_PITCH + c] = dst;
                        }
                    }
                }
            }
        }
    }
}

#undef FEATURES_THREADS_PER_BATCH
