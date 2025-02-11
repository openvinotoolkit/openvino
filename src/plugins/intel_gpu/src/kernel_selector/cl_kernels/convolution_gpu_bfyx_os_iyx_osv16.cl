// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"


// ---------------------------------------------------------------------------------------------------------------------
// Just-in-time macro definitions:
// ---------------------------------------------------------------------------------------------------------------------

// Required JIT constants:
//  - INPUT                - [tensor] Input dimensions (batch, spatial and feature).
//  - OUTPUT               - [tensor] Output dimensions (batch, spatial and feature).
//  - STRIDE               - [tensor] Stride (only spatial). Factors that describe step size in X or Y dimension of
//                           input position of application of convolution filter when next ouput value
//                           (step 1 in in X or Y dimension of output) is computed.
//  - INPUT0_OFFSET        - [tensor] Offset for the first element
//                           initial offset input position of application of convolution filter and output position.
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE       - [int] Size of used subgroup (SIMD).
//  - LEFTOVERS            - [int] Optional parameter, required only when number of ofm is not dividable by SUB_GROUP_SIZE
//                           see comment for FEATURES_THREADS_PER_BATCH for more informations

/*
gpu::make_jit_constant("OUTPUT_LIMIT",              output_size),
gpu::make_jit_constant("FILTER",                    filter_mem.argument().size),
gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
gpu::make_jit_constant("OUTPUT_BLOCK_WIDTH",        _kernel_data.block_width));
gpu::make_jit_constant("OUTPUT_BLOCK_HEIGHT",       _kernel_data.block_height));
gpu::make_jit_constant("IN_BLOCK_ARRAY_SIZE",       _kernel_data.input_block_array_size));
gpu::make_jit_constant("IN_BLOCK_WIDTH",            _kernel_data.input_block_width));
gpu::make_jit_constant("PREFETCH",                  _kernel_data.prefetch));
if (_kernel_data.leftovers)
    gpu::make_jit_constant("LEFTOVERS",             _kernel_data.leftovers));
*/

// FEATURES_THREADS_PER_BATCH defines how many threads in z-dimension are processing single batch.
// ideally, z-dimension of value n should indicate processing of n-th output feature. however, since
// threads are stack in groups of SUB_GROUP_SIZE, when number of ofm is not dividable by SUB_GROUP_SIZE
// there are dummy threads added in z-dimension in count of LEFTOVERS. We need to take them into consideration
// while calculating batch's id (see lines 86-87). Values calculated by dummy threads are discarded at line 210.
#ifdef LEFTOVERS
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM + LEFTOVERS)
#else
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM)
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_os_iyx_osv16)(
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
    const uint filter_physical_len = FILTER_GROUPS_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z * FILTER_IFM_NUM * ALIGN(FILTER_OFM_NUM, OSV_SIZE);
    const uint input0_physical_len = INPUT0_OFFSET_WITH_PADDING + INPUT0_BATCH_PITCH * INPUT0_BATCH_NUM; // bfyx format

#if GROUPED
    uint batch_idx = fm / (FEATURES_THREADS_PER_BATCH * FILTER_GROUPS_NUM);
    uint feature_idx = (fm % (FEATURES_THREADS_PER_BATCH * FILTER_GROUPS_NUM) % FEATURES_THREADS_PER_BATCH);
    uint fmg = feature_idx / SUB_GROUP_SIZE;
    const uint g = (fm % (FEATURES_THREADS_PER_BATCH * FILTER_GROUPS_NUM)) / FEATURES_THREADS_PER_BATCH;
    const uint feature_num = g * FILTER_OFM_NUM + feature_idx; // feature index for fused operations
#else
    uint batch_idx = fm / FEATURES_THREADS_PER_BATCH;
    uint feature_idx = fm % FEATURES_THREADS_PER_BATCH;
    uint fmg = feature_idx / SUB_GROUP_SIZE;
    const uint g = 0;
    const uint feature_num = feature_idx; // feature index for fused operations
#endif
    UNIT_TYPE in[IN_BLOCK_ARRAY_SIZE];
    UNIT_TYPE out[OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT];
    UNIT_TYPE w[PREFETCH];
    uint in_addr;
    uint weight_addr = fmg * FILTER_IFM_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * OSV_SIZE + lid;

#if GROUPED
    weight_addr += g * FILTER_GROUPS_PITCH;
#endif

    for(int i = 0; i < (OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT); i++) {
        out[i] = UNIT_VAL_ZERO;
    }

    uint in_split_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    in_addr = batch_idx * INPUT0_BATCH_PITCH;
    in_addr += in_split_offset + INPUT0_OFFSET_WITH_PADDING + (or * STRIDE_SIZE_Y * INPUT0_Y_PITCH) + (oc * STRIDE_SIZE_X + lid) * INPUT0_X_PITCH;

    for(int kd = 0; kd < FILTER_IFM_NUM; kd++)  // _ID = 3, RGB
    {
        uint tmp_in_addr = in_addr;

#if IN_BLOCK_WIDTH % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;
            uint idx = tmp_in_addr + (in_block_pos % IN_BLOCK_WIDTH) * INPUT0_X_PITCH;
            // index clipping to avoid out-of-bound memory access. Such data is not supposed to be used in actual computation.
            idx = min(idx, input0_physical_len - 1);
            in[in_block_pos / SUB_GROUP_SIZE] = input[idx];

            // If we have row break, move to the next row.
            if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                tmp_in_addr += INPUT0_Y_PITCH;
        }
#elif (2 * IN_BLOCK_WIDTH) % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            if (in_block_next_x_pos <= IN_BLOCK_WIDTH) {
                uint idx = tmp_in_addr + (in_block_pos % IN_BLOCK_WIDTH) * INPUT0_X_PITCH;
                idx = min(idx, input0_physical_len - 1);
                in[in_block_pos / SUB_GROUP_SIZE] = input[idx];

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

        for(int pf=0; pf<PREFETCH; pf++) {
            uint weight_addr_safe = min(weight_addr, filter_physical_len - 1);
            w[pf] = weights[weight_addr_safe];
            weight_addr += OSV_SIZE;
        }

        uint wi = 0;
        uint kr = 0; // kr = Kernel Row
#ifdef DISABLE_MANUAL_UNROLL
        unroll_for (; kr < FILTER_SIZE_Y; ++kr)
#else
        LOOP(FILTER_SIZE_Y, kr,  // LOOP is a macro that unrolls the loop.
#endif
        {
            uint kc = 0; // kc = Kernel Column
#ifdef DISABLE_MANUAL_UNROLL
        unroll_for (; kc < FILTER_SIZE_X; ++kc)
            {
                unroll_for (uint br = 0; br < OUTPUT_BLOCK_HEIGHT; br++) {
                    unroll_for(uint bc = 0; bc < OUTPUT_BLOCK_WIDTH; bc++) {
#else
            LOOP(FILTER_SIZE_X, kc,
            {
                for (uint br = 0; br < OUTPUT_BLOCK_HEIGHT; br++) {
                    for(uint bc = 0; bc < OUTPUT_BLOCK_WIDTH; bc++) {
#endif

#if IN_BLOCK_WIDTH != SUB_GROUP_SIZE
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * STRIDE_SIZE_Y + kr][bc * STRIDE_SIZE_X + kc];
                        UNIT_TYPE val = _sub_group_shuffle( in[(((br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X)) / SUB_GROUP_SIZE],
                                                                    (((br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X)) % SUB_GROUP_SIZE);
#else
                        UNIT_TYPE val = _sub_group_shuffle( in[br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y], bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X);
#endif
                        out[br * OUTPUT_BLOCK_WIDTH + bc] = mad(w[wi % PREFETCH], val, out[br * OUTPUT_BLOCK_WIDTH + bc]);
                    }
                }
                uint weight_addr_safe = min(weight_addr, filter_physical_len - 1);
                w[wi % PREFETCH] = weights[weight_addr_safe];
                weight_addr += OSV_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                wi++;
#ifdef DISABLE_MANUAL_UNROLL
            }
        }
#else
            });
        });
#endif
        // addr went beyond due to prefetch so move it back to correct location.
        weight_addr -= PREFETCH * OSV_SIZE;
    }
    

    uint out_split_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    uint out_addr = OUTPUT_OFFSET;
    out_addr += batch_idx * OUTPUT_BATCH_PITCH;
    out_addr += out_split_offset + feature_idx * OUTPUT_FEATURE_PITCH; // out_addr indices into start of 16 feature maps.
    out_addr += or * OUTPUT_Y_PITCH + oc;  // offset for the 4x3 block that this workitem is working on;

#if BIAS_TERM
    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
#if BIAS_PER_OUTPUT
            unsigned bias_index = feature_idx*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + or*OUTPUT_SIZE_X + oc;
#else
            unsigned bias_index = feature_idx;
#endif
#if GROUPED
            bias_index += g * FILTER_OFM_NUM;
#endif
            out[r * OUTPUT_BLOCK_WIDTH + c] += bias[bias_index];
        }
    }
#endif


    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
#if HAS_FUSED_OPS
            UNIT_TYPE dst = out[r * OUTPUT_BLOCK_WIDTH + c];
            FUSED_OPS;
            out[r * OUTPUT_BLOCK_WIDTH + c] = FUSED_OPS_RESULT;
#else
            out[r * OUTPUT_BLOCK_WIDTH + c] = ACTIVATION(out[r * OUTPUT_BLOCK_WIDTH + c], ACTIVATION_PARAMS);
#endif
        }
    }


//--------------------------------------------------------------------
// output phase
//--------------------------------------------------------------------

#ifdef LEFTOVERS
    if (feature_idx < FILTER_OFM_NUM)
#endif
    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        if(!(or + r >= OUTPUT_SIZE_Y))
        {

#if !IS_DYNAMIC
#if (OUTPUT_SIZE_X % OUTPUT_BLOCK_WIDTH) == 0
    #define CAN_SKIP_CHECK
#endif
#endif

#ifdef CAN_SKIP_CHECK // in this case we don't need to check if we're outside of X boundaries
            uint out_vstore_offset = 0;
            #if (OUT_BLOCK_WIDTH % 8) > 3
            MAKE_VECTOR_TYPE(UNIT_TYPE, 4) tmp = MAKE_VECTOR_TYPE(UNIT_TYPE, 4)(
                out[out_vstore_offset + 0 + r * OUTPUT_BLOCK_WIDTH],
                out[out_vstore_offset + 1 + r * OUTPUT_BLOCK_WIDTH],
                out[out_vstore_offset + 2 + r * OUTPUT_BLOCK_WIDTH],
                out[out_vstore_offset + 3 + r * OUTPUT_BLOCK_WIDTH]
            );

            vstore4(tmp, 0, output + out_addr + r * OUTPUT_Y_PITCH + out_vstore_offset * OUTPUT_X_PITCH);
            out_vstore_offset += 4;
            #endif

            #if (OUT_BLOCK_WIDTH % 4) > 1
            MAKE_VECTOR_TYPE(UNIT_TYPE, 2) tmp2 = MAKE_VECTOR_TYPE(UNIT_TYPE, 2)(
                out[out_vstore_offset + 0 + r * OUTPUT_BLOCK_WIDTH],
                out[out_vstore_offset + 1 + r * OUTPUT_BLOCK_WIDTH]
            );

            vstore2(tmp2, 0, output + out_addr + r * OUTPUT_Y_PITCH + out_vstore_offset * OUTPUT_X_PITCH);
            out_vstore_offset += 2;
            #endif
            for(uint c = out_vstore_offset; c < OUTPUT_BLOCK_WIDTH; c++) {
                // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
                output[out_addr + r * OUTPUT_Y_PITCH + c] = out[r * OUTPUT_BLOCK_WIDTH + c];
            }
#else
            for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
                // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
                if(!(oc + c >= OUTPUT_SIZE_X))
                    output[out_addr + r * OUTPUT_Y_PITCH + c] = out[r * OUTPUT_BLOCK_WIDTH + c];
            }
#endif
        }
    }
}

#undef FEATURES_THREADS_PER_BATCH
