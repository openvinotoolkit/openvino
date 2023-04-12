// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define VSIZE 8
#define VLOAD CAT(vload, VSIZE)
#define VSTORE CAT(vstore,VSIZE)
#define OUTPUT_VTYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VSIZE)

KERNEL (gather_nonzero_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    volatile __global INPUT1_TYPE* output_shape,
    __global OUTPUT_TYPE* output)
{
    int local_offset = 0;
    const int result_size = OV_INPUT_RANK * OUTPUT_FEATURE_NUM; // output shape: [ov_rank, count_nonzero]

    OUTPUT_TYPE* out_mem;
    bool use_local_mem = false;
    __local OUTPUT_TYPE out_mem_slm[MAX_LOCAL_MEM_SIZE];
#if IS_DYNAMIC
    const int dst_slm_size = TOTAL_DATA_SIZE * OV_INPUT_RANK;
    use_local_mem = dst_slm_size < MAX_LOCAL_MEM_SIZE;
#endif // IS_DYNAMIC
#ifdef USE_LOCAL_MEM
    use_local_mem = true;
#endif // USE_LOCAL_MEM
    if (use_local_mem) {
        out_mem = out_mem_slm;
    } else {
        out_mem = output;
    }

    int count_nzero = output_shape[0];
#if OV_INPUT_RANK == 1 // b
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        out_mem[local_offset++] = b;
#elif OV_INPUT_RANK == 2 // bf
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int b_pos = local_offset; \
        int f_pos = b_pos + count_nzero; \
        out_mem[b_pos] = b; \
        out_mem[f_pos] = f; \
        local_offset++;
#elif OV_INPUT_RANK == 3 // bfy
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Y_PITCH; \
        int b_pos = local_offset; \
        int f_pos = b_pos + count_nzero; \
        int y_pos = f_pos + count_nzero; \
        out_mem[b_pos] = b; \
        out_mem[f_pos] = f; \
        out_mem[y_pos] = y; \
        local_offset++;
#elif OV_INPUT_RANK == 4 // bfyx
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (y * INPUT0_Y_PITCH)) / INPUT0_X_PITCH; \
        int b_pos = local_offset; \
        int f_pos = b_pos + count_nzero; \
        int y_pos = f_pos + count_nzero; \
        int x_pos = y_pos + count_nzero; \
        out_mem[b_pos] = b; \
        out_mem[f_pos] = f; \
        out_mem[y_pos] = y; \
        out_mem[x_pos] = x; \
        local_offset++;
#elif OV_INPUT_RANK == 5 // bfzyx
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int z = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Z_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (z * INPUT0_Z_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (z * INPUT0_Z_PITCH) - (y * INPUT0_Y_PITCH)) / INPUT0_X_PITCH; \
        int b_pos = local_offset; \
        int f_pos = b_pos + count_nzero; \
        int z_pos = f_pos + count_nzero; \
        int y_pos = z_pos + count_nzero; \
        int x_pos = y_pos + count_nzero; \
        out_mem[b_pos] = b; \
        out_mem[f_pos] = f; \
        out_mem[z_pos] = z; \
        out_mem[y_pos] = y; \
        out_mem[x_pos] = x; \
        local_offset++;
#elif OV_INPUT_RANK == 6 // bfwzyx
    #define ADD_IDXS \
        int b = input_idx_v / INPUT0_BATCH_PITCH; \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int w = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_W_PITCH; \
        int z = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH)) / INPUT0_Z_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH) - (z * INPUT0_Z_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH) - (z * INPUT0_Z_PITCH) - (y * INPUT0_Y_PITCH)) / INPUT0_X_PITCH; \
        int b_pos = local_offset; \
        int f_pos = b_pos + count_nzero; \
        int w_pos = f_pos + count_nzero; \
        int z_pos = w_pos + count_nzero; \
        int y_pos = z_pos + count_nzero; \
        int x_pos = y_pos + count_nzero; \
        out_mem[b_pos] = b; \
        out_mem[f_pos] = f; \
        out_mem[w_pos] = w; \
        out_mem[z_pos] = z; \
        out_mem[y_pos] = y; \
        out_mem[x_pos] = x; \
        local_offset++;
#endif
    int input_idx = 0;
    int global_output_offset = 0;
    // load to local mem
    for (; input_idx + VSIZE <= TOTAL_DATA_SIZE; input_idx += VSIZE) {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, VSIZE) inputs = VLOAD(0, input + input_idx);
        for (int v = 0; v < VSIZE; ++v) {
            int input_idx_v = input_idx + v;
            if (inputs[v] != INPUT0_VAL_ZERO) {
                ADD_IDXS;
            }
        }
    }

    // leftovers
    for (;input_idx < TOTAL_DATA_SIZE; ++input_idx) {
        int input_idx_v = input_idx;
        int v = 0;
        if (input[input_idx] != INPUT0_VAL_ZERO) {
            ADD_IDXS;
         }
    }

    if (use_local_mem) {
        // write back to global mem
        int local_out_iter = 0;
        for (; local_out_iter + VSIZE < result_size; local_out_iter += VSIZE) {
            vstore8(VLOAD(0, out_mem + local_out_iter), 0, output + global_output_offset + local_out_iter);
        }
        // leftover
        for (; local_out_iter < result_size; ++local_out_iter) {
            output[global_output_offset + local_out_iter] = out_mem[local_out_iter];
        }
    }
}
#ifdef VLOAD
#undef VLOAD
#endif
#ifdef VSTORE
#undef VSTORE
#endif
#ifdef VSIZE
#undef VSIZE
#endif
#ifdef OUTPUT_VTYPE
#undef OUTPUT_VTYPE
#endif
