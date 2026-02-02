/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"
#ifdef BIAS_DT
DECLARE_2D_TILE(bias_tile_type, BIAS_DT, SUBGROUP_SIZE, ugemm_moe_sg_tile_m, 1, 1, 1)
#endif
DECLARE_2D_TILE(ugemm_moe_c_type_half, half, SUBGROUP_SIZE, ugemm_moe_c_type_block0, ugemm_moe_c_type_block1, ugemm_moe_c_type_nblock0, ugemm_moe_c_type_nblock1)
DECLARE_2D_TILE_COPY_REBLOCK(ugemm_moe_c_type, SUBGROUP_SIZE, ugemm_moe_c_type_block0, ugemm_moe_c_type_block1, ugemm_moe_c_type_nblock0, ugemm_moe_c_type_nblock1,
                             ugemm_moe_c_type_half, SUBGROUP_SIZE, ugemm_moe_c_type_block0, ugemm_moe_c_type_block1, ugemm_moe_c_type_nblock0, ugemm_moe_c_type_nblock1)

#define unroll_for __attribute__((opencl_unroll_hint)) for


#ifndef ENABLE_WORKLOAD_BALANCE

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global INPUT0_TYPE *input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
        const global uchar *weight_ptr,
#else
        const global INPUT1_TYPE *weight_ptr,
#endif
        global OUTPUT_TYPE *out_ptr,
#ifdef POST_PROC_SILU_MUL
        const global OUTPUT_TYPE *post_op_input,
#endif
        const global INPUT2_TYPE *experts_ids,
        const global INPUT3_TYPE *input_offset_per_expert,
        const global INPUT4_TYPE *n_array,
        int m, int k
#ifdef BIAS_DT
        , const global BIAS_DT *bias_ptr
#endif
#ifdef WEIGHT_COMPRESSED_INT4
        , const global WEIGHT_SCALE_DT *weight_scales
        #ifdef WEIGHT_ZP_DT
        , const global WEIGHT_ZP_DT *weight_zps
        #endif
#endif
#ifdef USE_SLM
        , local int* slm
#endif
) {
    uint batch = get_group_id(2);
    int input_offset = sub_group_broadcast(input_offset_per_expert[batch], 0);

    #ifdef IS_GENERATE
    if (INPUT_SEQ_LEN > 1) {
    #endif
    input_ptr += input_offset * INPUT_STRIDE;
    #ifdef IS_GENERATE
    }
    #endif
    out_ptr += input_offset * OUTPUT_STRIDE;
#ifdef POST_PROC_SILU_MUL
    post_op_input += input_offset * OUTPUT_STRIDE;
#endif
    INPUT2_TYPE expert_id = sub_group_broadcast(experts_ids[batch], 0);
    weight_ptr += expert_id * EXPERT_STRIDE;

    int ld_input = k;
#ifdef WEIGHT_COMPRESSED_INT4
    weight_scales += expert_id * m * NUM_GROUPS;
    #ifdef WEIGHT_ZP_DT
    #ifdef WEIGHT_COMPRESSED_ZP_INT4
    weight_zps += expert_id * m * NUM_GROUPS / 2;
    #else
    weight_zps += expert_id * m * NUM_GROUPS;
    #endif
    #endif
#endif
    int ld_weight = k;
    int cur_n_tokens = sub_group_broadcast(n_array[batch], 0);

    uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);
    uint sg_k = sub_group_broadcast(get_local_id(2), 0);

    uint wg_i0 = get_group_id(0) * ugemm_moe_wg_tile_m;
    uint wg_j0 = get_group_id(1) * ugemm_moe_wg_tile_n;
    uint sg_i0 = wg_i0 + sg_i * ugemm_moe_sg_tile_m;
    uint sg_j0 = wg_j0 + sg_j * ugemm_moe_sg_tile_n;
#ifdef WEIGHT_COMPRESSED_INT4
#ifdef SCALE_ZP_NO_TRANSPOSE
    /* This parameter is the leading dimension for scales/zp. Since scales/zp are non-transpose,
       the leading dimension is the stride between successive groups in the k dimension. */
    uint scale_zp_leading_dim = m;
#else
    uint scale_zp_leading_dim = NUM_GROUPS;
#endif
#endif
    if (wg_j0 >= cur_n_tokens)
        return;     /* early exit if outside batch */
#ifdef IS_GENERATE
#ifdef USE_SLM
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, sg_k, slm
#else
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, sg_k, 0
#endif
#else
#ifdef USE_SLM
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, slm
#else
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, 0
#endif
#endif
#ifdef WEIGHT_COMPRESSED_INT4
                                        , weight_scales
#ifdef WEIGHT_ZP_DT
                                        , weight_zps
#endif
                                        , scale_zp_leading_dim
#endif
);

    // Only the first sg stores data in kparallel microkernels
    if (sg_k > 0)
        return;

    ugemm_moe_c_type_half c_tile_half;
    tile_copy_reblock(c_tile, &c_tile_half);

#ifdef BIAS_DT
    bias_ptr += (expert_id * BIAS_STRIDE);
    int sglid = get_sub_group_local_id();
    const int br = ugemm_moe_c_type_block0;
    const int nbr = ugemm_moe_c_type_nblock0;
    const int bc = ugemm_moe_c_type_block1;
    const int nbc = ugemm_moe_c_type_nblock1;
    int sg = SUBGROUP_SIZE;
    for (int j = 0; j < bc * nbc; j++) {
        if (sg_j0 + j < cur_n_tokens) {
            for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                int i = i0 + sglid;
                if (sg_i0 + i < m) {
                    c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br)/sg + (j % bc) * (br / sg)] += bias_ptr[sg_i0 + i];
                }
            }
        }
    }
#endif

#ifdef POST_PROC_SILU_MUL
    {
        int sglid = get_sub_group_local_id();
        const int br = ugemm_moe_c_type_block0;
        const int nbr = ugemm_moe_c_type_nblock0;
        const int bc = ugemm_moe_c_type_block1;
        const int nbc = ugemm_moe_c_type_nblock1;
        int sg = SUBGROUP_SIZE;

        const global OUTPUT_TYPE* post_op_base = post_op_input + sg_j0 * m + sg_i0;
        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                const global OUTPUT_TYPE* post_op_row = post_op_base + j * m;
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        float post_val = post_op_row[i];
                        int reg_idx_i = (i0 / br) + nbr * (j / bc);
                        int reg_idx_j = (i0 % br)/sg + (j % bc) * (br / sg);
                        float val = c_tile_half.x[reg_idx_i][reg_idx_j];
                        float res = post_val * (val / (1.0f + native_exp(-val)));
                        c_tile_half.x[reg_idx_i][reg_idx_j] = res;
                    }
                }
            }
        }
    }
#endif

    tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
}

#else  // ENABLE_WORKLOAD_BALANCE

#ifndef MAX_EXPERTS_COUNT
#define MAX_EXPERTS_COUNT 1024
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global INPUT0_TYPE *input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
        const global uchar *weight_ptr,
#else
        const global INPUT1_TYPE *weight_ptr,
#endif
        global OUTPUT_TYPE *out_ptr,
#ifdef POST_PROC_SILU_MUL
        const global OUTPUT_TYPE *post_op_input,
#endif
        const global INPUT2_TYPE *experts_ids,
        const global INPUT3_TYPE *input_offset_per_expert,
        const global INPUT4_TYPE *n_array,
        int m, int k
#ifdef BIAS_DT
        , const global BIAS_DT *bias_ptr
#endif
#ifdef WEIGHT_COMPRESSED_INT4
        , const global WEIGHT_SCALE_DT *weight_scales
        #ifdef WEIGHT_ZP_DT
        , const global WEIGHT_ZP_DT *weight_zps
        #endif
#endif
        , int num_experts
#ifdef USE_SLM
        , local int* slm
#endif
) {
    const global INPUT0_TYPE *base_input_ptr = input_ptr;
#ifdef WEIGHT_COMPRESSED_INT4
    const global uchar *base_weight_ptr = weight_ptr;
#else
    const global INPUT1_TYPE *base_weight_ptr = weight_ptr;
#endif
    global OUTPUT_TYPE *base_out_ptr = out_ptr;
#ifdef POST_PROC_SILU_MUL
    const global OUTPUT_TYPE *base_post_op_input = post_op_input;
#endif
#ifdef BIAS_DT
    const global BIAS_DT *base_bias_ptr = bias_ptr;
#endif
#ifdef WEIGHT_COMPRESSED_INT4
    const global WEIGHT_SCALE_DT *base_weight_scales = weight_scales;
    #ifdef WEIGHT_ZP_DT
    const global WEIGHT_ZP_DT *base_weight_zps = weight_zps;
    #endif
#endif

    // LSM, Compute Prefix Scan of Tile Counts
    local uint expert_tile_offsets[MAX_EXPERTS_COUNT + 1]; // Supports up to MAX_EXPERTS_COUNT experts.
    uint lid = get_local_id(0) + get_local_id(1) * get_local_size(0);
    
    // Check if num_experts exceeds buffer
    if (num_experts > MAX_EXPERTS_COUNT) {
        return; // Early exit, or handle error as needed
    }
    if (lid < num_experts) {
        int n = n_array[lid];
        // Calculate number of tiles for this expert
        expert_tile_offsets[lid] = (n + ugemm_moe_wg_tile_n - 1) / ugemm_moe_wg_tile_n; 
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Single-thread Prefix Sum (Efficiency for small num_experts)
    if (lid == 0) {
        uint acc = 0;
        for (int i = 0; i < num_experts; ++i) {
            uint val = expert_tile_offsets[i];
            expert_tile_offsets[i] = acc;
            acc += val;
        }
        expert_tile_offsets[num_experts] = acc; // Total tiles
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint total_tiles = expert_tile_offsets[num_experts];
     
    // Flattened Persistent Loop
    // Dispatch is now 1D-like in Group counts: [1, PersistentGroups, 1]
    // We must iterate over both Tile Index (Dim1) and M-Block Index (Dim0)
    uint num_m_blocks = (m + ugemm_moe_wg_tile_m - 1) / ugemm_moe_wg_tile_m;
    uint total_tasks = total_tiles * num_m_blocks;

    uint global_stride = get_num_groups(1); // Since get_num_groups(0) is 1.
    uint global_idx = get_group_id(1);
    
    // Calculate per task.
    uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);
    uint sg_k = sub_group_broadcast(get_local_id(2), 0);

    uint expert_id = 0; // Cached expert_id
    for (uint task_idx = global_idx; task_idx < total_tasks; task_idx += global_stride) {
        // Correct task mapping: reuse expert (tile_idx) across inner M blocks
        // This ensures the same expert weight is used for M consecutive iterations if M-dim varies fast
        // For weight reuse, tile_idx will stay constant while m_block_idx changes.
        // Task_idx is linear.
        // Option A: task_0 = (tile_0, m_0), task_1 = (tile_0, m_1).
        // Option B: task_0 = (tile_0, m_0), task_1 = (tile_1, m_0).
        // A is better for L2 Cache (fetching Weight for Tile 0 once).
        uint tile_idx = task_idx / num_m_blocks;
        uint m_block_idx = task_idx % num_m_blocks;

        uint wg_i0 = m_block_idx * ugemm_moe_wg_tile_m;
        uint sg_i0 = wg_i0 + sg_i * ugemm_moe_sg_tile_m;

        // Binary Search to find Expert ID
        if (tile_idx >= expert_tile_offsets[expert_id + 1]) {
             uint l = expert_id + 1;
             uint r = num_experts;
             while (l < r) {
                 uint mid = l + (r - l) / 2;
                 if (tile_idx >= expert_tile_offsets[mid]) {
                     l = mid + 1;
                 } else {
                     r = mid;
                 }
             }
             expert_id = l - 1;
        }
        uint batch = expert_id;
        // Calculate parameters for this expert
        int cur_n_tokens = n_array[batch];

        // Calculate offsets
        int input_offset = sub_group_broadcast(input_offset_per_expert[batch], 0); 
        
        // Local tile index
        uint tile_in_expert = tile_idx - expert_tile_offsets[expert_id];
        uint wg_j0 = tile_in_expert * ugemm_moe_wg_tile_n;
        uint sg_j0 = wg_j0 + sg_j * ugemm_moe_sg_tile_n;
        
        if (wg_j0 >= cur_n_tokens) continue;

        // Pointers Setup
        const global INPUT0_TYPE *curr_input_ptr = base_input_ptr;
        // It is more benefit for prefilling stage, do we need generate stage? default is no.
        #ifdef IS_GENERATE
        if (INPUT_SEQ_LEN > 1) {
        #endif
        curr_input_ptr += input_offset * INPUT_STRIDE;
        #ifdef IS_GENERATE
        }
        #endif
        
        global OUTPUT_TYPE *curr_out_ptr = base_out_ptr + input_offset * OUTPUT_STRIDE;
        
        #ifdef POST_PROC_SILU_MUL
        const global OUTPUT_TYPE *curr_post_op_input = base_post_op_input + input_offset * OUTPUT_STRIDE;
        #endif

        INPUT2_TYPE actual_expert_id = experts_ids[batch]; 
        
        const global INPUT1_TYPE *curr_weight_ptr = (const global INPUT1_TYPE *)base_weight_ptr; // Cast for generic
        #ifdef WEIGHT_COMPRESSED_INT4
             const global uchar *curr_weight_ptr_u8 = base_weight_ptr + actual_expert_id * EXPERT_STRIDE;
        #else
             curr_weight_ptr += actual_expert_id * EXPERT_STRIDE;
        #endif

        int ld_input = k;
        int ld_weight = k;
        
#ifdef WEIGHT_COMPRESSED_INT4
        const global WEIGHT_SCALE_DT *curr_weight_scales = base_weight_scales + actual_expert_id * m * NUM_GROUPS;
        #ifdef WEIGHT_ZP_DT
            const global WEIGHT_ZP_DT *curr_weight_zps = base_weight_zps;
            #ifdef WEIGHT_COMPRESSED_ZP_INT4
            curr_weight_zps += actual_expert_id * m * NUM_GROUPS / 2;
            #else
            curr_weight_zps += actual_expert_id * m * NUM_GROUPS;
            #endif
        #endif
        
        #ifdef SCALE_ZP_NO_TRANSPOSE
            uint scale_zp_leading_dim = m;
        #else
            uint scale_zp_leading_dim = NUM_GROUPS;
        #endif
#endif

    // Default will not enter generate stage
#ifdef IS_GENERATE
#ifdef USE_SLM
    ugemm_moe_c_type c_tile = ugemm_moe(
        #ifdef WEIGHT_COMPRESSED_INT4
            curr_weight_ptr_u8,
        #else
            curr_weight_ptr,
        #endif
        ld_weight, curr_input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, sg_k, slm
#else
    ugemm_moe_c_type c_tile = ugemm_moe(
        #ifdef WEIGHT_COMPRESSED_INT4
            curr_weight_ptr_u8,
        #else
            curr_weight_ptr,
        #endif
        ld_weight, curr_input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, sg_k, 0
#endif
#else
#ifdef USE_SLM
    ugemm_moe_c_type c_tile = ugemm_moe(
        #ifdef WEIGHT_COMPRESSED_INT4
            curr_weight_ptr_u8,
        #else
            curr_weight_ptr,
        #endif
        ld_weight, curr_input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, slm
#else
    ugemm_moe_c_type c_tile = ugemm_moe(
        #ifdef WEIGHT_COMPRESSED_INT4
            curr_weight_ptr_u8,
        #else
            curr_weight_ptr,
        #endif
        ld_weight, curr_input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, 0
#endif
#endif
#ifdef WEIGHT_COMPRESSED_INT4
                                        , curr_weight_scales
#ifdef WEIGHT_ZP_DT
                                        , curr_weight_zps
#endif
                                        , scale_zp_leading_dim
#endif
);

    // Only the first sg stores data in kparallel microkernels
    if (sg_k > 0)
        continue;

    ugemm_moe_c_type_half c_tile_half;
    tile_copy_reblock(c_tile, &c_tile_half);

#ifdef BIAS_DT
    const global BIAS_DT *curr_bias_ptr = base_bias_ptr + (actual_expert_id * BIAS_STRIDE);
    int sglid = get_sub_group_local_id();
    const int br = ugemm_moe_c_type_block0;
    const int nbr = ugemm_moe_c_type_nblock0;
    const int bc = ugemm_moe_c_type_block1;
    const int nbc = ugemm_moe_c_type_nblock1;
    int sg = SUBGROUP_SIZE;
    for (int j = 0; j < bc * nbc; j++) {
        if (sg_j0 + j < cur_n_tokens) {
            for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                int i = i0 + sglid;
                if (sg_i0 + i < m) {
                    c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br)/sg + (j % bc) * (br / sg)] += curr_bias_ptr[sg_i0 + i];
                }
            }
        }
    }
#endif

#ifdef POST_PROC_SILU_MUL
    {
        int sglid = get_sub_group_local_id();
        const int br = ugemm_moe_c_type_block0;
        const int nbr = ugemm_moe_c_type_nblock0;
        const int bc = ugemm_moe_c_type_block1;
        const int nbc = ugemm_moe_c_type_nblock1;
        int sg = SUBGROUP_SIZE;

        const global OUTPUT_TYPE* post_op_base = curr_post_op_input + sg_j0 * m + sg_i0;
        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                const global OUTPUT_TYPE* post_op_row = post_op_base + j * m;
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        float post_val = post_op_row[i];
                        int reg_idx_i = (i0 / br) + nbr * (j / bc);
                        int reg_idx_j = (i0 % br)/sg + (j % bc) * (br / sg);
                        float val = c_tile_half.x[reg_idx_i][reg_idx_j];
                        float res = post_val * (val / (1.0f + native_exp(-val)));
                        c_tile_half.x[reg_idx_i][reg_idx_j] = res;
                    }
                }
            }
        }
    }
#endif

    tile_store(c_tile_half, curr_out_ptr, m, cur_n_tokens, sg_i0, sg_j0);

    } // End of Scan Loop
}

#endif