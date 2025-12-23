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

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global INPUT0_TYPE *input_ptr,
#ifdef WEIGHT_COMPRESSED_INT4
        const global uchar *weight_ptr,
#else
        const global INPUT1_TYPE *weight_ptr,
#endif
        global OUTPUT_TYPE *out_ptr,
        const global INPUT2_TYPE *experts_ids,
        const global INPUT3_TYPE * input_offset_per_expert,
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
    int input_offset = input_offset_per_expert[batch];

    #ifdef IS_GENERATE
    if (INPUT_SEQ_LEN > 1) {
    #endif
    input_ptr += input_offset * INPUT_STRIDE;
    #ifdef IS_GENERATE
    }
    #endif
    out_ptr += input_offset * OUTPUT_STRIDE;
    weight_ptr += experts_ids[batch] * EXPERT_STRIDE;

    int ld_input = k;
#ifdef WEIGHT_COMPRESSED_INT4
    weight_scales += experts_ids[batch] * m * NUM_GROUPS;
    #ifdef WEIGHT_ZP_DT
    #ifdef WEIGHT_COMPRESSED_ZP_INT4
    weight_zps += experts_ids[batch] * m * NUM_GROUPS / 2;
    #else
    weight_zps += experts_ids[batch] * m * NUM_GROUPS;
    #endif
    #endif
#endif
    int ld_weight = k;
    int cur_n_tokens = n_array[batch];

    uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);

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
#ifdef USE_SLM
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, slm
#else
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, cur_n_tokens, k, wg_i0, wg_j0, 0, sg_i, sg_j, 0
#endif
#ifdef WEIGHT_COMPRESSED_INT4
                                        , weight_scales
#ifdef WEIGHT_ZP_DT
                                        , weight_zps
#endif
                                        , scale_zp_leading_dim
#endif
);
    ugemm_moe_c_type_half c_tile_half;
    tile_copy_reblock(c_tile, &c_tile_half);

#ifdef BIAS_DT
    bias_ptr += (experts_ids[batch] * BIAS_STRIDE);
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
                if (sg_i0 + i < m)
                    c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br)/sg + (j % bc) * (br / sg)] += bias_ptr[sg_i0 + i];
            }
        }
    }
#endif
    tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
}
