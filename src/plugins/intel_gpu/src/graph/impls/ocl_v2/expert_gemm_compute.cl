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

// Shared expert GEMM computation body (GatherMatmul + MOE).
// #include this inside a kernel function after the dispatch preamble.
//
// Preamble must set before #include:
//   expert_id    — expert index for weight/scale/zp/bias lookup
//   input_ptr    — already offset to this dispatch's input activations
//   cur_n_tokens — number of tokens in this dispatch unit
//   m, k         — output features and input features (kernel params)
//   weight_ptr   — raw kernel param (offset by expert here)
//   weight_scales, weight_zps — raw kernel params (offset by expert here, if quantized)
//   bias_ptr     — raw kernel param (if BIAS_DT defined)
//   post_op_input — already offset (if POST_PROC_SILU_MUL defined)
//   slm          — shared local memory (if USE_SLM defined)
//
// After this block:
//   c_tile_half — result tile in half precision, ready for store
//   sg_i0, sg_j0 — subgroup tile coordinates for store
//
// When SWIGLU_FUSED is defined:
//   m = N (output features after SwiGLU)
//   SWIGLU_LENGTH = N, SWIGLU_GATE_IDX, SWISH_BETA
//   GEMM runs twice (gate at wg_i0, value at wg_i0 + N), SwiGLU in-register

    // --- GEMM dimension: m is output size, m_gemm is full weight dimension ---
#ifdef SWIGLU_FUSED
    int m_gemm = m + SWIGLU_LENGTH;
#define OUTER_OFM 2
#else
    int m_gemm = m;
#define OUTER_OFM 1
#endif

    // --- Weight / scale / zero-point offset by expert ---
    weight_ptr += expert_id * EXPERT_STRIDE;

    int ld_input = k;
#ifdef WEIGHT_COMPRESSED_INT4
    weight_scales += expert_id * m_gemm * NUM_GROUPS;
#    ifdef WEIGHT_ZP_DT
#        ifdef WEIGHT_COMPRESSED_ZP_INT4
    weight_zps += expert_id * m_gemm * NUM_GROUPS / 2;
#        else
    weight_zps += expert_id * m_gemm * NUM_GROUPS;
#        endif
#    endif
#endif
    int ld_weight = k;

    // --- Subgroup / workgroup tile index setup ---
    uint sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);
    uint sg_k = sub_group_broadcast(get_local_id(2), 0);

    uint wg_i0 = get_group_id(0) * UGEMM_WG_TILE_M;
    uint wg_j0 = get_group_id(1) * UGEMM_WG_TILE_N;
    uint sg_i0 = wg_i0 + sg_i * UGEMM_SG_TILE_M;
    uint sg_j0 = wg_j0 + sg_j * UGEMM_SG_TILE_N;

#ifdef WEIGHT_COMPRESSED_INT4
#    ifdef SCALE_ZP_NO_TRANSPOSE
    uint scale_zp_leading_dim = m_gemm;
#    else
    uint scale_zp_leading_dim = NUM_GROUPS;
#    endif
#endif

    if (wg_j0 >= cur_n_tokens)
        return;

    // --- Micro-kernel GEMM calls (OUTER_OFM=1 normal, =2 for SwiGLU) ---
    UGEMM_C_TYPE_HALF tiles[OUTER_OFM];

    __attribute__((opencl_unroll_hint(1)))
    for (uint oi = 0; oi < OUTER_OFM; ++oi) {
#ifdef SWIGLU_FUSED
        uint m_offset = wg_i0 + oi * SWIGLU_LENGTH;
#else
        uint m_offset = wg_i0;
#endif

#ifdef IS_GENERATE
#    ifdef USE_SLM
        UGEMM_C_TYPE c_tile = UG()(weight_ptr, ld_weight, input_ptr, ld_input, m_gemm, cur_n_tokens, k, m_offset, wg_j0, 0, sg_i, sg_j, sg_k, slm
#    else
        UGEMM_C_TYPE c_tile = UG()(weight_ptr, ld_weight, input_ptr, ld_input, m_gemm, cur_n_tokens, k, m_offset, wg_j0, 0, sg_i, sg_j, sg_k, 0
#    endif
#else
#    ifdef USE_SLM
        UGEMM_C_TYPE c_tile = UG()(weight_ptr, ld_weight, input_ptr, ld_input, m_gemm, cur_n_tokens, k, m_offset, wg_j0, 0, sg_i, sg_j, slm
#    else
        UGEMM_C_TYPE c_tile = UG()(weight_ptr, ld_weight, input_ptr, ld_input, m_gemm, cur_n_tokens, k, m_offset, wg_j0, 0, sg_i, sg_j, 0
#    endif
#endif
#ifdef WEIGHT_COMPRESSED_INT4
                                            , weight_scales
#    ifdef WEIGHT_ZP_DT
                                            , weight_zps
#    endif
                                            , scale_zp_leading_dim
#endif
        );

        // kparallel: all subgroups must call ugemm, only sg_k=0 saves the result
        if (sg_k == 0)
            tile_copy_reblock(c_tile, &tiles[oi]);
    }

    // kparallel: non-primary subgroups exit after all GEMM calls complete
    if (sg_k > 0)
        return;

    UGEMM_C_TYPE_HALF c_tile_half = tiles[0];

    // --- Fused SwiGLU: result = value * swish(gate) ---
#ifdef SWIGLU_FUSED
    {
        int sglid = get_sub_group_local_id();
        const int br = UGEMM_C_TYPE_BLOCK0;
        const int nbr = UGEMM_C_TYPE_NBLOCK0;
        const int bc = UGEMM_C_TYPE_BLOCK1;
        const int nbc = UGEMM_C_TYPE_NBLOCK1;
        int sg = SUBGROUP_SIZE;

        unroll_for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                unroll_for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        int ri = i0 / br + nbr * (j / bc);
                        int rj = (i0 % br) / sg + (j % bc) * (br / sg);
#if SWIGLU_GATE_IDX == 0
                        float gate = tiles[0].x[ri][rj];
                        float value = tiles[1].x[ri][rj];
#else
                        float gate = tiles[1].x[ri][rj];
                        float value = tiles[0].x[ri][rj];
#endif
                        float swish = gate / (1.0f + native_exp(-SWISH_BETA * gate));
                        c_tile_half.x[ri][rj] = (half)(value * swish);
                    }
                }
            }
        }
    }
#endif

    // --- Bias addition ---
#ifdef BIAS_DT
    {
        const global BIAS_DT* bias_base = bias_ptr + expert_id * BIAS_STRIDE;
        int sglid = get_sub_group_local_id();
        const int br = UGEMM_C_TYPE_BLOCK0;
        const int nbr = UGEMM_C_TYPE_NBLOCK0;
        const int bc = UGEMM_C_TYPE_BLOCK1;
        const int nbc = UGEMM_C_TYPE_NBLOCK1;
        int sg = SUBGROUP_SIZE;
        for (int j = 0; j < bc * nbc; j++) {
            if (sg_j0 + j < cur_n_tokens) {
                for (int i0 = 0; i0 < br * nbr; i0 += sg) {
                    int i = i0 + sglid;
                    if (sg_i0 + i < m) {
                        c_tile_half.x[i0 / br + nbr * (j / bc)][(i0 % br) / sg + (j % bc) * (br / sg)] += bias_base[sg_i0 + i];
                    }
                }
            }
        }
    }
#endif

    // --- Fused activation: SiLU-gated multiply (MOE gate * SiLU(up)) ---
#ifdef POST_PROC_SILU_MUL
    {
        int sglid = get_sub_group_local_id();
        const int br = UGEMM_C_TYPE_BLOCK0;
        const int nbr = UGEMM_C_TYPE_NBLOCK0;
        const int bc = UGEMM_C_TYPE_BLOCK1;
        const int nbc = UGEMM_C_TYPE_NBLOCK1;
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
                        int reg_idx_j = (i0 % br) / sg + (j % bc) * (br / sg);
                        float val = c_tile_half.x[reg_idx_i][reg_idx_j];
                        float res = post_val * (val / (1.0f + native_exp(-val)));
                        c_tile_half.x[reg_idx_i][reg_idx_j] = res;
                    }
                }
            }
        }
    }
#endif
#undef OUTER_OFM
