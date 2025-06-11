// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef SECOND_TOKEN_A
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(horizontal_fused_second_token_a)(OPTIONAL_SHAPE_INFO_ARG
                                        __global INPUT1_TYPE* lora_input,
                                        __global STATE_TYPE* state_a_0,
                                        __global STATE_TYPE* state_a_1,
#if LORA_COUNT == 3
                                        __global STATE_TYPE* state_a_2,
#endif
                                        __global ACCUMULATOR_TYPE* output_a)
{
    int gid0 = get_group_id(0);
    int gid2 = get_group_id(2);
    int sgid = get_sub_group_id();

    int sgN = LORA_RANK / SUBGROUP_SIZE;
    int gemma_sgK = GEMMA_SGK;
    if (LORA_RANK * LORA_COUNT <= MAX_WORKGROUP_SIZE) {
        sgN *= LORA_COUNT;
        gemma_sgK /= LORA_COUNT;
    }

    int sgid_k = sgid / sgN;

    int n_idx = (gid2 * sgN + sgid % sgN) * SUBGROUP_SIZE;
    int n_off = n_idx % LORA_RANK;
    int state_a_idx = n_idx / LORA_RANK;

    __global STATE_TYPE* state_a = state_a_idx == 0 ? state_a_0 : state_a_1;
#if LORA_COUNT == 3
    state_a = state_a_idx == 2 ? state_a_2 : state_a;
#endif

    state_a += n_off;

    int lid = get_sub_group_local_id();

    // How many K is accumulated in the WG.
    int bk_wg = gemma_sgK * GEMMA_SG_BK;
    int k_start_wg = gid0 * bk_wg;
    int wg_k_len = (k_start_wg + bk_wg) > K ? (K - k_start_wg) : bk_wg;
    int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

    // Store each sg accumulation result into SLM. Will reduce sg result into wg result.
    __local ACCUMULATOR_TYPE fma_buff[MAX_GEMMA_SGK * MAX_GEMMA_N];
    __local ACCUMULATOR_TYPE* sg_fma_buff = fma_buff + sgid_k * MAX_GEMMA_N;
    int k_offset = sgid_k * GEMMA_SG_BK;
    int k_idx = k_start_wg + k_offset;

    // The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
    if (sgid_k * GEMMA_SG_BK < wg_k_len) {
        int klen_sg = (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
        __global INPUT1_TYPE* A_ptr = lora_input + k_start_wg + k_offset;
        __global STATE_TYPE* B_ptr = state_a + k_idx * LORA_RANK;

        ACCUMULATOR_TYPE sum = 0.f;
        for (int kk = 0; kk < klen_sg; kk += SUBGROUP_SIZE) {
#if INPUT1_TYPE_SIZE == 4
            uint input = intel_sub_group_block_read((const __global uint*)(A_ptr + kk));
#else
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr + kk));
#endif
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SUBGROUP_SIZE; j++) {

#if STATE_TYPE_SIZE == 4
                ACCUMULATOR_TYPE bb = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read((const __global uint*)(B_ptr))));
#else
                ACCUMULATOR_TYPE bb = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read_us((const __global ushort*)(B_ptr))));
#endif
                ACCUMULATOR_TYPE aa = TO_ACCUMULATOR_TYPE(AS_INPUT1_TYPE(sub_group_broadcast(input, j)));

                sum = fma(aa, bb, sum);
                B_ptr += LORA_RANK;
            }
        }
        *(sg_fma_buff + n_idx + lid) = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __global ACCUMULATOR_TYPE* C_ptr = output_a + LORA_RANK * LORA_COUNT * gid0;

    // Only need sg on N dimenion to update data.
    if (sgid_k != 0) {
        return;
    }

    ACCUMULATOR_TYPE sum = 0.f;
    if (sgK == gemma_sgK) {
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < gemma_sgK; i++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
#else
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
#endif
        }
    } else {
        // Can't unroll, tail handling
        for (int i = 0; i < sgK; i++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
#else
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
#endif
        }
    }
#if ACCUMULATOR_TYPE_SIZE == 4
    intel_sub_group_block_write((const __global uint*)(C_ptr + n_idx), as_int(sum));
#else
    intel_sub_group_block_write_us((const __global ushort*)(C_ptr + n_idx), as_short(sum));
#endif
}
#endif

#ifdef SECOND_TOKEN_B
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(second_token_b)(OPTIONAL_SHAPE_INFO_ARG
                       __global OUTPUT_TYPE* main_input,
                       __global ACCUMULATOR_TYPE* output_a,
                       __global STATE_TYPE* state_alpha_0,
                       __global STATE_TYPE* state_b_0,
                       __global STATE_TYPE* state_alpha_1,
                       __global STATE_TYPE* state_b_1,
#if LORA_COUNT == 3
                       __global STATE_TYPE* state_alpha_2,
                       __global STATE_TYPE* state_b_2,
#endif
                       __global OUTPUT_TYPE* output) {
    int wg_id = get_group_id(0);
    int sg_id = get_sub_group_id();
    int sg_num = get_num_sub_groups();
    int n_idx = (wg_id * sg_num + sg_id) * SUBGROUP_SIZE;
    int id_sg_local = get_sub_group_local_id();

    // One WG maybe would cross Q, K, V. So reduce the 3 A matrice in each WG for easiness.
    __local ACCUMULATOR_TYPE reduce[MAX_GEMMA_N];

    __global STATE_TYPE* state_b = state_b_0 + n_idx;
    __global STATE_TYPE* state_alpha = state_alpha_0;

    int slm_offset = 0;
    int b_stride = N0;

#if LORA_COUNT == 3
    if (n_idx >= (N0 + N1_2)) {
        // V projection
        state_b = state_b_2 + n_idx - N0 - N1_2;
        b_stride = N1_2;
        state_alpha = state_alpha_2;
        slm_offset = LORA_RANK * 2;
    } else
#endif
    if (n_idx >= N0) {
        // K projection
        state_b = state_b_1 + n_idx - N0;
        state_alpha = state_alpha_1;
        slm_offset = LORA_RANK;
#if LORA_COUNT == 3
        b_stride = N1_2;
#endif
    }

    // 1. Reduce
    // Eech WG would reduce input activation and save into local memory `reduce[LORA_RANK]`.
    int local_sz = get_local_size(0);
    for (int offset = sg_id * SUBGROUP_SIZE; offset < MAX_GEMMA_N; offset += local_sz) {
        __global ACCUMULATOR_TYPE *A_ptr = output_a + offset;
        ACCUMULATOR_TYPE sum = 0.f;

        __attribute__((opencl_unroll_hint))
        for (int part_idx = 0; part_idx < GEMMB_PART_NUM; part_idx++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            ACCUMULATOR_TYPE partial_val = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)A_ptr));
#else
            ACCUMULATOR_TYPE partial_val = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)A_ptr));
#endif
            sum += partial_val;
            A_ptr += LORA_RANK * LORA_COUNT;
        }
        reduce[offset + id_sg_local] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (n_idx >= N) {
        return;
    }

    // 2. GEMMB
    __global OUTPUT_TYPE* C_ptr = output + n_idx;
    __local ACCUMULATOR_TYPE* reduce_ptr = reduce + slm_offset;
    ACCUMULATOR_TYPE sum = 0;

    __attribute__((opencl_unroll_hint))
    for (int kk = 0; kk < LORA_RANK; kk += SUBGROUP_SIZE) {
#if STATE_TYPE_SIZE == 4
        ACCUMULATOR_TYPE scale = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read((const __global uint*)(state_alpha + kk))));
#else
        ACCUMULATOR_TYPE scale = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read_us((const __global ushort*)(state_alpha + kk))));
#endif

#if ACCUMULATOR_TYPE_SIZE == 4
        ACCUMULATOR_TYPE input = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(reduce_ptr + kk)));
#else
        ACCUMULATOR_TYPE input = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(reduce_ptr + kk)));
#endif
        input *= scale / TO_ACCUMULATOR_TYPE(LORA_RANK);

        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < SUBGROUP_SIZE; j++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            ACCUMULATOR_TYPE aa = AS_ACCUMULATOR_TYPE(sub_group_broadcast(as_uint(input), j));
#else
            ACCUMULATOR_TYPE aa = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(as_ushort(input), j));
#endif

#if STATE_TYPE_SIZE == 4
            ACCUMULATOR_TYPE bb = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read((const __global uint*)state_b)));
#else
            ACCUMULATOR_TYPE bb = TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(intel_sub_group_block_read_us((const __global ushort*)state_b)));
#endif
            sum = fma(aa, bb, sum);
            state_b += b_stride;
        }
    }
    __global OUTPUT_TYPE* main_input_ptr = main_input + n_idx;
#if INPUT0_TYPE_SIZE == 4
    INPUT0_TYPE m_input = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)main_input_ptr));
    intel_sub_group_block_write((const __global uint*)C_ptr, as_int(sum + m_input));
#else
    INPUT0_TYPE m_input = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));
    intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum + m_input));
#endif
}
#endif

#ifdef FIRST_TOKEN_A
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(first_token_a)(OPTIONAL_SHAPE_INFO_ARG
                      __global INPUT1_TYPE* lora_input,
                      __global STATE_TYPE* state_a_0,
                      __global STATE_TYPE* state_alpha_0,
                      __global STATE_TYPE* state_a_1,
                      __global STATE_TYPE* state_alpha_1,
#if LORA_COUNT == 3
                      __global STATE_TYPE* state_a_2,
                      __global STATE_TYPE* state_alpha_2,
#endif
                      __global ACCUMULATOR_TYPE* C)
{
    int sgid = get_sub_group_id();
    int sgN = get_local_size(1) / SUBGROUP_SIZE;
    int sgM = get_local_size(0);
    int sgid_N = sgid % sgN;
    int sgid_M = sgid / sgN;

    int BM = REG_M * sgM;
    int BN = get_local_size(1) * REG_N;

    int m_idx = get_group_id(0) * BM + sgid_M * REG_M;
    int n_idx = get_group_id(1) * BN + sgid_N * SUBGROUP_SIZE * REG_N;

    if (m_idx >= M || n_idx >= N) {
        return;
    }

    if (m_idx + REG_M > M) {
        m_idx = M - REG_M;
    }

    if (n_idx + REG_N * SUBGROUP_SIZE > N) {
        n_idx = N - REG_N * SUBGROUP_SIZE;
    }

    int strideA = K;
    int strideB = LORA_RANK;

    __global INPUT1_TYPE* ptrA = lora_input + m_idx * strideA;
    __global STATE_TYPE* ptrB = state_a_0 + n_idx;
    __global STATE_TYPE* alpha_ptr = state_alpha_0 + n_idx;
    __global ACCUMULATOR_TYPE* ptrC = C + m_idx * N + n_idx;

#if LORA_COUNT == 3
    if (n_idx >= LORA_RANK * 2) {
        ptrB = state_a_2 + n_idx - LORA_RANK * 2;
        alpha_ptr = state_alpha_2 + n_idx - LORA_RANK * 2;
    } else
#endif
    if (n_idx >= LORA_RANK) {
        ptrB = state_a_1 + n_idx - LORA_RANK;
        alpha_ptr = state_alpha_1 + n_idx - LORA_RANK;
    }

    MAIN_MATMUL_CODE

    MULTIPLY_AND_STORE_CODE
}
#endif


#ifdef FIRST_TOKEN_B
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(first_token_b)(OPTIONAL_SHAPE_INFO_ARG
                      __global OUTPUT_TYPE* main_input,
                      __global ACCUMULATOR_TYPE* output_a,
                      __global STATE_TYPE* state_b_0,
                      __global STATE_TYPE* state_b_1,
#if LORA_COUNT == 3
                      __global STATE_TYPE* state_b_2,
#endif
                      __global OUTPUT_TYPE* output)
{
    int sgid = get_sub_group_id();
    int sgN = get_local_size(1) / SUBGROUP_SIZE;
    int sgM = get_local_size(0);
    int sgid_N = sgid % sgN;
    int sgid_M = sgid / sgN;

    int BM = REG_M * sgM;
    int BN = get_local_size(1) * REG_N;

    int m_idx = get_group_id(0) * BM + sgid_M * REG_M;
    int n_idx = get_group_id(1) * BN + sgid_N * SUBGROUP_SIZE * REG_N;

    if (m_idx >= M || n_idx >= N ) {
        return;
    }

    if (m_idx + REG_M > M) {
        m_idx = M - REG_M;
    }

    if (n_idx + REG_N * SUBGROUP_SIZE > N) {
        n_idx = N - REG_N * SUBGROUP_SIZE;
    }

    int strideA = K * LORA_COUNT;
    int strideB = N0;

    __global ACCUMULATOR_TYPE* ptrA = output_a + m_idx * strideA;
    __global STATE_TYPE* ptrB = state_b_0 + n_idx;
    __global OUTPUT_TYPE* ptrC = output + m_idx * N + n_idx;

#if LORA_COUNT == 3
    strideB = N1_2;
    if (n_idx >= N0 + N1_2) {
        ptrB = state_b_2 + n_idx - N0 - N1_2;
        ptrA += K * 2;
    } else
#endif
    if (n_idx >= N0) {
        ptrB = state_b_1 + n_idx - N0;
        ptrA += K;
    }

    MAIN_MATMUL_CODE

    __global INPUT0_TYPE *main_ptr = main_input + m_idx * N + n_idx;

    ADD_AND_STORE_CODE
}
#endif
