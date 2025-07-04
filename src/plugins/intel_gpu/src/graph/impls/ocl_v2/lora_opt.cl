// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef SECOND_TOKEN_A
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(second_token_a)(OPTIONAL_SHAPE_INFO_ARG
                       __global INPUT1_TYPE* lora_input,
                       __global STATE_TYPE* state_a,
                       __global ACCUMULATOR_TYPE* output_a)
{
    int gid = get_group_id(0);
    int sgid = get_sub_group_id();

    // For 2nd token, rank is small. sg in one wg would be divided by 2 dimensions to increase threads number in wg.
    int sgN = LORA_RANK / SUBGROUP_SIZE;
    int sgid_k = sgid / sgN;
    int n_idx = sgid % sgN * SUBGROUP_SIZE;
    int lid = get_sub_group_local_id();
    // How many K is accumulated in the WG.
    int bk_wg = GEMMA_SGK * GEMMA_SG_BK;
    int k_start_wg = gid * bk_wg;
    int wg_k_len = (k_start_wg + bk_wg) > K ? (K - k_start_wg) : bk_wg;
    int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

    // Store each sg accumulation result into SLM. Will reduce sg result into wg result.
    __local ACCUMULATOR_TYPE fma_buff[MAX_GEMMA_SGK * MAX_LORA_RANK];
    __local ACCUMULATOR_TYPE *sg_fma_buff = fma_buff + sgid_k * MAX_LORA_RANK;

    // Put all need input activation into SLM. 'sgN' sgs would share same input.
    __local ACCUMULATOR_TYPE local_input[MAX_GEMMA_SGK * GEMMA_SG_BK];

    // sg could diverge here. Not all the sgs can satisfy 'offset < k_len'.
    int local_sz = get_num_sub_groups() * SUBGROUP_SIZE;
    for (int offset = sgid * SUBGROUP_SIZE; offset < wg_k_len; offset += local_sz) {
        __global INPUT1_TYPE *input_ptr = lora_input + k_start_wg + offset;
#if INPUT1_TYPE_SIZE == 4
        ACCUMULATOR_TYPE copy_val = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)input_ptr));
#else
        ACCUMULATOR_TYPE copy_val = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
#endif
        local_input[offset + lid] = copy_val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int k_offset = sgid_k * GEMMA_SG_BK;
    int k_idx = k_start_wg + k_offset;

    // The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
    if (sgid_k * GEMMA_SG_BK < wg_k_len) {
        int klen_sg = (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
        __global STATE_TYPE *B_ptr = state_a + k_idx * LORA_RANK + n_idx;
        __local ACCUMULATOR_TYPE *A_ptr = local_input + k_offset;
        ACCUMULATOR_TYPE sum = 0.f;

        for (int kk = 0; kk < klen_sg; kk += SUBGROUP_SIZE) {
#if ACCUMULATOR_TYPE_SIZE == 4
            uint input = intel_sub_group_block_read((const __local uint*)(A_ptr + kk));
#else
            ushort input = intel_sub_group_block_read_us((const __local ushort*)(A_ptr + kk));
#endif
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SUBGROUP_SIZE; j++) {
#if ACCUMULATOR_TYPE_SIZE == 4
                ACCUMULATOR_TYPE bb = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(B_ptr)));
#else
                ACCUMULATOR_TYPE bb = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
#endif
                ACCUMULATOR_TYPE aa = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                B_ptr += LORA_RANK;
            }
        }
        *(sg_fma_buff + n_idx + lid) = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global ACCUMULATOR_TYPE *C_ptr = output_a + LORA_RANK * gid;

    // Only need sg on N dimenion to update data.
    if (sgid_k != 0) {
        return;
    }

    ACCUMULATOR_TYPE sum = 0.f;
    if (sgK == GEMMA_SGK) {
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < GEMMA_SGK; i++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
#else
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
#endif
        }
    } else {
        // Can't unroll, tail handling.
        for (int i = 0; i < sgK; i++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
#else
            sum += AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
#endif
        }
    }
#if ACCUMULATOR_TYPE_SIZE == 4
    intel_sub_group_block_write((__global uint*)(C_ptr + n_idx), as_int(sum));
#else
    intel_sub_group_block_write_us((__global ushort*)(C_ptr + n_idx), as_short(sum));
#endif
}
#endif

#ifdef SECOND_TOKEN_B
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(second_token_b)(OPTIONAL_SHAPE_INFO_ARG
                       __global INPUT0_TYPE* main_input,
                       __global ACCUMULATOR_TYPE* output_a,
                       __global STATE_TYPE* state_alpha,
                       __global STATE_TYPE* state_b,
                       __global OUTPUT_TYPE* output)
{
    int wg_id = get_group_id(0);
    int sg_id = get_sub_group_id();
    int sg_num = get_num_sub_groups();
    int n_idx = (wg_id * sg_num + sg_id) * SUBGROUP_SIZE;
    int id_sg_local = get_sub_group_local_id();

    __local ACCUMULATOR_TYPE reduce[MAX_LORA_RANK];

    __global STATE_TYPE *B_ptr = state_b + n_idx;

    // 1. Reduce
    // EACH WG would reduce input activation and save into local memory `reduce[MAX_LORA_RANK]`.
    int local_sz = get_local_size(0);
    for (int offset = sg_id * SUBGROUP_SIZE; offset < LORA_RANK; offset += local_sz) {
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
            A_ptr += LORA_RANK;
        }
        reduce[offset + id_sg_local] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (n_idx >= N) {
        return;
    }

    //2. GEMMB
    __global OUTPUT_TYPE *C_ptr = output + n_idx;

    ACCUMULATOR_TYPE sum = 0;
    __attribute__((opencl_unroll_hint))
    for (int kk = 0; kk < LORA_RANK; kk += SUBGROUP_SIZE) {
#if ACCUMULATOR_TYPE_SIZE == 4
        ACCUMULATOR_TYPE scale = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(state_alpha + kk)));
        ACCUMULATOR_TYPE input = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __local uint*)(reduce + kk)));
#else
        ACCUMULATOR_TYPE scale = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(state_alpha + kk)));
        ACCUMULATOR_TYPE input = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __local ushort*)(reduce + kk)));
#endif
        input *= scale / TO_ACCUMULATOR_TYPE(LORA_RANK);

        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < SUBGROUP_SIZE; j++) {
#if ACCUMULATOR_TYPE_SIZE == 4
            ACCUMULATOR_TYPE aa = AS_ACCUMULATOR_TYPE(sub_group_broadcast(as_uint(input), j));
            ACCUMULATOR_TYPE bb = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)B_ptr));
#else
            ACCUMULATOR_TYPE aa = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(as_ushort(input), j));
            ACCUMULATOR_TYPE bb = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
#endif
            sum = fma(aa, bb, sum);
            B_ptr += N;
        }
    }
    __global INPUT0_TYPE *main_input_ptr = main_input + n_idx;
#if INPUT0_TYPE_SIZE == 4
    INPUT0_TYPE m_input = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)main_input_ptr));
    intel_sub_group_block_write((__global uint*)C_ptr, as_int(sum + m_input));
#else
    INPUT0_TYPE m_input = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));
    intel_sub_group_block_write_us((__global ushort*)C_ptr, as_short(sum + m_input));
#endif
}
#endif

#ifdef FIRST_TOKEN_A
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(first_token_a)(OPTIONAL_SHAPE_INFO_ARG
                      __global INPUT1_TYPE* lora_input,
                      __global STATE_TYPE* state_a,
                      __global STATE_TYPE* state_alpha,
                      __global ACCUMULATOR_TYPE* output_a)
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

    if (m_idx >= M || n_idx >= LORA_RANK) {
        return;
    }

    if (m_idx + REG_M > M) {
        m_idx = M - REG_M;
    }

    if (n_idx + REG_N * SUBGROUP_SIZE > LORA_RANK) {
        n_idx = LORA_RANK - REG_N * SUBGROUP_SIZE;
    }

    __global INPUT1_TYPE* ptrA = lora_input + m_idx * K;
    __global STATE_TYPE* ptrB = state_a + n_idx;
    __global ACCUMULATOR_TYPE* ptrC = output_a + m_idx * LORA_RANK + n_idx;

    MAIN_MATMUL_CODE

    __global STATE_TYPE *alpha_ptr = state_alpha + n_idx;

    MULTIPLY_AND_STORE_CODE
}
#endif


#ifdef FIRST_TOKEN_B
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(first_token_b)(OPTIONAL_SHAPE_INFO_ARG
                      __global INPUT0_TYPE* main_input,
                      __global ACCUMULATOR_TYPE* output_a,
                      __global STATE_TYPE* state_b,
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

    if (m_idx >= M || n_idx >= N) {
        return;
    }

    if (m_idx + REG_M > M) {
        m_idx = M - REG_M;
    }

    if (n_idx + REG_N * SUBGROUP_SIZE > N) {
        n_idx = N - REG_N * SUBGROUP_SIZE;
    }

    __global ACCUMULATOR_TYPE* ptrA = output_a + m_idx * LORA_RANK;
    __global STATE_TYPE* ptrB = state_b + n_idx;
    __global OUTPUT_TYPE* ptrC = output + m_idx * N + n_idx;

    MAIN_MATMUL_CODE

    __global INPUT0_TYPE *main_ptr = main_input + m_idx * N + n_idx;

    ADD_AND_STORE_CODE
}
#endif
