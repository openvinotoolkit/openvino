
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define unroll_for __attribute__((opencl_unroll_hint)) for

// Fake group size for compatibility and computation performance balance
#define FAKE_GROUP_SIZE 128

#if GATE_UP_ENABLE
inline void gate_up_gemv_n2x_u4(const __global uchar* weight,
                                __global half* scales,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 2;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            uchar z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#    if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s2 = fma(a.s2, (convert_half(b.s2 & 0x0F)), 0);
            sum0.s3 = fma(a.s3, (convert_half(b.s3 & 0x0F)), 0);

            sum0.s0 = fma(a.s4, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s5, (convert_half(b.s1 >> 4)), sum0.s1);
            sum0.s2 = fma(a.s6, (convert_half(b.s2 >> 4)), sum0.s2);
            sum0.s3 = fma(a.s7, (convert_half(b.s3 >> 4)), sum0.s3);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s2 = fma(a.s2, (convert_half(b2.s2 & 0x0F)), 0);
            sum1.s3 = fma(a.s3, (convert_half(b2.s3 & 0x0F)), 0);

            sum1.s0 = fma(a.s4, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s5, (convert_half(b2.s1 >> 4)), sum1.s1);
            sum1.s2 = fma(a.s6, (convert_half(b2.s2 >> 4)), sum1.s2);
            sum1.s3 = fma(a.s7, (convert_half(b2.s3 >> 4)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= sum_all0 / (1 + exp(-sum_all0));
                y[n + 1] *= sum_all1 / (1 + exp(-sum_all1));
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_u8(const __global uchar* weight,
                                __global half* scales,
                                __global uchar* zps,
                                __global half* y,
                                int N,
                                int K,
                                half* x2,
                                float* xg_sum,
                                const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
        __global uchar* Z = zps + n;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);

#    if SUBGROUP_SIZE == 32
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(convert_half(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(convert_half(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(convert_half(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(convert_half(b2.s3)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(convert_half(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(convert_half(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(convert_half(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(convert_half(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(convert_half(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(convert_half(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(convert_half(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(convert_half(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(convert_half(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(convert_half(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(convert_half(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(convert_half(b2.s7)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= sum_all0 / (1 + exp(-sum_all0));
                y[n + 1] *= sum_all1 / (1 + exp(-sum_all1));
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

inline void gate_up_gemv_n2x_f16(const __global half* weight, __global half* y, int N, int K, half* x2, const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
#    if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2= as_half8(intel_sub_group_block_read_us8((const __global ushort*)(B + K + gk * FAKE_GROUP_SIZE)));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3] ;
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= sum_all0 / (1 + exp(-sum_all0));
                y[n + 1] *= sum_all1 / (1 + exp(-sum_all1));
            } else {
                y[n] = sum_all0;
                y[n + 1] = sum_all1;
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_gate_up)(const __global int* expert_list,
                                                                              const __global MOE_WEI_DT* gate_weight_addr,
                                                                              const __global MOE_SCALE_DT* gate_scale_addr,
                                                                              const __global MOE_ZP_DT* gate_zp_addr,
                                                                              const __global MOE_WEI_DT* up_weight_addr,
                                                                              const __global MOE_SCALE_DT* up_scale_addr,
                                                                              const __global MOE_ZP_DT* up_zp_addr,
                                                                              __global MOE_DTYPE* x,    // [1, HIDDEN_SIZE]
                                                                              __global MOE_DTYPE* y) {  // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;

#    if WEIGHT_COMPRESSEION_DT == 0
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / GATE_UP_GROUP_SIZE;
#    else
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / GATE_UP_GROUP_SIZE;
#    endif

    int expert_id = expert_list[expert_no];

    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* gate_weight = (__global MOE_WEI_DT*)(gate_weight_addr + expert_id * expert_wei_size);
    __global MOE_SCALE_DT* gate_scale = (__global MOE_SCALE_DT*)(gate_scale_addr + expert_id * expert_scale_size);
    __global MOE_ZP_DT* gate_zp = (__global MOE_ZP_DT*)(gate_zp_addr + expert_id * expert_zp_size);

    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global MOE_WEI_DT* up_weight = (__global MOE_WEI_DT*)(up_weight_addr + expert_id * expert_wei_size);
    __global MOE_SCALE_DT* up_scale = (__global MOE_SCALE_DT*)(up_scale_addr + expert_id * expert_scale_size);
    __global MOE_ZP_DT* up_zp = (__global MOE_ZP_DT*)(up_zp_addr + expert_id * expert_zp_size);

#    if GATE_UP_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("GATE_UP_GROUP_SIZE(%d) must be divisible by FAKE_GROUP_SIZE(%d)", GATE_UP_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#    endif

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE / FAKE_GROUP_SIZE];

#    if WEIGHT_COMPRESSEION_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
            x_group_sum += value;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#    if WEIGHT_COMPRESSEION_DT == 0
    gate_up_gemv_n2x_u4(up_weight, up_scale, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gate_up_gemv_n2x_u4(gate_weight, gate_scale, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif WEIGHT_COMPRESSEION_DT == 1
    gate_up_gemv_n2x_u8(up_weight, up_scale, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gate_up_gemv_n2x_u8(gate_weight, gate_scale, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
#    elif WEIGHT_COMPRESSEION_DT == 2
    gate_up_gemv_n2x_f16(up_weight, up_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
    gate_up_gemv_n2x_f16(gate_weight, gate_zp, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
#    endif
}

#elif DOWN_ENABLE

inline void down_gemv_n2x_u4(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             __global MOE_DTYPE* routing_weights,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 2;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            ushort z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#    if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K / 2) + gk * FAKE_GROUP_SIZE / 2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s2 = fma(a.s2, (convert_half(b.s2 & 0x0F)), 0);
            sum0.s3 = fma(a.s3, (convert_half(b.s3 & 0x0F)), 0);

            sum0.s0 = fma(a.s4, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s5, (convert_half(b.s1 >> 4)), sum0.s1);
            sum0.s2 = fma(a.s6, (convert_half(b.s2 >> 4)), sum0.s2);
            sum0.s3 = fma(a.s7, (convert_half(b.s3 >> 4)), sum0.s3);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s2 = fma(a.s2, (convert_half(b2.s2 & 0x0F)), 0);
            sum1.s3 = fma(a.s3, (convert_half(b2.s3 & 0x0F)), 0);

            sum1.s0 = fma(a.s4, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s5, (convert_half(b2.s1 >> 4)), sum1.s1);
            sum1.s2 = fma(a.s6, (convert_half(b2.s2 >> 4)), sum1.s2);
            sum1.s3 = fma(a.s7, (convert_half(b2.s3 >> 4)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n + 1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

inline void down_gemv_n2x_u8(const __global uchar* weight,
                             __global half* scales,
                             __global uchar* zps,
                             __global MOE_DTYPE* routing_weights,
                             __global half* y,
                             int N,
                             int K,
                             half* x2,
                             float* xg_sum) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global uchar* B = weight + n * K;
        __global half* S = scales + n;
        __global uchar* Z = zps + n;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            half z0 = convert_half(Z[zp_offset]);
            half z1 = convert_half(Z[zp_offset + 1]);

#    if SUBGROUP_SIZE == 32
            float2 sum0;
            float2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)B + K + gk * FAKE_GROUP_SIZE);

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s0 = fma((float)a.s2, (float)(convert_half(b.s2)), sum0.s0);
            sum0.s1 = fma((float)a.s3, (float)(convert_half(b.s3)), sum0.s1);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s0 = fma((float)a.s2, (float)(convert_half(b2.s2)), sum1.s0);
            sum1.s1 = fma((float)a.s3, (float)(convert_half(b2.s3)), sum1.s1);

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z1) * s1;
#    else
            float4 sum0;
            float4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar8 b = intel_sub_group_block_read_uc8((const __global uchar*)B + gk * FAKE_GROUP_SIZE);
            uchar8 b2 = intel_sub_group_block_read_uc8((const __global uchar*)(B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma((float)a.s0, (float)(convert_half(b.s0)), 0.0f);
            sum0.s1 = fma((float)a.s1, (float)(convert_half(b.s1)), 0.0f);
            sum0.s2 = fma((float)a.s2, (float)(convert_half(b.s2)), 0.0f);
            sum0.s3 = fma((float)a.s3, (float)(convert_half(b.s3)), 0.0f);

            sum0.s0 = fma((float)a.s4, (float)(convert_half(b.s4)), sum0.s0);
            sum0.s1 = fma((float)a.s5, (float)(convert_half(b.s5)), sum0.s1);
            sum0.s2 = fma((float)a.s6, (float)(convert_half(b.s6)), sum0.s2);
            sum0.s3 = fma((float)a.s7, (float)(convert_half(b.s7)), sum0.s3);

            sum1.s0 = fma((float)a.s0, (float)(convert_half(b2.s0)), 0.0f);
            sum1.s1 = fma((float)a.s1, (float)(convert_half(b2.s1)), 0.0f);
            sum1.s2 = fma((float)a.s2, (float)(convert_half(b2.s2)), 0.0f);
            sum1.s3 = fma((float)a.s3, (float)(convert_half(b2.s3)), 0.0f);

            sum1.s0 = fma((float)a.s4, (float)(convert_half(b2.s4)), sum1.s0);
            sum1.s1 = fma((float)a.s5, (float)(convert_half(b2.s5)), sum1.s1);
            sum1.s2 = fma((float)a.s6, (float)(convert_half(b2.s6)), sum1.s2);
            sum1.s3 = fma((float)a.s7, (float)(convert_half(b2.s7)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z1) * s1;
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n + 1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

inline void down_gemv_n2x_f16(const __global half* weight, __global MOE_DTYPE* routing_weights, __global half* y, int N, int K, half* x2) {
    int id_local = get_sub_group_local_id();
    int expert_no = get_global_id(0);
    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for(int n = n_start; n < n_end; n += 2) {
        const __global half* B = weight + n * K;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for(int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {

#    if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __global ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half4 b = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half4 b2 = as_half4(intel_sub_group_block_read_us4((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s0 = fma(a.s2, b.s2, sum0.s0);
            sum0.s1 = fma(a.s3, b.s3, sum0.s1);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s0 = fma(a.s2, b2.s2, sum1.s0);
            sum1.s1 = fma(a.s3, b2.s3, sum1.s1);

            sum_all0 += sum0[0] + sum0[1];
            sum_all1 += sum1[0] + sum1[1];
#    else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            half8 b = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + gk * FAKE_GROUP_SIZE));
            half8 b2 = as_half8(intel_sub_group_block_read_us8((const __global ushort*)B + K + gk * FAKE_GROUP_SIZE));

            sum0.s0 = fma(a.s0, b.s0, 0);
            sum0.s1 = fma(a.s1, b.s1, 0);
            sum0.s2 = fma(a.s2, b.s2, 0);
            sum0.s3 = fma(a.s3, b.s3, 0);

            sum0.s0 = fma(a.s4, b.s4, sum0.s0);
            sum0.s1 = fma(a.s5, b.s5, sum0.s1);
            sum0.s2 = fma(a.s6, b.s6, sum0.s2);
            sum0.s3 = fma(a.s7, b.s7, sum0.s3);

            sum1.s0 = fma(a.s0, b2.s0, 0);
            sum1.s1 = fma(a.s1, b2.s1, 0);
            sum1.s2 = fma(a.s2, b2.s2, 0);
            sum1.s3 = fma(a.s3, b2.s3, 0);

            sum1.s0 = fma(a.s4, b2.s4, sum1.s0);
            sum1.s1 = fma(a.s5, b2.s5, sum1.s1);
            sum1.s2 = fma(a.s6, b2.s6, sum1.s2);
            sum1.s3 = fma(a.s7, b2.s7, sum1.s3);

            sum_all0 += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            sum_all1 += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#    endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n + 1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_down)(const __global int* expert_list,
                                                                           const __global MOE_WEI_DT* down_weight_addr,
                                                                           const __global MOE_SCALE_DT* down_scale_addr,
                                                                           const __global MOE_ZP_DT* down_zp_addr,
                                                                           const __global MOE_DTYPE* x,          // [MAX_TOPK, INTERMEDIATE_SIZE]
                                                                           __global MOE_DTYPE* routing_weights,  // [MAX_TOPK]
                                                                           __global MOE_DTYPE* y) {              // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;

#    if WEIGHT_COMPRESSEION_DT == 0
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / DOWN_GROUP_SIZE;
#    else
    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / DOWN_GROUP_SIZE;
#    endif
    int expert_id = expert_list[expert_no];

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global MOE_WEI_DT* weight = (__global MOE_WEI_DT*)(down_weight_addr + expert_id * expert_wei_size);
    __global MOE_SCALE_DT* scales = (__global MOE_SCALE_DT*)(down_scale_addr + expert_id * expert_scale_size);
    __global MOE_ZP_DT* zps = (__global MOE_ZP_DT*)(down_zp_addr + expert_id * expert_zp_size);

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE / FAKE_GROUP_SIZE];

#    if WEIGHT_COMPRESSEION_DT == 0
    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SUBGROUP_SIZE) {
            half even = px[2 * j + 0];
            half odd = px[2 * j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE / 2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    else
    //# load x into slm
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half* px = x + id_sg * FAKE_GROUP_SIZE;
    half* px2 = x2 + id_sg * FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE / FAKE_GROUP_SIZE; i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE; j += SUBGROUP_SIZE) {
            half value = px[j];
            px2[j] = value;
            x_group_sum += value;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#    endif

    barrier(CLK_LOCAL_MEM_FENCE);

#    if WEIGHT_COMPRESSEION_DT == 0
    down_gemv_n2x_u4(weight, scales, zps, routing_weights, y, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 1
    down_gemv_n2x_u8(weight, scales, zps, routing_weights, y, N, K, x2, xg_sum);
#    elif WEIGHT_COMPRESSEION_DT == 2
    down_gemv_n2x_f16(weight, routing_weights, y, N, K, x2);
#    endif
}

#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(mlp_reduce)(const __global MOE_DTYPE* x,  // [MAX_TOPK, HIDDEN_SIZE]
                                                                             __global MOE_DTYPE* y) {      // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    half sum[MAX_TOPK] = {0};
    __attribute__((opencl_unroll_hint(MAX_TOPK))) for (int i = 0; i < MAX_TOPK; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + i * HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < MAX_TOPK; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((__global ushort*)(y + n), as_ushort(sum[0]));
}
#endif
