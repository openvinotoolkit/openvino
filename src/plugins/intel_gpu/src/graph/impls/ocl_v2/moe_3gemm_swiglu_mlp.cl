
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define unroll_for __attribute__((opencl_unroll_hint)) for

// Fake group size for compatibility and computation performance balance
#define FAKE_GROUP_SIZE 128

#if GATE_UP_ENABLE
inline void gemv_n2x(const __global uchar* weight,
                    __global half* scales,
                    __global uchar* zps,
                    const __global half* x,
                    __global half* y, int N, int K,
                    half* x2,
                    float* xg_sum,
                    const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    unroll_for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        unroll_for (int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / GATE_UP_GROUP_SIZE) * N / 2;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            uchar z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*FAKE_GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*FAKE_GROUP_SIZE/2));

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
#else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk * FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk * FAKE_GROUP_SIZE / 2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk * FAKE_GROUP_SIZE / 2));

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
#endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= sum_all0 / (1 + exp(-sum_all0));
                y[n+1] *= sum_all1 / (1 + exp(-sum_all1));
            } else {
                y[n] = sum_all0;
                y[n+1] = sum_all1;
            }
        }
    }
}
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global int* expert_list,
    const __global uchar* gate_weight_addr,
    const __global uchar* gate_scale_addr,
    const __global uchar* gate_zp_addr,
    const __global uchar* up_weight_addr,
    const __global uchar* up_scale_addr,
    const __global uchar* up_zp_addr,
    __global MOE_DTYPE* x,                        // [1, HIDDEN_SIZE]
    __global MOE_DTYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;

    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2 / GATE_UP_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / GATE_UP_GROUP_SIZE;
    int expert_id = expert_list[expert_no];

    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)(gate_weight_addr + expert_id * expert_wei_size);
    __global half* gate_scale = (__global half*)(gate_scale_addr + expert_id * expert_scale_size);
    __global uchar* gate_zp = (__global uchar*)(gate_zp_addr + expert_id * expert_zp_size);

    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)(up_weight_addr + expert_id * expert_wei_size);
    __global half* up_scale = (__global half*)(up_scale_addr + expert_id * expert_scale_size);
    __global uchar* up_zp = (__global uchar*)(up_zp_addr + expert_id * expert_zp_size);

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE/32];

#if GATE_UP_GROUP_SIZE % FAKE_GROUP_SIZE != 0
    if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
        printf("GATE_UP_GROUP_SIZE(%d) must be divisible by FAKE_GROUP_SIZE(%d)", GATE_UP_GROUP_SIZE, FAKE_GROUP_SIZE);
    }
    return;
#endif

    //# interleaving x into x2
    int id_sg = get_sub_group_id();
    int num_sg = get_num_sub_groups();
    int id_local = get_sub_group_local_id();
    half * px = x + id_sg*FAKE_GROUP_SIZE;
    half * px2 = x2 + id_sg*FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < HIDDEN_SIZE/FAKE_GROUP_SIZE; i += num_sg, px += num_sg*FAKE_GROUP_SIZE, px2 += num_sg*FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    gemv_n2x(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gemv_n2x(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
}

#elif DOWN_ENABLE
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_down)(
    const __global int* expert_list,
    const __global uchar* down_weight_addr,
    const __global uchar* down_scale_addr,
    const __global uchar* down_zp_addr,
    const __global MOE_DTYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global MOE_DTYPE* routing_weights,                       // [MAX_TOPK]
    __global MOE_DTYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;

    const int expert_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;
    const int expert_scale_size = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2 / DOWN_GROUP_SIZE;
    const int expert_zp_size = INTERMEDIATE_SIZE * HIDDEN_SIZE / 2 / DOWN_GROUP_SIZE;
    int expert_id = expert_list[expert_no];

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* weight = (__global uchar*)(down_weight_addr + expert_id * expert_wei_size);
    __global half* scales = (__global half*)(down_scale_addr + expert_id * expert_scale_size);
    __global uchar* zps = (__global uchar*)(down_zp_addr + expert_id * expert_zp_size);

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE/32];

    //# interleaving x into x2
    __global half * px = x + id_sg*FAKE_GROUP_SIZE;
    __local half * px2 = x2 + id_sg*FAKE_GROUP_SIZE;
    unroll_for(int i = id_sg; i < INTERMEDIATE_SIZE/FAKE_GROUP_SIZE; i += num_sg, px += num_sg*FAKE_GROUP_SIZE, px2 += num_sg*FAKE_GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        unroll_for(int j = id_local; j < FAKE_GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + FAKE_GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    unroll_for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        unroll_for (int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
            int scale_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N;
            int zp_offset = gk * (FAKE_GROUP_SIZE / DOWN_GROUP_SIZE) * N / 2;
            half s0 = S[scale_offset];
            half s1 = S[scale_offset + 1];
            ushort z = Z[zp_offset];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*FAKE_GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*FAKE_GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*FAKE_GROUP_SIZE/2));

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
#else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk*FAKE_GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk*FAKE_GROUP_SIZE/2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk*FAKE_GROUP_SIZE/2));

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
#endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n+1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_reduce)(const __global MOE_DTYPE* x,           // [MAX_TOPK, HIDDEN_SIZE]
    __global MOE_DTYPE* y) {                               // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    half sum[MAX_TOPK] = {0};
    __attribute__((opencl_unroll_hint(MAX_TOPK)))
    for (int i = 0; i < MAX_TOPK; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + i*HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < MAX_TOPK; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((__global ushort*)(y + n), as_ushort(sum[0]));
}
#endif
