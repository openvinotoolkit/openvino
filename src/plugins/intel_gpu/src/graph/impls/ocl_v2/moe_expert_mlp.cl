// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

//#define MAX_TOPK 8
//#define HIDDEN_SIZE 2048
//#define INTERMEDIATE_SIZE 768
//#define GROUP_SIZE 128
//#define N_BLOCK_SIZE 64
//#define SUBGROUP_SIZE 32

typedef struct {
    __global void* weight[3];
    __global void* zp[3];
    __global void* scale[3];
} FUNC(expert_info);

#if GATE_UP_ENABLE
// x: [1, K]
// y: [1, N]
inline void gemv_n2(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, const bool silu) { // __local half x_cache[K]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    for (int n = n_start; n < n_end; n++) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        half sum_all = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half4 sum = 0;
            half s = S[0];
            ushort z = Z[0];
            half z_hf = convert_half((n & 1) ? (z >> 4) : (z & 0xf));
            // read 1 group(32*2 or 16*4 bytes)
            ushort bs;
            half b_even, b_odd;
#define ACC(idx, idx_sum)    \
            bs = b[idx]; b_even = convert_half(bs & 0xf); b_odd = convert_half(bs >> 4);    \
            sum[2 * idx_sum + 0] = fma(a[2 * idx + 0], b_even - z_hf, sum[2 * idx_sum + 0]);    \
            sum[2 * idx_sum + 1] = fma(a[2 * idx + 1], b_odd - z_hf,  sum[2 * idx_sum + 1]);

#if SUBGROUP_SIZE == 32
            half4 a = ((const __global half4*)(x + gk * GROUP_SIZE))[id_local];
            uchar2 b = ((const __global uchar2*)(B + gk * GROUP_SIZE / 2))[id_local];
            ACC(0, 0); ACC(1, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
#else
            half8 a = ((const __global half8*)(x + gk * GROUP_SIZE))[id_local];
            uchar4 b = ((const __global uchar4*)(B + gk * GROUP_SIZE / 2))[id_local];
            ACC(0, 0); ACC(1, 1); ACC(2, 0); ACC(3, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
#endif
#undef ACC
        }

        sum_all = sub_group_reduce_add(sum_all);
        if (id_local == 0) {
			if (silu) {
                y[n] *= sum_all / (1 + exp(-sum_all));
			} else {
                y[n] = sum_all;
			}
        }
    }
}

inline void gemv_n2x(const __global uchar* weight,
                    __global half* scales,
                    __global uchar* zps,
                    const __global half* x,
                    __global half* y, int N, int K,
                    half* x2,
                    float* xg_sum,
                    const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();
#if 1
    //# interleaving x into x2
    half * px = x + id_sg*GROUP_SIZE;
    half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < HIDDEN_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        for(int j = id_local; j < GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half s0 = S[0];
            half s1 = S[1];
            ushort z = Z[0];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

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
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

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

__global FUNC(expert_info) FUNC(g_info_ptrs)[] = {
WEIGHT_POINTERS
};

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global int* expert_list,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;
    __global FUNC(expert_info)* info_ptr = &(FUNC(g_info_ptrs)[expert_list[expert_no]]);
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)info_ptr->weight[1];
    __global half* up_scale = (__global half*)info_ptr->scale[1];
    __global uchar* up_zp = (__global uchar*)info_ptr->zp[1];
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)info_ptr->weight[0];
    __global half* gate_scale = (__global half*)info_ptr->scale[0];
    __global uchar* gate_zp = (__global uchar*)info_ptr->zp[0];

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE/32];
    gemv_n2x(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gemv_n2x(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);

    //gemv_n2(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, false);
    //gemv_n2(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, true);
}

#elif DOWN_ENABLE
// x: [1, K]
// y: [1, N]
inline void gemv_n(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, half routing_weights) { // __local half x_cache[K]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    for (int n = n_start; n < n_end; n++) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        half sum_all = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half4 sum = 0;
            half s = S[0];
            ushort z = Z[0];
            half z_hf = convert_half((n & 1) ? (z >> 4) : (z & 0xf));
            // read 1 group(32*2 or 16*4 bytes)
            ushort bs;
            half b_even, b_odd;
#define ACC(idx, idx_sum)    \
            bs = b[idx]; b_even = convert_half(bs & 0xf); b_odd = convert_half(bs >> 4);    \
            sum[2 * idx_sum + 0] = fma(a[2 * idx + 0], b_even - z_hf, sum[2 * idx_sum + 0]);    \
            sum[2 * idx_sum + 1] = fma(a[2 * idx + 1], b_odd - z_hf,  sum[2 * idx_sum + 1]);

#if SUBGROUP_SIZE == 32
            half4 a = ((const __global half4*)(x + gk * GROUP_SIZE))[id_local];
            uchar2 b = ((const __global uchar2*)(B + gk * GROUP_SIZE / 2))[id_local];
            ACC(0, 0); ACC(1, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
#else
            half8 a = ((const __global half8*)(x + gk * GROUP_SIZE))[id_local];
            uchar4 b = ((const __global uchar4*)(B + gk * GROUP_SIZE / 2))[id_local];
            ACC(0, 0); ACC(1, 1); ACC(2, 0); ACC(3, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
#endif
#undef ACC
        }

        sum_all = sub_group_reduce_add(sum_all);

        if (id_local == 0)
            y[n] = sum_all * routing_weights;
    }
}

__global FUNC(expert_info) FUNC(g_info_ptrs)[] = {
WEIGHT_POINTERS
};

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_down)(
    const __global int* expert_list,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;

    __global FUNC(expert_info)* info_ptr = &(FUNC(g_info_ptrs)[expert_list[expert_no]]);
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* weight = (__global uchar*)info_ptr->weight[2];
    __global half* scales = (__global half*)info_ptr->scale[2];
    __global uchar* zps = (__global uchar*)info_ptr->zp[2];
    
    //gemv_n(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, routing_weights[expert_no]);


    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE/32];
#if 1
    //# interleaving x into x2
    __global half * px = x + id_sg*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < INTERMEDIATE_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        for(int j = id_local; j < GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    //# global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half s0 = S[0];
            half s1 = S[1];
            ushort z = Z[0];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

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
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

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
KERNEL (mlp_reduce)(const __global TYPE* x,                // [MAX_TOPK, HIDDEN_SIZE]
    __global TYPE* y) {                                    // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    half sum[MAX_TOPK] = {0};
    __attribute__((opencl_unroll_hint(MAX_TOPK)))
    for (int i = 0; i < MAX_TOPK; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + i*HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < MAX_TOPK; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((const __global ushort*)(y + n), as_ushort(sum[0]));
}
#endif
