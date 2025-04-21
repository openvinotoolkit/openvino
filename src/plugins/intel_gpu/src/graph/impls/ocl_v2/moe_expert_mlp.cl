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

    gemv_n2(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, false);
    gemv_n2(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, true);
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
    __global uchar* down_weight = (__global uchar*)info_ptr->weight[2];
    __global half* down_scale = (__global half*)info_ptr->scale[2];
    __global uchar* down_zp = (__global uchar*)info_ptr->zp[2];
    gemv_n(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, routing_weights[expert_no]);
}

#else
//__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_reduce)(const __global TYPE* x,                // [MAX_TOPK, HIDDEN_SIZE]
    __global TYPE* y) {                                    // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    float sum = 0;
    for (int i = 0; i < MAX_TOPK; i++) {
        sum += x[n];
        x += HIDDEN_SIZE;
    }
    y[n] = sum;
}
#endif
