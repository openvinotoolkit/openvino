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
    int routing_offset;
    int pad;
} FUNC(expert_info);

#if GATE_UP_ENABLE
#if 0
//__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global FUNC(expert_info)* info_ptrs,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    int expert_no = get_global_id(0);
    int n = get_global_id(1);
    y += expert_no * INTERMEDIATE_SIZE;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)info_ptr->weight[1] + n / 2;
    __global half* up_scale = (__global half*)info_ptr->scale[1] + n;
    __global uchar* up_zp = (__global uchar*)info_ptr->zp[1] + n / 2;
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)info_ptr->weight[0] + n / 2;
    __global half* gate_scale = (__global half*)info_ptr->scale[0] + n;
    __global uchar* gate_zp = (__global uchar*)info_ptr->zp[0] + n / 2;
    float up_sum = 0, gate_sum = 0;
    for (int k = 0; k < HIDDEN_SIZE; k += GROUP_SIZE) {
        TYPE up_scale_v = up_scale[k / GROUP_SIZE * INTERMEDIATE_SIZE];
        uchar up_zp_org = up_zp[k / GROUP_SIZE * INTERMEDIATE_SIZE / 2];
        TYPE up_zp_v = convert_half((n & 1) ? (up_zp_org >> 4) : (up_zp_org & 0xf));
        TYPE gate_scale_v = gate_scale[k / GROUP_SIZE * INTERMEDIATE_SIZE];
        uchar gate_zp_org = gate_zp[k / GROUP_SIZE * INTERMEDIATE_SIZE / 2];
        TYPE gate_zp_v = convert_half((n & 1) ? (gate_zp_org >> 4) : (gate_zp_org & 0xf));
        float up_sum_group = 0, gate_sum_group = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            uchar up_weight_org = up_weight[(k + i) * INTERMEDIATE_SIZE / 2];
            TYPE up_w_v = convert_half((n & 1) ? (up_weight_org >> 4) : (up_weight_org & 0xf));
            up_sum_group += (up_w_v - up_zp_v) * x[k + i];
            uchar gate_weight_org = gate_weight[(k + i) * INTERMEDIATE_SIZE / 2];
            TYPE gate_w_v = convert_half((n & 1) ? (gate_weight_org >> 4) : (gate_weight_org & 0xf));
            gate_sum_group += (gate_w_v - gate_zp_v) * x[k + i];
        }
        up_sum += up_sum_group * up_scale_v;
        gate_sum += gate_sum_group * gate_scale_v;
    }

    // silu
    gate_sum *= 1 / (1 + exp(-gate_sum));
    // up*silu(gate)
    y[n] = up_sum * gate_sum;
}
#else
inline void splitter2(const int n, const int team, const int tid, int* n_start, int* n_end) {
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
    } else {
        int n1 = (n + team - 1) / team;
        int n2 = n1 - 1;
        int T1 = n - n2 * team;
        *n_end = tid < T1 ? n1 : n2;
        *n_start = tid <= T1 ? tid * n1 : T1 * n1 + (tid - T1) * n2;
    }
    *n_end += *n_start;
}

// x: [1, K]
// y: [1, N]
// all_sum: [N, sg_num]
inline void gemv_64n2(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, __local float* all_sum, const bool silu) { // __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM]
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);
    int ithr = get_local_id(2);     // 0~7
    int nthr = get_local_size(2);   // 8

    int K_groups = K / GROUP_SIZE;
    int gk0, gk1;
    splitter2(K_groups, nthr, ithr, &gk0, &gk1);

    half2 sum_all = 0;
    for(int gk = gk0; gk < gk1; gk++) {
        const int k = gk * GROUP_SIZE;
        const __global half* A = x + k;
        const __global uchar* B = weight + k * N / 2;

        half8 sum = 0;
        half2 scale = as_half2(((const __global ushort2*)(scales + gk * N))[0]);
        uchar zp = zps[gk * N / 2];
        char zp_even = (zp & 0xf);
        char zp_odd = (char)(zp >> 4);
        __attribute__((opencl_unroll_hint(4)))
        for(int g = 0; g < GROUP_SIZE; g += SUBGROUP_SIZE, B += SUBGROUP_SIZE * N / 2) {
            // read 32/2 elememts of A
            ushort vAs = as_short(intel_sub_group_block_read_us((const __global ushort*)(A + g)));
            // read 32 int4
            uchar b;
            half i4_even, i4_odd;
#define ACC(B_row, sum_idx) \
            b = B[B_row * N / 2];   \
            i4_even = convert_half((b & 0xf) - zp_even);  \
            i4_odd = convert_half((char)(b >> 4) - zp_odd);  \
            sum[sum_idx] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_even, sum[sum_idx]);    \
            sum[sum_idx + 1] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_odd, sum[sum_idx + 1]);
            ACC(0, 0); ACC(1, 2); ACC(2, 4);  ACC(3, 6); ACC(4, 0); ACC(5, 2); ACC(6, 4); ACC(7, 6);
            ACC(8, 0); ACC(9, 2); ACC(10, 4); ACC(11, 6); ACC(12, 0); ACC(13, 2); ACC(14, 4); ACC(15, 6);
#if SUBGROUP_SIZE == 32
            ACC(16, 0); ACC(17, 2); ACC(18, 4); ACC(19, 6); ACC(20, 0); ACC(21, 2); ACC(22, 4); ACC(23, 6);
            ACC(24, 0); ACC(25, 2); ACC(26, 4); ACC(27, 6); ACC(28, 0); ACC(29, 2); ACC(30, 4); ACC(31, 6);
#endif
#undef ACC
        }

        // scales applied once
        sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale[0];
        sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale[1];
    }
    all_sum[id_local * 2 * SUBGROUP_NUM + id_sg] = sum_all[0];
    all_sum[(id_local * 2 + 1) * SUBGROUP_NUM + id_sg] = sum_all[1];

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint()))
    for (int i = 0; i < SUBGROUP_SIZE * 2; i += SUBGROUP_NUM) {
        float sum_all = 0;
        if (id_local < SUBGROUP_NUM) {
            sum_all = all_sum[(i + id_sg) * SUBGROUP_NUM + id_local];
        }
        sum_all = sub_group_reduce_add(sum_all);
        if (id_local == 0) {
			if (silu) {
                y[i + id_sg] *= sum_all / (1 + exp(-sum_all));
			} else {
                y[i + id_sg] = sum_all;
			}
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global FUNC(expert_info)* info_ptrs,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    int n = get_global_id(1) * 2;
    y += expert_no * INTERMEDIATE_SIZE + n;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)info_ptr->weight[1] + n / 2;
    __global half* up_scale = (__global half*)info_ptr->scale[1] + n;
    __global uchar* up_zp = (__global uchar*)info_ptr->zp[1] + n / 2;
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)info_ptr->weight[0] + n / 2;
    __global half* gate_scale = (__global half*)info_ptr->scale[0] + n;
    __global uchar* gate_zp = (__global uchar*)info_ptr->zp[0] + n / 2;

    __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM];
    gemv_64n2(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, all_sum, false);
    gemv_64n2(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, all_sum, true);
}

#endif

#elif DOWN_ENABLE
inline void splitter(const int n, const int team, const int tid, int* n_start, int* n_end) {
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
    } else {
        int n1 = (n + team - 1) / team;
        int n2 = n1 - 1;
        int T1 = n - n2 * team;
        *n_end = tid < T1 ? n1 : n2;
        *n_start = tid <= T1 ? tid * n1 : T1 * n1 + (tid - T1) * n2;
    }
    *n_end += *n_start;
}

// x: [1, K]
// y: [1, N]
// all_sum: [N, sg_num]
inline void gemv_64n(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, __local float* all_sum, half routing_weights) { // __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM]
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);
    int ithr = get_local_id(2);     // 0~7
    int nthr = get_local_size(2);   // 8

    int K_groups = K / GROUP_SIZE;
    int gk0, gk1;
    splitter(K_groups, nthr, ithr, &gk0, &gk1);

    half2 sum_all = 0;
    for(int gk = gk0; gk < gk1; gk++) {
        const int k = gk * GROUP_SIZE;
        const __global half* A = x + k;
        const __global uchar* B = weight + k * N / 2;

        half8 sum = 0;
        half2 scale = as_half2(((const __global ushort2*)(scales + gk * N))[0]);
        uchar zp = zps[gk * N / 2];
        char zp_even = (zp & 0xf);
        char zp_odd = (char)(zp >> 4);
        __attribute__((opencl_unroll_hint(4)))
        for(int g = 0; g < GROUP_SIZE; g += SUBGROUP_SIZE, B += SUBGROUP_SIZE * N / 2) {
            // read 32/2 elememts of A
            ushort vAs = as_short(intel_sub_group_block_read_us((const __global ushort*)(A + g)));
            // read 32 int4
            uchar b;
            half i4_even, i4_odd;
#define ACC(B_row, sum_idx) \
            b = B[B_row * N / 2];   \
            i4_even = convert_half((b & 0xf) - zp_even);  \
            i4_odd = convert_half((char)(b >> 4) - zp_odd);  \
            sum[sum_idx] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_even, sum[sum_idx]);    \
            sum[sum_idx + 1] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_odd, sum[sum_idx + 1]);
            ACC(0, 0); ACC(1, 2); ACC(2, 4);  ACC(3, 6); ACC(4, 0); ACC(5, 2); ACC(6, 4); ACC(7, 6);
            ACC(8, 0); ACC(9, 2); ACC(10, 4); ACC(11, 6); ACC(12, 0); ACC(13, 2); ACC(14, 4); ACC(15, 6);
#if SUBGROUP_SIZE == 32
            ACC(16, 0); ACC(17, 2); ACC(18, 4); ACC(19, 6); ACC(20, 0); ACC(21, 2); ACC(22, 4); ACC(23, 6);
            ACC(24, 0); ACC(25, 2); ACC(26, 4); ACC(27, 6); ACC(28, 0); ACC(29, 2); ACC(30, 4); ACC(31, 6);
#endif
#undef ACC
        }

        // scales applied once
        sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale[0];
        sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale[1];
    }
    all_sum[id_local * 2 * SUBGROUP_NUM + id_sg] = sum_all[0];
    all_sum[(id_local * 2 + 1) * SUBGROUP_NUM + id_sg] = sum_all[1];

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint()))
    for (int i = 0; i < SUBGROUP_SIZE * 2; i += SUBGROUP_NUM) {
        float sum_all = 0;
        if (id_local < SUBGROUP_NUM) {
            sum_all = all_sum[(i + id_sg) * SUBGROUP_NUM + id_local];
        }
        sum_all = sub_group_reduce_add(sum_all);
        if (id_local == 0) {
            y[i + id_sg] = sum_all * routing_weights;
        }
    }
}

#if 0 // reference
//__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_down)(
    const __global FUNC(expert_info)* info_ptrs,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    int expert_no = get_global_id(0);
    int n = get_global_id(1);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* down_weight = (__global uchar*)info_ptr->weight[2] + n / 2;
    __global half* down_scale = (__global half*)info_ptr->scale[2] + n;
    __global uchar* down_zp = (__global uchar*)info_ptr->zp[2] + n / 2;
    float down_sum = 0;
    for (int k = 0; k < INTERMEDIATE_SIZE; k += GROUP_SIZE) {
        TYPE down_scale_v = down_scale[k / GROUP_SIZE * HIDDEN_SIZE];
        uchar down_zp_org = down_zp[k / GROUP_SIZE * HIDDEN_SIZE / 2];
        TYPE down_zp_v = convert_half((n & 1) ? (down_zp_org >> 4) : (down_zp_org & 0xf));
        float down_sum_group = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            uchar down_weight_org = down_weight[(k + i) * HIDDEN_SIZE / 2];
            TYPE down_w_v = convert_half((n & 1) ? (down_weight_org >> 4) : (down_weight_org & 0xf));
            down_sum_group += (down_w_v - down_zp_v) * x[k + i];
        }
        down_sum += down_sum_group * down_scale_v;
    }
    // down * routing
    y[n] = down_sum * routing_weights[info_ptr->routing_offset];
}
#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_down)(
    const __global FUNC(expert_info)* info_ptrs,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    int n = get_global_id(1) * 2;
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE + n;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* down_weight = (__global uchar*)info_ptr->weight[2] + n / 2;
    __global half* down_scale = (__global half*)info_ptr->scale[2] + n;
    __global uchar* down_zp = (__global uchar*)info_ptr->zp[2] + n / 2;
    __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM];
    gemv_64n(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, all_sum, routing_weights[info_ptr->routing_offset]);
}
#endif
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
