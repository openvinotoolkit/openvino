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
//__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global FUNC(expert_info)* info_ptrs,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                        // [MAX_TOPK, INTERMEDIATE_SIZE]
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

#elif DOWN_ENABLE

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
