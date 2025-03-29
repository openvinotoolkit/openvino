// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/gemm_common.cl"

#define MULT(C_, A_, i_)                   \
    DOT8i(C_,  B0, A_, i_ + 0);            \
    DOT8i(C_,  B8, A_, i_ + 1);            \
    DOT8i(C_, B16, A_, i_ + 2);            \
    DOT8i(C_, B24, A_, i_ + 3);

__attribute__((reqd_work_group_size(16, TY, 1)))
REQD_SUB_GROUP_SIZE(16)
KERNEL(convolution_gpu_bfyx_1x1_hgemm_buf_16x1)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __read_only image2d_t weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
)
{

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint group_x = get_group_id(0);
    const uint group_y = get_group_id(1);
    const uint batch = (uint)get_global_id(2);

#if BIAS_TERM
    const uint bias_index = group_x * TILE_N + local_x;
    half8 C0 = biases[bias_index];
    half8 C8 = biases[bias_index];
#else
    //      i [16.1]
    half8 C0 = 0.0h;
    half8 C8 = 0.0h;
#endif

    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint i = TILE_M * group_y + local_y * 16 + lxd4;

    __global const half8 *A_load = (__global const half8*)&input[batch * INPUT0_BATCH_PITCH + i*K + (lxm4<<3)];

    uint j = group_x << 4;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(short), 0);

    for (uint k8 = 0; k8 < K8; k8 += 4) {

        // 512 MADs

        half8 B0 = as_half8(intel_sub_group_block_read_us8(weights, coordB));
        coordB.y += 8;
        half8 B8 = as_half8(intel_sub_group_block_read_us8(weights, coordB));
        coordB.y += 8;

        half8 B16 = as_half8(intel_sub_group_block_read_us8(weights, coordB));
        coordB.y += 8;
        half8 B24 = as_half8(intel_sub_group_block_read_us8(weights, coordB));
        coordB.y += 8;

        half8 A0 = A_load[K8*0 + k8];
        half8 A4 = A_load[K8*4 + k8];

        MULT(C0.s0, A0, 0);
        MULT(C0.s1, A0, 4);
        MULT(C0.s2, A0, 8);
        MULT(C0.s3, A0, 12);
        MULT(C0.s4, A4, 0);
        MULT(C0.s5, A4, 4);
        MULT(C0.s6, A4, 8);
        MULT(C0.s7, A4, 12);

        A0 = A_load[K8* 8 + k8];
        A4 = A_load[K8*12 + k8];

        MULT(C8.s0, A0, 0);
        MULT(C8.s1, A0, 4);
        MULT(C8.s2, A0, 8);
        MULT(C8.s3, A0, 12);
        MULT(C8.s4, A4, 0);
        MULT(C8.s5, A4, 4);
        MULT(C8.s6, A4, 8);
        MULT(C8.s7, A4, 12);
    }

    uint y0 = group_y * TILE_M + (local_y << 4);
    __global half *C_write = &output[batch * OUTPUT_BATCH_PITCH + group_x * TILE_N + y0 * N + local_x];

    if (group_y < NUM_WHOLE_GROUPS_Y || local_y < NUM_WHOLE_SUBGROUPS_Y) {
        C_write[0*N] = ACTIVATION(C0.s0, ACTIVATION_PARAMS);
        C_write[1*N] = ACTIVATION(C0.s1, ACTIVATION_PARAMS);
        C_write[2*N] = ACTIVATION(C0.s2, ACTIVATION_PARAMS);
        C_write[3*N] = ACTIVATION(C0.s3, ACTIVATION_PARAMS);
        C_write[4*N] = ACTIVATION(C0.s4, ACTIVATION_PARAMS);
        C_write[5*N] = ACTIVATION(C0.s5, ACTIVATION_PARAMS);
        C_write[6*N] = ACTIVATION(C0.s6, ACTIVATION_PARAMS);
        C_write[7*N] = ACTIVATION(C0.s7, ACTIVATION_PARAMS);
        C_write[8*N] = ACTIVATION(C8.s0, ACTIVATION_PARAMS);
        C_write[9*N] = ACTIVATION(C8.s1, ACTIVATION_PARAMS);
        C_write[10*N] = ACTIVATION(C8.s2, ACTIVATION_PARAMS);
        C_write[11*N] = ACTIVATION(C8.s3, ACTIVATION_PARAMS);
        C_write[12*N] = ACTIVATION(C8.s4, ACTIVATION_PARAMS);
        C_write[13*N] = ACTIVATION(C8.s5, ACTIVATION_PARAMS);
        C_write[14*N] = ACTIVATION(C8.s6, ACTIVATION_PARAMS);
        C_write[15*N] = ACTIVATION(C8.s7, ACTIVATION_PARAMS);
    } else {
#if 0 < LAST_LOCAL_Y
        C_write[0*N] = ACTIVATION(C0.s0, ACTIVATION_PARAMS);
#endif
#if 1 < LAST_LOCAL_Y
        C_write[1*N] = ACTIVATION(C0.s1, ACTIVATION_PARAMS);
#endif
#if 2 < LAST_LOCAL_Y
        C_write[2*N] = ACTIVATION(C0.s2, ACTIVATION_PARAMS);
#endif
#if 3 < LAST_LOCAL_Y
        C_write[3*N] = ACTIVATION(C0.s3, ACTIVATION_PARAMS);
#endif
#if 4 < LAST_LOCAL_Y
        C_write[4*N] = ACTIVATION(C0.s4, ACTIVATION_PARAMS);
#endif
#if 5 < LAST_LOCAL_Y
        C_write[5*N] = ACTIVATION(C0.s5, ACTIVATION_PARAMS);
#endif
#if 6 < LAST_LOCAL_Y
        C_write[6*N] = ACTIVATION(C0.s6, ACTIVATION_PARAMS);
#endif
#if 7 < LAST_LOCAL_Y
        C_write[7*N] = ACTIVATION(C0.s7, ACTIVATION_PARAMS);
#endif
#if 8 < LAST_LOCAL_Y
        C_write[8*N] = ACTIVATION(C8.s0, ACTIVATION_PARAMS);
#endif
#if 9 < LAST_LOCAL_Y
        C_write[9*N] = ACTIVATION(C8.s1, ACTIVATION_PARAMS);
#endif
#if 10 < LAST_LOCAL_Y
        C_write[10*N] = ACTIVATION(C8.s2, ACTIVATION_PARAMS);
#endif
#if 11 < LAST_LOCAL_Y
        C_write[11*N] = ACTIVATION(C8.s3, ACTIVATION_PARAMS);
#endif
#if 12 < LAST_LOCAL_Y
        C_write[12*N] = ACTIVATION(C8.s4, ACTIVATION_PARAMS);
#endif
#if 13 < LAST_LOCAL_Y
        C_write[13*N] = ACTIVATION(C8.s5, ACTIVATION_PARAMS);
#endif
#if 14 < LAST_LOCAL_Y
        C_write[14*N] = ACTIVATION(C8.s6, ACTIVATION_PARAMS);
#endif
#if 15 < LAST_LOCAL_Y
        C_write[15*N] = ACTIVATION(C8.s7, ACTIVATION_PARAMS);
#endif
    }
}
