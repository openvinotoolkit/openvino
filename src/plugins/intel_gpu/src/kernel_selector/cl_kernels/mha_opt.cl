// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// inline uint FUNC(print_matrix_half)(__constant char *title, half* buf, int rows, int cols) {
//     printf("%s\n", title);
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%.3f  ", (float)buf[cols * i + j]);
//         }
//         printf("\n");
//     }
// }

// inline uint FUNC(print_matrix_float)(__constant char *title, float* buf, int rows, int cols) {
//     printf("%s\n", title);
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             printf("%.3f  ", (float)buf[cols * i + j]);
//         }
//         printf("\n");
//     }
// }

/*
 * DEFINES
 * d:  DEPTH_SIZE
 * Br: BLK_ROW_SIZE
 * Bc: BLK_COL_SIZE
 * Tr: NUM_BLK_ROW
 * Tc: NUM_BLK_COL
 * Br * d: Q_BLK_SIZE
 * Bc * d: K_BLK_SIZE
 * Bc * d: V_BLK_SIZE
 * Br * Bc: SCORE_MAT_SIZE
 * Br * d: OUT_BLK_SIZE
 * NUM_COL_THREAD
 * COL_THREAD_SIZE
 */
// REQD_SUB_GROUP_SIZE(16)
// __attribute__((reqd_work_group_size(1, 1, 16)))
KERNEL(mha_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* inputq,
    const __global INPUT1_TYPE* inputk,
    const __global INPUT2_TYPE* inputv,
    __global OUTPUT_TYPE* output)
{
    const uint b = (uint)get_global_id(0) / OUTPUT_FEATURE_NUM; // batch index
    const uint f = (uint)get_global_id(0) % OUTPUT_FEATURE_NUM; // head index
    const uint block_id = (uint)get_global_id(1);
    const uint row_id = (uint)get_global_id(2);

    half p_m = -HALF_MAX;
    half m = -HALF_MAX;
    half p_l = 0;
    half l = 0;
    __local half P[SCORE_MAT_SIZE];   // SCORE_MAT_SIZE = Br * Bc
    __local half O[OUT_BLK_SIZE];     // OUT_BLK_SIZE = Br * d
    __local half k_block[K_BLK_SIZE];
    __local half v_block[V_BLK_SIZE];
    __local half q_block[Q_BLK_SIZE];

    // Read i-th row of Q block
    const int q_row_idx = BLK_ROW_SIZE * block_id + row_id;
    for (int c = 0; c < DEPTH_SIZE; c++) {
        // Replace GET_INDEX_SAFE, it's slow.
        q_block[DEPTH_SIZE * row_id + c] = inputq[INPUT0_GET_INDEX_SAFE(b, f, q_row_idx, c)];
        O[DEPTH_SIZE * row_id + c] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    half row_max = -HALF_MAX;
    half row_sum = 0.0f;

// note: DEPTH_SIZE is supposed to be multiple of 4 at this moment
#define VEC_TYPE half4
#define VEC_SIZE 4

    for (int j = 0; j < NUM_BLK_COL; j++) {
        // Fill Key block
        const int x_offset = BLK_COL_SIZE * j; // X-axis
        unroll_for (int y = row_id; y < DEPTH_SIZE; y += BLK_ROW_SIZE) {
            int kidx = INPUT1_GET_INDEX(b, f, y, x_offset);
            unroll_for (int x = 0; x < BLK_COL_SIZE; x++) {
                k_block[(DEPTH_SIZE * x ) + y] = inputk[kidx + x];
            }
        }

        // Fill Value block
        const int y_offset = BLK_COL_SIZE * j; // Y-axis
        unroll_for (int y = row_id; y < BLK_COL_SIZE; y += BLK_ROW_SIZE) {
            int vidx = INPUT2_GET_INDEX(b, f, y_offset + y, 0);
            unroll_for (int x = 0; x < DEPTH_SIZE; x++) {
                v_block[(BLK_COL_SIZE * x) + y] = inputv[vidx + x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // S = matmul(Q, K) and get max value.
        row_max = -HALF_MAX;
        for (int c = 0; c < BLK_COL_SIZE; c++) {
            VEC_TYPE acc4 = 0.f;
            unroll_for (int d = 0; d < DEPTH_SIZE; d += VEC_SIZE) {
                acc4 = mad(*(VEC_TYPE*)(q_block + DEPTH_SIZE * row_id + d), *(VEC_TYPE*)(k_block + DEPTH_SIZE * c + d), acc4);
            }
            half acc = 0.f;
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                acc += acc4[i];
            }
            P[BLK_COL_SIZE * row_id + c] = acc;
            row_max = max(row_max , acc);
        }
        m = max(p_m, row_max);

        // Calculate P
        row_sum = 0.0f;
        half4 e = 0.f;
        unroll_for (int x = 0; x < BLK_COL_SIZE; x += VEC_SIZE) {
            e = exp((*(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) - (VEC_TYPE)m));
            *(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + x) = e;
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                row_sum += e[i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Calculate l value.
        half exp_m = exp(p_m - m);
        l = exp_m * p_l + row_sum;

        // Calculate O + PV block.
        half acc = 0.f;
        VEC_TYPE acc4 = 0.f;
        for (int d = 0; d < DEPTH_SIZE; d++) {
            acc4 = 0.f;
            unroll_for (int c = 0; c < BLK_COL_SIZE; c += VEC_SIZE) {
                acc4 = mad(*(VEC_TYPE*)(P + BLK_COL_SIZE * row_id + c), *(VEC_TYPE*)(v_block + BLK_COL_SIZE * d + c), acc4);
            }
            acc = 0.f;
            unroll_for (int i = 0; i < VEC_SIZE; i++) {
                acc += acc4[i];
            }
            O[DEPTH_SIZE * row_id + d] = exp_m * O[DEPTH_SIZE * row_id + d] + acc;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Set m(j-1) and l(j-1)
        p_m = m;
        p_l = l;
    }

    const int out_row_idx = BLK_ROW_SIZE * block_id + row_id;
    int oidx = OUTPUT_GET_INDEX(b, f, out_row_idx, 0);
    unroll_for (int c = 0; c < DEPTH_SIZE; c++) {
        output[oidx + c] = O[DEPTH_SIZE * row_id + c]/l;
    }
}
