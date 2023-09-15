// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

inline uint FUNC(print_matrix_half)(__constant char *title, half* buf, int rows, int cols) {
    printf("%s\n", title);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f  ", (float)buf[cols * i + j]);
        }
        printf("\n");
    }
}

inline uint FUNC(print_matrix_float)(__constant char *title, float* buf, int rows, int cols) {
    printf("%s\n", title);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f  ", (float)buf[cols * i + j]);
        }
        printf("\n");
    }
}

KERNEL(mha_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* inputq,
    const __global INPUT1_TYPE* inputk,
    const __global INPUT2_TYPE* inputv,
    __global OUTPUT_TYPE* output)
{
    int i;
    float s[10000];  // It will fail for large input
    // FIXME: kernel.cpp should limit input size
    const int N = INPUT0_SIZE_Y; // FIXME: to be defined from jitter
    // FIXME: need to use ACCUMULATOR_TYPE

    for (int i = 0; i < INPUT0_FEATURE_NUM; i++) { // handle batch
        // Matmul
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                float acc = 0.f;
                for (int m = 0; m < INPUT0_SIZE_X; m++) {
                    acc += (float)(inputq[INPUT0_GET_INDEX_SAFE(0, i, j, m)]) * (float)(inputk[INPUT1_GET_INDEX_SAFE(0, i, m, k)]);
                }
                s[N * j + k] = acc;
            }
        }
        FUNC(print_matrix_float)("score matrix", s, N, N);

        // Softmax
        for (int j = 0; j < N; j++) {
            float max_val = -1000.f;
            for (int k = 0; k < N; k++) {
                if (s[N * j + k] > max_val)
                    max_val = s[N * j + k];
            }
            float l = 0.f;
            for (int k = 0; k < N; k++) {
                s[N * j + k] = exp(s[N * j + k] - max_val);
                l += s[N * j + k];
            }

            for (int k = 0; k < N; k++) {
                s[N * j + k] /= l;
            }
        }

        FUNC(print_matrix_float)("softmax(score) matrix", s, N, N);

        // Matmul
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < INPUT2_SIZE_X; k++) {
                float acc = 0.f;
                for (int m = 0; m < N; m++) {
                    acc += (float)(s[N * j + m]) * (float)(inputv[INPUT2_GET_INDEX_SAFE(0, i, m, k)]);
                }
                output[OUTPUT_GET_INDEX(0, i, j, k)] = acc;
            }
        }
        FUNC(print_matrix_half)("result matrix", output, N, INPUT2_SIZE_X);
    }
}
