// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

static inline void matrixMult(float *A, float *B, float *C, int m, int n, int k, bool transposeB = false) {
    if (transposeB) {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int rowB = 0; rowB < n; rowB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[rowB * k + colA];
                }

                C[rowA * n + rowB] = sum;
            }
        }
    } else {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int colB = 0; colB < n; colB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[colA * n + colB];
                }

                C[rowA * n + colB] = sum;
            }
        }
    }
}