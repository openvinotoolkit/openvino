// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// floatmath.cpp : unoptimized floating point math routines (for reference)
//

#include "floatmath.h"

#include <cstdint>
#include <cstdio>

#ifdef _NO_MKL_
void cblas_sgemm1(const CBLAS_LAYOUT Layout,
                  const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB,
                  const MKL_INT M,
                  const MKL_INT N,
                  const MKL_INT K,
                  const float alpha,
                  const float* A,
                  const MKL_INT lda,
                  const float* B,
                  const MKL_INT ldb,
                  const float beta,
                  float* C,
                  const MKL_INT ldc) {
    int i, j, k;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_sgemm!\n");
        throw - 1;
    }

    if ((TransA == CblasNoTrans) && (TransB == CblasNoTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[i * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasNoTrans) && (TransB == CblasTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum;
                sum = beta * C[i * ldc + j];
                for (k = 0; k < K; k++) {
                    sum += alpha * A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasTrans) && (TransB == CblasNoTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[i * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * lda + i] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else {
        fprintf(stderr, "Expected A not transposed in cblas_sgemm!\n");
        throw - 1;
    }
}
void cblas_ssbmv1(const CBLAS_LAYOUT Layout,
                  const CBLAS_UPLO Uplo,
                  const MKL_INT N,
                  const MKL_INT K,
                  const float alpha,
                  const float* A,
                  const MKL_INT lda,
                  const float* X,
                  const MKL_INT incX,
                  const float beta,
                  float* Y,
                  const MKL_INT incY) {
    int i;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_ssbmv!\n");
        throw - 1;
    }
    if (Uplo != CblasLower) {
        fprintf(stderr, "Only lower format is supported in cblas_ssbmv!\n");
        throw - 1;
    }
    if (K != 0) {
        fprintf(stderr, "Only diagonal matrices supported in cblas_ssbmv at this time!\n");
        throw - 1;
    }
    if ((alpha == 1.0) && (beta == 1.0) && (incX == 1) && (incY == 1)) {
        for (i = 0; i < N; i++) {
            Y[i] += A[i] * X[i];
        }
    } else {
        fprintf(stderr, "Only alpha=1, beta=1, incX=1, incY=1, LDA=1 supported in cblas_ssbmv at this time!\n");
        throw - 1;
    }
}
#endif  // #ifdef _NO_MKL_

void cblas_sgemm_subset(const CBLAS_LAYOUT Layout,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const MKL_INT M,
                        const MKL_INT N,
                        const MKL_INT K,
                        const float alpha,
                        const float* A,
                        const MKL_INT lda,
                        const float* B,
                        const MKL_INT ldb,
                        const float beta,
                        float* C,
                        const MKL_INT ldc,
                        const uint32_t* OutputList,
                        const MKL_INT L) {
    int i, j, k, l;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_sgemm_subset!\n");
        throw - 1;
    }

    if ((TransA == CblasNoTrans) && (TransB == CblasNoTrans)) {
        for (l = 0; l < L; l++) {
            i = OutputList[l];
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[l * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[l * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasNoTrans) && (TransB == CblasTrans)) {
        for (i = 0; i < M; i++) {
            for (l = 0; l < L; l++) {
                float sum;
                j = OutputList[l];
                sum = beta * C[i * ldc + l];
                for (k = 0; k < K; k++) {
                    sum += alpha * A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + l] = sum;
            }
        }
    } else if ((TransA == CblasTrans) && (TransB == CblasNoTrans)) {
        for (l = 0; l < L; l++) {
            i = OutputList[l];
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[l * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * lda + i] * B[k * ldb + j];
                }
                C[l * ldc + j] = sum;
            }
        }
    } else {
        fprintf(stderr, "Expected A not transposed in cblas_sgemm_subset!\n");
        throw - 1;
    }
}

// C = [ A1 A2 ] * X + B
void sgemv_split(const uint32_t N,
                 const uint32_t K1,
                 const uint32_t K2,
                 const float* A1,
                 const float* A2,
                 const float* X,
                 const float* B,
                 float* C) {
    uint32_t num_columns = K1 + K2;
    uint32_t num_rows = N;
    uint32_t i, j;

    for (i = 0; i < num_rows; i++) {
        float sum = B[i];
        for (j = 0; j < K1; j++) {
            sum += A1[j] * X[i * num_columns + j];
        }
        for (j = K1; j < num_columns; j++) {
            sum += A2[j - K1] * X[i * num_columns + j];
        }
        C[i] = sum;
    }
}
