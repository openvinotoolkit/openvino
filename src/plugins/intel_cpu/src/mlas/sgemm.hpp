// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu {
/**
 * @brief  Computes the length in bytes for the packed matrix B buffer(SGEMM).
 *
 * @param N       Supplies the number of columns of matrix B.
 * @param K       Supplies the number of rows of matrix B.
 * @return        bytes of the packing buffer
 */
size_t mlas_sgemm_pack_get_size(const int64_t N, const int64_t K);

/**
 * @brief  Packs the contents of matrix B
 *
 * @param transb  T for transpose B, N for none-tranpose B
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param ldb     Supplies the first dimension of matrix B.
 * @param src     Supplies the address of matrix B
 * @param dst     Supplies pointer to prePacked B buffer
 */
void mlas_sgemm_pack(const char* transb,
                     const int64_t N,
                     const int64_t K,
                     const int64_t ldb,
                     const float* src,
                     float* dst);

/**
 * @brief  SGEMM with planar B matrix
 *
 * @param transa       T for transpose A, N for none-tranpose A.
 * @param transb       T for transpose B, N for none-tranpose B.
 * @param M            Supplies the number of rows of matrix A and matrix C.
 * @param N            Supplies the number of columns of matrix B and matrix C.
 * @param K            Supplies the number of columns of matrix A and the number
                       of rows of matrix B.
 * @param alpha        Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A            Supplies the address of matrix A
 * @param lda          Supplies the first dimension of matrix A.
 * @param B            Supplies the address of matrix B
 * @param ldb          Supplies the first dimension of matrix B.
 * @param beta         Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C            Supplies the address of matrix C
 * @param ldc          Supplies the first dimension of matrix C.
 * @param thread_num   0 for all threads, otherwise use thread_num
 */
void mlas_sgemm(const char* transa,
                const char* transb,
                const int64_t M,
                const int64_t N,
                const int64_t K,
                const float alpha,
                const float* A,
                const int64_t lda,
                const float* B,
                const int64_t ldb,
                const float beta,
                float* C,
                const int64_t ldc,
                size_t thread_num = 0);

/**
 * @brief SGEMM with B matrix prepacked
 *
 * @param transa       T for transpose A, N for none-tranpose A.
 * @param transb       T for transpose B, N for none-tranpose B.
 * @param M            Supplies the number of rows of matrix A and matrix C.
 * @param N            Supplies the number of columns of matrix B and matrix C.
 * @param K            Supplies the number of columns of matrix A and the number
                       of rows of matrix B.
 * @param alpha        Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A            Supplies the address of matrix A
 * @param lda          Supplies the first dimension of matrix A.
 * @param B            Supplies the address of matrix B
 * @param ldb          Supplies the first dimension of matrix B.
 * @param beta         Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C            Supplies the address of matrix C
 * @param ldc          Supplies the first dimension of matrix C.
 * @param bias         Supplies the address of by-channel bias
 * @param thread_num   0 for all threads, otherwise use thread_num
 */
void mlas_sgemm_compute(const char* transa,
                        const char* transb,
                        const int64_t M,
                        const int64_t N,
                        const int64_t K,
                        const float alpha,
                        const float* A,
                        const int64_t lda,
                        const float* B,
                        const int64_t ldb,
                        const float beta,
                        float* C,
                        const int64_t ldc,
                        const float* bias = nullptr,
                        size_t thread_num = 0);
}  // namespace ov::intel_cpu
