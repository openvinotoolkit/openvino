// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstddef>
#include <cstdint>

/**
 * @brief  Computes the length in bytes for the packed matrix B buffer(SGEMM).
 *
 * @param N       Supplies the number of columns of matrix B.
 * @param K       Supplies the number of rows of matrix B.
 * @return        bytes of the packing buffer
 */
size_t ov_sgemm_pack_get_size(const int64_t N, const int64_t K);

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
void ov_sgemm_pack(const char* transb,
                   const int64_t N,
                   const int64_t K,
                   const int64_t ldb,
                   const float* src,
                   float* dst);

/**
 * @brief  Signle threaded SGEMM
 *
 * @param transa  T for transpose A, N for none-tranpose A.
 * @param transb  T for transpose A, N for none-tranpose A.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param alpha   Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A       Supplies the address of matrix A
 * @param lda     Supplies the first dimension of matrix A.
 * @param B       Supplies the address of matrix B
 * @param ldb     Supplies the first dimension of matrix B.
 * @param beta    Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C       Supplies the address of matrix C
 * @param ldc     Supplies the first dimension of matrix C.
 */
void ov_sgemm_compute(const char* transa,
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
                      const int64_t ldc);

/**
 * @brief SGEMM with B matrix prepacked
 *
 * @param transa  T for transpose A, N for none-tranpose A.
 * @param transb  T for transpose A, N for none-tranpose A.
 * @param M       Supplies the number of rows of matrix A and matrix C.
 * @param N       Supplies the number of columns of matrix B and matrix C.
 * @param K       Supplies the number of columns of matrix A and the number
                  of rows of matrix B.
 * @param alpha   Supplies the scalar alpha multiplier (see SGEMM definition)
 * @param A       Supplies the address of matrix A
 * @param lda     Supplies the first dimension of matrix A.
 * @param B       Supplies the address of matrix B
 * @param ldb     Supplies the first dimension of matrix B.
 * @param beta    Supplies the scalar beta multiplier (see SGEMM definition)
 * @param C       Supplies the address of matrix C
 * @param ldc     Supplies the first dimension of matrix C.
 * @param bias    Supplies the address of by-channel bias
 */
void ov_sgemm_pack_compute(const char* transa,
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
                           const float* bias = nullptr);