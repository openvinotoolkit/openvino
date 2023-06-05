// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstddef>
#include <cstdint>

size_t ov_sgemm_pack_get_size(const char *identifier, const int64_t M, const int64_t N, const int64_t K);

void ov_sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const int64_t M, const int64_t N, const int64_t K,
        const int64_t lda, const int64_t ldb, const float *src, float *dst);

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
                      const int64_t ldc);