// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mlas.h"
#include "ie_parallel.hpp"
#include "ov_cpu_gemm.h"
#include <vector>
#include <string>

namespace ov {
namespace cpu {
class ThreadPool {
public:
    ThreadPool() = default;
};
size_t getTotalThreads() {
    return parallel_get_max_threads();
}
void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    parallel_for(total, fn);
}
};  // namespace cpu
};  // namespace ov


size_t ov_sgemm_pack_get_size(const char *identifier, const int64_t M, const int64_t N, const int64_t K) {
    //TODO API to wrap MKL
    //MLAS only support pack B
    if (*identifier == 'A')
        return -1;
    else
        return MlasGemmPackBSize(N, K);
}

void ov_sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const int64_t M, const int64_t N, const int64_t K,
        const int64_t lda, const int64_t ldb, const float *src, float *dst) {
    if (*identifier == 'B') {
        MlasGemmPackB(CblasNoTrans,
            N,
            K,
            src,
            transb ? K : N,
            dst);
    }
}

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
                      const int64_t ldc) {
    // C = alpha*op( A )op( B ) + beta * C
    std::vector<MLAS_SGEMM_DATA_PARAMS> data(1);
    data[0].BIsPacked = false;
    data[0].A = A;
    data[0].lda = lda;
    data[0].B = B;
    data[0].ldb = ldb;
    data[0].C = C;
    data[0].ldc = ldc;
    data[0].alpha = alpha;
    data[0].beta = beta;
    auto _transa = *transa == 'N' ? CblasNoTrans : CblasTrans;
    auto _transb = *transb == 'N' ? CblasNoTrans : CblasTrans;
    MlasGemmBatch(_transa, _transb, M, N, K, data.data(), 1, nullptr);
}

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
                      const int64_t ldc) {
    // C = alpha*op( A )op( B ) + beta * C
    ov::cpu::ThreadPool threadPool;
    std::vector<MLAS_SGEMM_DATA_PARAMS> data(1);
    data[0].BIsPacked = true;
    data[0].A = A;
    data[0].lda = lda;
    data[0].B = B;
    data[0].ldb = ldb;
    data[0].C = C;
    data[0].ldc = ldc;
    data[0].alpha = alpha;
    data[0].beta = beta;
    auto _transa = *transa == 'N' ? CblasNoTrans : CblasTrans;
    auto _transb = *transb == 'N' ? CblasNoTrans : CblasTrans;
    MlasGemmBatch(_transa, _transb, M, N, K, data.data(), 1, &threadPool);
}