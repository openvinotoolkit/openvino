// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_cpu_gemm.h"

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "onednn/dnnl.h"
#include "mlas.h"

namespace ov {
namespace cpu {
class ThreadPool {
public:
    ThreadPool() = delete;
    explicit ThreadPool(const size_t& threadNum) : threadNum(threadNum) {}
public:
    // the actual threads used for sgemm
    size_t threadNum = 0;
};
size_t DegreeOfParallelism(ThreadPool* tp) {
    // threadpool nullptr means single threaded
    return tp ? tp->threadNum : 1;
}
void TrySimpleParallelFor(ThreadPool* tp, const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    if (tp == nullptr) {
        for (std::ptrdiff_t i = 0; i < total; i++) {
            fn(i);
        }
    } else {
        ov::parallel_nt(tp->threadNum, [&](const size_t ithr, const size_t nthr) {
            std::ptrdiff_t start = 0, end = 0;
            ov::splitter(total, nthr, ithr, start, end);
            for (std::ptrdiff_t i = start; i < end; i++) {
                fn(i);
            }
        });
    }
}
size_t getCacheSize(int level, bool perCore) {
    return dnnl::utils::get_cache_size(level, perCore);
}
};  // namespace cpu
};  // namespace ov

size_t ov_sgemm_pack_get_size(const int64_t N, const int64_t K) {
    return MlasGemmPackBSize(N, K);
}

void ov_sgemm_pack(const char* transb,
                   const int64_t N,
                   const int64_t K,
                   const int64_t ldb,
                   const float* src,
                   float* dst) {
    MlasGemmPackB(*transb == 'T' ? CblasTrans : CblasNoTrans, N, K, src, ldb, dst);
}

void ov_sgemm(const char* transa,
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
              size_t thread_num) {
    // C = alpha*op( A )op( B ) + beta * C
    MLAS_SGEMM_DATA_PARAMS sgemmParam;
    sgemmParam.BIsPacked = false;
    sgemmParam.A = A;
    sgemmParam.lda = lda;
    sgemmParam.B = B;
    sgemmParam.ldb = ldb;
    sgemmParam.C = C;
    sgemmParam.ldc = ldc;
    sgemmParam.alpha = alpha;
    sgemmParam.beta = beta;
    auto _transa = *transa == 'N' ? CblasNoTrans : CblasTrans;
    auto _transb = *transb == 'N' ? CblasNoTrans : CblasTrans;
    ov::cpu::ThreadPool threadPool(0 == thread_num ? parallel_get_num_threads() : thread_num);
    MlasGemmBatch(_transa, _transb, M, N, K, &sgemmParam, 1, &threadPool);
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
                      const int64_t ldc,
                      const float* bias) {
    // C = alpha*op( A )op( B ) + beta * C
    ov::cpu::ThreadPool threadPool(parallel_get_num_threads());
    MLAS_SGEMM_DATA_PARAMS sgemmParam;
    sgemmParam.BIsPacked = true;
    sgemmParam.A = A;
    sgemmParam.lda = lda;
    sgemmParam.B = B;
    sgemmParam.ldb = ldb;
    sgemmParam.C = C;
    sgemmParam.ldc = ldc;
    sgemmParam.alpha = alpha;
    sgemmParam.beta = beta;
    sgemmParam.bias = bias;
    auto _transa = *transa == 'N' ? CblasNoTrans : CblasTrans;
    auto _transb = *transb == 'N' ? CblasNoTrans : CblasTrans;
    MlasGemmBatch(_transa, _transb, M, N, K, &sgemmParam, 1, &threadPool);
}