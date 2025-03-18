// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sgemm.hpp"

#include <string>
#include <vector>

#include "mlas.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "thread_pool.hpp"

namespace ov::intel_cpu {

size_t mlas_sgemm_pack_get_size(const int64_t N, const int64_t K) {
    return MlasGemmPackBSize(N, K);
}

void mlas_sgemm_pack(const char* transb,
                     const int64_t N,
                     const int64_t K,
                     const int64_t ldb,
                     const float* src,
                     float* dst) {
    MlasGemmPackB(*transb == 'T' ? CblasTrans : CblasNoTrans, N, K, src, ldb, dst);
}

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
    ov::cpu::OVMlasThreadPool threadPool(0 == thread_num ? parallel_get_max_threads() : thread_num);
    MlasGemmBatch(_transa, _transb, M, N, K, &sgemmParam, 1, &threadPool);
}

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
                        const float* bias,
                        size_t thread_num) {
    // C = alpha*op( A )op( B ) + beta * C
    ov::cpu::OVMlasThreadPool threadPool(0 == thread_num ? parallel_get_max_threads() : thread_num);
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
}  // namespace ov::intel_cpu
