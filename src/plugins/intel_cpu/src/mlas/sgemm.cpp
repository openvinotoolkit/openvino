// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sgemm.hpp"
#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "mlas.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "thread_pool.hpp"

namespace ov {
namespace intel_cpu {

size_t mlas_sgemm_pack_get_size(const int64_t N, const int64_t K) {
    return MlasGemmPackBSize(N, K);
}

size_t mlas_half_sgemm_pack_get_size(const int64_t N, const int64_t K, bool float2half) {
    return MlasHalfGemmPackBSize(N, K, float2half);
}

template<>
void mlas_sgemm_pack(const char* transb,
                     const int64_t N,
                     const int64_t K,
                     const int64_t ldb,
                     const float* src,
                     float* dst) {
    MlasGemmPackB(*transb == 'T' ? CblasTrans : CblasNoTrans, N, K, src, ldb, dst);
}

template<>
void mlas_sgemm_pack(const char* transb,
                     const int64_t N,
                     const int64_t K,
                     const int64_t ldb,
                     const float* src,
                     uint16_t* dst) {
    MlasHalfGemmConvertPackB(N, K, src, ldb, dst);
}

template<>
void mlas_sgemm_pack(const char* transb,
                     const int64_t N,
                     const int64_t K,
                     const int64_t ldb,
                     const uint16_t* src,
                     uint16_t* dst) {
    MlasHalfGemmPackB(N, K, reinterpret_cast<const MLAS_FP16*>(src), ldb, dst);
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
    ov::cpu::OVMlasThreadPool threadPool(0 == thread_num ? parallel_get_num_threads() : thread_num);
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
    ov::cpu::OVMlasThreadPool threadPool(0 == thread_num ? parallel_get_num_threads() : thread_num);
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

void mlas_half_sgemm_compute(const int64_t M,
                             const int64_t N,
                             const int64_t K,
                             const float alpha,
                             const uint16_t* A,
                             const int64_t lda,
                             const uint16_t* B,
                             const int64_t ldb,
                             const float beta,
                             uint16_t* C,
                             const int64_t ldc,
                             const uint16_t* bias,
                             size_t thread_num) {
    // C = alpha*op( A )op( B ) + beta * C
    ov::cpu::OVMlasThreadPool threadPool(0 == thread_num ? parallel_get_num_threads() : thread_num);
    MLAS_HALF_GEMM_DATA_PARAMS sgemmParam;
    sgemmParam.A = A;
    sgemmParam.lda = lda;
    sgemmParam.B = B;
    sgemmParam.ldb = ldb;
    sgemmParam.C = reinterpret_cast<MLAS_FP16*>(C);
    sgemmParam.ldc = ldc;
    sgemmParam.Bias = reinterpret_cast<const MLAS_FP16*>(bias);
    MlasHalfGemmBatch(M, N, K, 1, &sgemmParam, &threadPool);
}

}  // namespace intel_cpu
}  // namespace ov
