/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dnnl/dnnl.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/gemm/gemm.hpp"
#endif

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"

using namespace dnnl::impl;

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
namespace {
const char *c2f_offsetC(const char *offC) {
    if (offC) {
        if (offC[0] == 'R' || offC[0] == 'r') return "C";
        if (offC[0] == 'C' || offC[0] == 'c') return "R";
    }
    return offC;
}
} // namespace
#endif

dnnl_status_t dnnl_sgemm(char transa, char transb, dim_t M, dim_t N, dim_t K,
        float alpha, const float *A, dim_t lda, const float *B, const dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    return cpu::extended_sgemm(&transb, &transa, &N, &M, &K, &alpha, B, &ldb, A,
            &lda, &beta, C, &ldc);
#else
    return dnnl::impl::status::unimplemented;
#endif
}

dnnl_status_t dnnl_gemm_u8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const uint8_t *A, dim_t lda, uint8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    return cpu::gemm_s8x8s32(&transb, &transa, c2f_offsetC(&offsetc), &N, &M,
            &K, &alpha, B, &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
#else
    return dnnl::impl::status::unimplemented;
#endif
}

dnnl_status_t dnnl_gemm_s8s8s32(char transa, char transb, char offsetc, dim_t M,
        dim_t N, dim_t K, float alpha, const int8_t *A, dim_t lda, int8_t ao,
        const int8_t *B, dim_t ldb, int8_t bo, float beta, int32_t *C,
        dim_t ldc, const int32_t *co) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    return cpu::gemm_s8x8s32<int8_t>(&transb, &transa, c2f_offsetC(&offsetc),
            &N, &M, &K, &alpha, B, &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
#else
    return dnnl::impl::status::unimplemented;
#endif
}

extern "C" dnnl_status_t DNNL_API dnnl_gemm_bf16bf16f32(char transa,
        char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    return cpu::gemm_bf16bf16f32(&transb, &transa, &N, &M, &K, &alpha, B, &ldb,
            A, &lda, &beta, C, &ldc);
#else
    return dnnl::impl::status::unimplemented;
#endif
}

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
dnnl_status_t dnnl_threadpool_interop_sgemm(char transa, char transb, dim_t M,
        dim_t N, dim_t K, float alpha, const float *A, dim_t lda,
        const float *B, const dim_t ldb, float beta, float *C, dim_t ldc,
        void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = cpu::extended_sgemm(&transb, &transa, &N, &M, &K, &alpha,
            B, &ldb, A, &lda, &beta, C, &ldc, nullptr, false);
    threadpool_utils::deactivate_threadpool();
    return status;
}

dnnl_status_t dnnl_threadpool_interop_gemm_u8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const uint8_t *A,
        dim_t lda, uint8_t ao, const int8_t *B, dim_t ldb, int8_t bo,
        float beta, int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = cpu::gemm_s8x8s32(&transb, &transa, c2f_offsetC(&offsetc),
            &N, &M, &K, &alpha, B, &ldb, &bo, A, &lda, &ao, &beta, C, &ldc, co);
    threadpool_utils::deactivate_threadpool();
    return status;
}

dnnl_status_t dnnl_threadpool_interop_gemm_s8s8s32(char transa, char transb,
        char offsetc, dim_t M, dim_t N, dim_t K, float alpha, const int8_t *A,
        dim_t lda, int8_t ao, const int8_t *B, dim_t ldb, int8_t bo, float beta,
        int32_t *C, dim_t ldc, const int32_t *co, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = cpu::gemm_s8x8s32<int8_t>(&transb, &transa,
            c2f_offsetC(&offsetc), &N, &M, &K, &alpha, B, &ldb, &bo, A, &lda,
            &ao, &beta, C, &ldc, co);
    threadpool_utils::deactivate_threadpool();
    return status;
}

extern "C" dnnl_status_t DNNL_API dnnl_threadpool_interop_gemm_bf16bf16f32(
        char transa, char transb, dim_t M, dim_t N, dim_t K, float alpha,
        const bfloat16_t *A, dim_t lda, const bfloat16_t *B, dim_t ldb,
        float beta, float *C, dim_t ldc, void *th) {
    threadpool_utils::activate_threadpool(
            (dnnl::threadpool_interop::threadpool_iface *)th);
    status_t status = cpu::gemm_bf16bf16f32(&transb, &transa, &N, &M, &K,
            &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
    threadpool_utils::deactivate_threadpool();
    return status;
}
#endif
