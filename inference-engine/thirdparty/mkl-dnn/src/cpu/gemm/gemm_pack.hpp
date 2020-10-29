/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMM_PACK_HPP
#define GEMM_PACK_HPP

#include "mkldnn_types.h"

#include "cpu_isa_traits.hpp"

#include "os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#if USE_MKL_PACKED_GEMM
static inline bool pack_sgemm_supported() {
    return true;
}
#else
static inline bool pack_sgemm_supported() {
    return mayiuse(sse42);
}
#endif

static inline bool pack_gemm_bf16bf16f32_supported() {
    return mayiuse(avx512_core);
}

mkldnn_status_t MKLDNN_API sgemm_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API gemm_bf16bf16f32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API gemm_s8s8s32_pack_get_size(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, size_t *size,
        bool *pack = nullptr);

mkldnn_status_t MKLDNN_API sgemm_pack(const char *identifier, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const int *lda, const int *ldb, const float *src, float *dst);

mkldnn_status_t MKLDNN_API gemm_bf16bf16f32_pack(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const mkldnn_bfloat16_t *src,
        mkldnn_bfloat16_t *dst);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_pack(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const void *src,
        void *dst);

mkldnn_status_t MKLDNN_API gemm_s8s8s32_pack(const char *identifier,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const void *src,
        void *dst);

mkldnn_status_t MKLDNN_API sgemm_compute(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *A,
        const int *lda, const float *B, const int *ldb, const float *beta,
        float *C, const int *ldc);

mkldnn_status_t MKLDNN_API gemm_bf16bf16f32_compute(const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const mkldnn_bfloat16_t *A, const int *lda, const mkldnn_bfloat16_t *B,
        const int *ldb, const float *beta, float *C, const int *ldc);

mkldnn_status_t MKLDNN_API gemm_s8u8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const int *M, const int *N,
        const int *K, const int8_t *A, const int *lda, const uint8_t *B,
        const int *ldb, const float *beta, int32_t *C, const int *ldc,
        const int32_t *co);

mkldnn_status_t MKLDNN_API gemm_s8s8s32_compute(const char *transa,
        const char *transb, const char *offsetc, const int *M, const int *N,
        const int *K, const int8_t *A, const int *lda, const int8_t *B,
        const int *ldb, const float *beta, int32_t *C, const int *ldc,
        const int32_t *co);

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif // GEMM_PACK_HPP
