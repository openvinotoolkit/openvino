/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#ifndef GEMM_HPP
#define GEMM_HPP
namespace mkldnn {
namespace impl {
namespace cpu {
mkldnn_status_t extended_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc,
        const float *bias = nullptr, bool force_jit_gemm = false);

template <typename b_dt>
mkldnn_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const b_dt *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *c, const int *ldc, const int32_t *co);

template <typename data_t>
void ref_gemm(const char *transa, const char *transb, const int *M,
        const int *N, const int *K, const data_t *alpha, const data_t *A,
        const int *lda, const data_t *B, const int *ldb, const data_t *beta,
        data_t *C, const int *ldc, const data_t *bias);
#ifdef USE_CBLAS
#define GEMM_IMPL_STR "gemm:blas"
#else
#define GEMM_IMPL_STR "gemm:jit"
#endif
}
}
}
#endif
