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
        const float *bias = nullptr);
void ref_gemm(const char *transa, const char *transb, const int *M,
        const int *N, const int *K, const float *alpha, const float *A,
        const int *lda, const float *B, const int *ldb, const float *beta,
        float *C, const int *ldc, const float *bias);
#ifdef USE_CBLAS
#define GEMM_IMPL_STR "gemm:blas"
#else
#define GEMM_IMPL_STR "gemm:jit"
#endif
}
}
}
#endif
