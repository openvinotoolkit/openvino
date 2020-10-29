/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include "cpu_isa_traits.hpp"
#include "mkldnn_types.h"
#include "os_blas.hpp"

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
        const b_dt *B, const int *ldb, const b_dt *bo, const float *beta,
        int32_t *c, const int *ldc, const int32_t *co);

mkldnn_status_t gemm_bf16bf16f32(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const mkldnn_bfloat16_t *A, const int *lda, const mkldnn_bfloat16_t *B,
        const int *ldb, const float *beta, float *C, const int *ldc);

#ifdef USE_CBLAS
#define GEMM_IMPL_STR "gemm:blas"
#else
#define GEMM_IMPL_STR "gemm:jit"
#endif

#if USE_MKL_IGEMM
#define IGEMM_S8U8S32_IMPL_STR "igemm_s8u8s32:blas"
#define IGEMM_S8S8S32_IMPL_STR "igemm_s8s8s32:blas"
#else
#define IGEMM_S8U8S32_IMPL_STR "igemm_s8u8s32:jit"
#define IGEMM_S8S8S32_IMPL_STR "igemm_s8s8s32:jit"
#endif

#ifndef USE_MKL_IGEMM
#define IGEMM_S8U8S32_ISA_STR \
    JIT_IMPL_NAME_HELPER(IGEMM_S8U8S32_IMPL_STR ":", \
            mayiuse(avx512_core_vnni) \
                    ? avx512_core_vnni \
                    : (mayiuse(avx512_core) ? avx512_core : isa_any), \
            "")
#else
#define IGEMM_S8U8S32_ISA_STR IGEMM_S8U8S32_IMPL_STR
#endif

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif // GEMM_HPP
