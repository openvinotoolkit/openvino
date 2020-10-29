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

#include "mkldnn.h"

#include "mkldnn_traits.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "gemm.hpp"

#include "f32/jit_avx512_common_gemm_f32.hpp"
#include "f32/jit_avx_gemm_f32.hpp"
#include "f32/ref_gemm_f32.hpp"

#include "gemm_driver.hpp"
#include "s8x8s32/ref_gemm_s8x8s32.hpp"
#include "s8x8s32/simple_gemm_s8s8s32.hpp"

#include "os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const void *A, const int *lda,
        const void *B, const int *ldb, const void *C, const int *ldc,
        const float *alpha, const float *beta, const bool with_bias) {
    if (utils::any_null(
                transa, transb, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta))
        return mkldnn_invalid_arguments;
    if (with_bias && *beta != 0) return mkldnn_unimplemented;
    bool consistency = true
            && utils::one_of(*transa, 'T', 't', 'N', 'n', 'P', 'p')
            && utils::one_of(*transb, 'T', 't', 'N', 'n', 'P', 'p') && *M >= 0
            && *N >= 0 && *K >= 0;

    if (!consistency) return mkldnn_invalid_arguments;

    bool is_packed_a = utils::one_of(*transa, 'P', 'p');
    bool is_packed_b = utils::one_of(*transb, 'P', 'p');
    bool is_trans_a = utils::one_of(*transa, 'T', 't');
    bool is_trans_b = utils::one_of(*transb, 'T', 't');
    int nrow_a = is_trans_a ? *K : *M;
    int nrow_b = is_trans_b ? *N : *K;
    consistency = true && (is_packed_a || *lda >= nstl::max(1, nrow_a))
            && (is_packed_b || *ldb >= nstl::max(1, nrow_b))
            && *ldc >= nstl::max(1, *M);
    if (!consistency) return mkldnn_invalid_arguments;

    return mkldnn_success;
}

mkldnn_status_t check_gemm_x8x8x32_input(const char *offsetc, const char *transa,
        const char *transb, const int *M, const int *N, const int *K,
        const void *A, const int *lda, const void *B, const int *ldb,
        const void *C, const int *ldc, const float *alpha, const float *beta,
        const bool with_bias) {
    if (offsetc == nullptr) return mkldnn_invalid_arguments;
    if (!utils::one_of(*offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return mkldnn_invalid_arguments;

    return check_gemm_input(transa, transb, M, N, K, A, lda, B, ldb, C, ldc,
            alpha, beta, with_bias);
}

mkldnn_status_t extended_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc, const float *bias,
        const bool force_jit_nocopy_gemm) {
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K, A, lda, B,
            ldb, C, ldc, alpha, beta, bias != nullptr);
    if (status != mkldnn_success) return status;

#ifdef USE_CBLAS
    if (!force_jit_nocopy_gemm && utils::one_of(*transa, 'n', 'N', 't', 'T')
            && utils::one_of(*transb, 'n', 'N', 't', 'T')) {
        bool trA = *transa == 't' || *transa == 'T';
        bool trB = *transb == 't' || *transb == 'T';
        CBLAS_TRANSPOSE Cblas_trA = trA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE Cblas_trB = trB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasColMajor, Cblas_trA, Cblas_trB, *M, *N, *K, *alpha, A,
                *lda, B, *ldb, *beta, C, *ldc);
        if (bias) {
            // Add bias if necessary (bias is applied to columns of C)
            int incx = 1, incy = 1;
            parallel_nd(*N, [&](int n) {
                ptrdiff_t offset = (ptrdiff_t)n * (*ldc);
                cblas_saxpy(*M, 1.0, bias, incx, C + offset, incy);
            });
        }
        msan_unpoison_matrix(C, *M, *N, *ldc, sizeof(*C));
        return mkldnn_success;
    }
#endif

    if (mayiuse(sse42)) {
        float *dummy_ao = NULL;
        float *dummy_bo = NULL;
        status = gemm_driver(transa, transb, bias ? "C" : NULL, M, N, K, alpha,
                A, lda, dummy_ao, B, ldb, dummy_bo, beta, C, ldc, bias,
                force_jit_nocopy_gemm);
    } else {
        status = ref_gemm<float>(transa, transb, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc, bias);
    }
    return status;
}

// Tries calling Intel MKL cblas_gemm_s8u8s32 if applicable and available
mkldnn_status_t try_cblas_gemm_s8u8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const uint8_t *B, const int *LDB, const uint8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {
#if USE_MKL_IGEMM
    // cblas_gemm_s8u8s32 uses `+` to apply offsets,
    // hence we need to inverse ao and b0.
    if (*ao == -128 || *bo > 128) return mkldnn_unimplemented;

    assert(-127 <= *ao && *ao <= 127);
    assert(*bo <= 128);

    int8_t ao_s8 = -(*ao);
    int8_t bo_s8 = (int8_t)(-(int32_t)*bo);

    bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
    bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
    bool AisN = (*transa == 'N' || *transa == 'n');
    bool BisN = (*transb == 'N' || *transb == 'n');

    CBLAS_TRANSPOSE Cblas_trA = AisN ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE Cblas_trB = BisN ? CblasNoTrans : CblasTrans;
    CBLAS_OFFSET Cblas_offsetc = OCisR
            ? CblasRowOffset
            : (OCisC ? CblasColOffset : CblasFixOffset);
    cblas_gemm_s8u8s32(CblasColMajor, Cblas_trA, Cblas_trB, Cblas_offsetc, *M,
            *N, *K, *alpha, A, *LDA, ao_s8, B, *LDB, bo_s8, *beta, C, *LDC, co);
    msan_unpoison_matrix(C, *M, *N, *LDC, sizeof(*C));
    return mkldnn_success;
#else
    return mkldnn_unimplemented;
#endif
}

template <>
mkldnn_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const uint8_t *B, const int *LDB, const uint8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {
    mkldnn_status_t status = check_gemm_x8x8x32_input(offsetc, transa, transb, M,
            N, K, A, LDA, B, LDB, C, LDC, alpha, beta, false);
    if (status != mkldnn_success) return status;

    if (*M == 0 || *N == 0 || *K == 0) return mkldnn_success;

    status = try_cblas_gemm_s8u8s32(transa, transb, offsetc, M, N, K, alpha, A,
            LDA, ao, B, LDB, bo, beta, C, LDC, co);
    if (status == mkldnn_success) return status;

    if (mayiuse(sse42) && !mayiuse(avx512_mic))
        status = gemm_driver(transa, transb, offsetc, M, N, K, alpha, A, LDA,
                ao, B, LDB, bo, beta, C, LDC, co, false);
    else
        status = ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K, alpha, A,
                LDA, ao, B, LDB, bo, beta, C, LDC, co);

    return status;
}

template <>
mkldnn_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const int8_t *B, const int *LDB, const int8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {
    mkldnn_status_t status = check_gemm_x8x8x32_input(offsetc, transa, transb, M,
            N, K, A, LDA, B, LDB, C, LDC, alpha, beta, false);
    if (status != mkldnn_success) return status;

    if (*M == 0 || *N == 0 || *K == 0) return mkldnn_success;

    bool use_jit = mayiuse(avx512_core);
    bool use_s8u8 = true
            && utils::everyone_is(0, *ao, *bo) // so far a requirement
            && IMPLICATION(USE_MKL_IGEMM == 0, mayiuse(sse42));

    if (use_jit)
        status = gemm_driver(transa, transb, offsetc, M, N, K, alpha, A, LDA,
                ao, B, LDB, bo, beta, C, LDC, co, false);
    else if (use_s8u8)
        status = simple_gemm_s8s8s32(transa, transb, offsetc, M, N, K, alpha, A,
                LDA, ao, B, LDB, bo, beta, C, LDC, co);
    else
        status = ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K, alpha, A,
                LDA, ao, B, LDB, bo, beta, C, LDC, co);

    return status;
}

mkldnn_status_t gemm_bf16bf16f32(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const mkldnn_bfloat16_t *A, const int *lda, const mkldnn_bfloat16_t *B,
        const int *ldb, const float *beta, float *C, const int *ldc) {
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K, A, lda, B,
            ldb, C, ldc, alpha, beta, false);
    if (status != mkldnn_success) return status;

    char *dummyOffsetC = NULL;
    mkldnn_bfloat16_t *dummy_ao = NULL;
    mkldnn_bfloat16_t *dummy_bo = NULL;
    float *dummy_co = NULL;

    if (mayiuse(avx512_core)) {
        return gemm_driver(transa, transb, dummyOffsetC, M, N, K, alpha,
                (const mkldnn_bfloat16_t *)A, lda, dummy_ao, (const mkldnn_bfloat16_t *)B,
                ldb, dummy_bo, beta, (float *)C, ldc, dummy_co, false);
    } else {
        return mkldnn_unimplemented;
    }
}

} // namespace cpu
} // namespace impl
} // namespace mkldnn

using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;

mkldnn_status_t mkldnn_sgemm(char transa, char transb, int M, int N,
        int K, float alpha, const float *A, int lda, const float *B,
        const int ldb, float beta, float *C, int ldc) {
    int M_s32 = (int)M;
    int N_s32 = (int)N;
    int K_s32 = (int)K;
    int lda_s32 = (int)lda;
    int ldb_s32 = (int)ldb;
    int ldc_s32 = (int)ldc;
    return extended_sgemm(&transb, &transa, &N_s32, &M_s32, &K_s32, &alpha, B,
            &ldb_s32, A, &lda_s32, &beta, C, &ldc_s32);
}

namespace {
const char *c2f_offsetC(const char *offC) {
    if (offC) {
        if (offC[0] == 'R' || offC[0] == 'r') return "C";
        if (offC[0] == 'C' || offC[0] == 'c') return "R";
    }
    return offC;
}
} // namespace

mkldnn_status_t mkldnn_gemm_u8s8s32(char transa, char transb, char offsetc,
        int M, int N, int K, float alpha, const uint8_t *A,
        int lda, uint8_t ao, const int8_t *B, int ldb, int8_t bo,
        float beta, int32_t *C, int ldc, const int32_t *co) {
    int M_s32 = (int)M;
    int N_s32 = (int)N;
    int K_s32 = (int)K;
    int lda_s32 = (int)lda;
    int ldb_s32 = (int)ldb;
    int ldc_s32 = (int)ldc;

    return gemm_s8x8s32(&transb, &transa, c2f_offsetC(&offsetc), &N_s32, &M_s32,
            &K_s32, &alpha, B, &ldb_s32, &bo, A, &lda_s32, &ao, &beta, C,
            &ldc_s32, co);
}

mkldnn_status_t mkldnn_gemm_s8s8s32(char transa, char transb, char offsetc,
        int M, int N, int K, float alpha, const int8_t *A,
        int lda, int8_t ao, const int8_t *B, int ldb, int8_t bo,
        float beta, int32_t *C, int ldc, const int32_t *co) {
    int M_s32 = (int)M;
    int N_s32 = (int)N;
    int K_s32 = (int)K;
    int lda_s32 = (int)lda;
    int ldb_s32 = (int)ldb;
    int ldc_s32 = (int)ldc;

    return gemm_s8x8s32<int8_t>(&transb, &transa, c2f_offsetC(&offsetc), &N_s32,
            &M_s32, &K_s32, &alpha, B, &ldb_s32, &bo, A, &lda_s32, &ao, &beta,
            C, &ldc_s32, co);
}

extern "C" {
mkldnn_status_t MKLDNN_API mkldnn_gemm_bf16bf16f32(char transa, char transb,
        int M, int N, int K, float alpha,
        const mkldnn_bfloat16_t *A, int lda, const mkldnn_bfloat16_t *B,
        int ldb, float beta, float *C, int ldc) {
    int M_s32 = (int)M;
    int N_s32 = (int)N;
    int K_s32 = (int)K;
    int lda_s32 = (int)lda;
    int ldb_s32 = (int)ldb;
    int ldc_s32 = (int)ldc;

    return gemm_bf16bf16f32(&transb, &transa, &N_s32, &M_s32, &K_s32, &alpha, B,
            &ldb_s32, A, &lda_s32, &beta, C, &ldc_s32);
}
}
