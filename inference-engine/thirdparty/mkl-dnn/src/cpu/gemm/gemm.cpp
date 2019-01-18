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
#include <mutex>

#include "mkldnn.h"

#include "verbose.hpp"

#include "jit_avx_gemm_f32.hpp"
#include "jit_avx512_common_gemm_f32.hpp"
#include "gemm.hpp"
#include "../jit_generator.hpp"
#include "nstl.hpp"
#include "os_blas.hpp"
#include "math_utils.hpp"
#include "mkldnn_traits.hpp"

/* USE_MKL      USE_CBLAS       effect
 * -------      ---------       ------
 * yes          yes             use Intel(R) MKL CBLAS
 * yes          no              use jit
 * no           yes             system-dependent CBLAS
 * no           no              use jit
 */

namespace mkldnn {
namespace impl {
namespace cpu {
using namespace mkldnn::impl::status;
mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const int *lda,
        const int *ldb, const int *ldc, const float *alpha, const float *beta,
        const bool with_bias) {
    if (utils::any_null(transa, transb, M, N, K, lda, ldb, ldc, alpha, beta))
        return invalid_arguments;
    if (with_bias && *beta != 0)
        return unimplemented;
    bool consistency = true
        && utils::one_of(*transa, 'T', 't', 'N', 'n')
        && utils::one_of(*transb, 'T', 't', 'N', 'n')
        && *M >= 0
        && *N >= 0
        && *K >= 0;

    if (!consistency) return invalid_arguments;
    bool isTransA = utils::one_of(*transa, 'T', 't');
    bool isTransB = utils::one_of(*transb, 'T', 't');
    int nrowA = isTransA ? *K : *M;
    int nrowB = isTransB ? *N : *K;
    consistency = true
        && *lda >= nstl::max(1, nrowA)
        && *ldb >= nstl::max(1, nrowB)
        && *ldc >= nstl::max(1, *M);
    if (!consistency) return invalid_arguments;

    return success;
}

mkldnn_status_t check_gemm_x8x8x32_input(const char *offsetc,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const int *ldc,
        const float *alpha, const float *beta, const bool with_bias) {

    if (offsetc == nullptr) return invalid_arguments;
    if (!utils::one_of(*offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return invalid_arguments;

    return check_gemm_input(transa, transb, M, N, K, lda, ldb, ldc, alpha,
        beta, with_bias);
}

struct gemm_impl_t {
    gemm_impl_t(char transa, char transb, bool zero_beta, bool with_bias) {
        //jit kernel has three codepaths: beta is 0, 1 or arbitrary
        //we will generate kernel for 0 and arbitrary beta
        float zero = 0.0f, arbitrary_float = 2.0f;
        if (mayiuse(avx512_common)) {
            isa_ = avx512_common;
            ker_ = (void *)new jit_avx512_common_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float,
                    with_bias);
        }
        else if (mayiuse(avx)) {
            isa_ = avx;
            ker_ = (void *)new jit_avx_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float,
                    with_bias);
        }
    }

    mkldnn_status_t call(const char *transa, const char *transb, const int *M,
            const int *N, const int *K, const float *alpha, const float *A,
            const int *lda, const float *B, const int *ldb, const float *beta,
            float *C, const int *ldc, const float *bias = nullptr) {
        switch (isa_) {
            case avx:
                ((jit_avx_gemm_f32*)ker_)->sgemm(transa, transb, M, N, K,
                    alpha, A, lda, B, ldb, beta, C, ldc, bias);
                break;
            case avx512_common:
                ((jit_avx512_common_gemm_f32*)ker_)->sgemm(transa, transb,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, bias);
                break;
            default:
                ref_gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta,
                        C, ldc, bias);
                break;
        }
        return mkldnn_success;
    }

    void *ker_;
    cpu_isa_t isa_;
};
//Gemm implementations for: zero/nonzero beta, transA, transB
static gemm_impl_t *gemm_impl[2][2][2];
//Gemm with bias implementations for: transA, transB
//Gemm with bias for beta!=0. is not supported
static gemm_impl_t *gemm_bias_impl[2][2];

void initialize() {
    for (int i = 0; i < 2; ++i) {
        gemm_impl[i][0][0] = new gemm_impl_t('n', 'n', (bool)i, false);
        gemm_impl[i][0][1] = new gemm_impl_t('n', 't', (bool)i, false);
        gemm_impl[i][1][0] = new gemm_impl_t('t', 'n', (bool)i, false);
        gemm_impl[i][1][1] = new gemm_impl_t('t', 't', (bool)i, false);
    }
    gemm_bias_impl[0][0] = new gemm_impl_t('n', 'n', true, true);
    gemm_bias_impl[0][1] = new gemm_impl_t('n', 't', true, true);
    gemm_bias_impl[1][0] = new gemm_impl_t('t', 'n', true, true);
    gemm_bias_impl[1][1] = new gemm_impl_t('t', 't', true, true);
}

mkldnn_status_t extended_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc,
        const float *bias, const bool force_jit_gemm) {
    //Check input
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K,
            lda, ldb, ldc, alpha, beta, bias != nullptr);
    if (status != mkldnn_success)
        return status;
    if (*M == 0 || *N == 0 || *K == 0)
        return mkldnn_success;
    int trA = *transa == 't' || *transa == 'T';
    int trB = *transb == 't' || *transb == 'T';
#ifdef USE_CBLAS
    if (!force_jit_gemm) {
        //Call cblas
        CBLAS_TRANSPOSE Cblas_trA = trA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE Cblas_trB = trB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasColMajor, Cblas_trA, Cblas_trB,
                *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
        //Add bias if necessary (bias is applied to columns of C)
        if (bias) {
            cblas_int incx = 1, incy = 1;
            parallel_nd(*N, [&](int n) {
                cblas_saxpy(*M, 1.0, bias, incx, C + n*(*ldc), incy);
            });
        }
        return mkldnn_success;
    }
#endif
    //Generate jit kernel and call sgemm with bias
    volatile static int initialized = 0;
    if (!initialized) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized) {
            mkldnn::impl::cpu::initialize();
            initialized = 1;
        }
    }
    if (bias)
        gemm_bias_impl[trA][trB]->call(
                transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                bias);
    else
        gemm_impl[*beta == 0.f][trA][trB]->call(
                transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return mkldnn_success;
}

template <typename b_dt>
mkldnn_status_t gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const b_dt *B, const int *LDB, const int8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {

    mkldnn_status_t status = check_gemm_x8x8x32_input(offsetc, transa, transb,
        M, N, K, LDA, LDB, LDC, alpha, beta, false);

    if (status != mkldnn_success)
        return status;

    if (*M == 0 || *N == 0 || *K == 0)
        return mkldnn_success;

    bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
    bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
    bool AisN = (*transa == 'N' || *transa == 'n');
    bool BisN = (*transb == 'N' || *transb == 'n');

#if defined(USE_MKL) && defined(USE_CBLAS)
    if (data_traits<b_dt>::data_type == data_type::u8) {
        CBLAS_TRANSPOSE Cblas_trA = AisN ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE Cblas_trB = BisN ? CblasNoTrans : CblasTrans;
        CBLAS_OFFSET Cblas_offsetc =
            OCisR
            ? CblasRowOffset
            : OCisC
            ? CblasColOffset
            : CblasFixOffset;
        cblas_gemm_s8u8s32(CblasColMajor, Cblas_trA, Cblas_trB, Cblas_offsetc,
            *M, *N, *K, *alpha, A, *LDA, *ao, (b_dt*)B, *LDB, *bo, *beta, C, *LDC, co);
        return mkldnn_success;
    }
#endif
    int m = *M, n = *N, k = *K, lda = *LDA, ldb = *LDB, ldc = *LDC;
    size_t sizeA = AisN ? lda * k : lda * m;
    size_t sizeB = BisN ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;

    double *dA = (double *)malloc(sizeA * sizeof(double), PAGE_4K);
    double *dB = (double *)malloc(sizeB * sizeof(double), PAGE_4K);
    double *dC = (double *)malloc(sizeC * sizeof(double), PAGE_4K);

    if (utils::any_null(dA, dB, dC)) {
        free(dA);
        free(dB);
        free(dC);
        return mkldnn_out_of_memory;
    }

    auto da_setter = [=] (int i, int j, double v) { dA[j * lda + i] = v; };
    auto db_setter = [=] (int i, int j, double v) { dB[j * ldb + i] = v; };

    auto ia_accessor = [=] (int i, int j) { return A[j * lda + i]; };
    auto ib_accessor = [=] (int i, int j) { return B[j * ldb + i]; };

    const int a_rows = AisN ? m : k;
    const int a_cols = AisN ? k : m;
    mkldnn::impl::parallel_nd(a_cols, a_rows, [&](int j, int i) {
        da_setter(i, j,
            static_cast<double>(ia_accessor(i, j)) + static_cast<double>(ao[0]));
    });

    const int b_rows = BisN ? k : n;
    const int b_cols = BisN ? n : k;
    mkldnn::impl::parallel_nd(b_cols, b_rows, [&](int j, int i) {
        db_setter(i, j,
            static_cast<double>(ib_accessor(i, j)) + static_cast<double>(bo[0]));
    });
    double one = 1.0, zero = 0.0;
    ref_gemm<double>(transa, transb, M, N, K, &one, dA, LDA, dB, LDB, &zero,
        dC, LDC, nullptr);

    auto i2d = [=] (int32_t v) { return static_cast<double>(v); };
    auto f2d = [=] (float v)   { return static_cast<double>(v); };

    mkldnn::impl::parallel_nd(n, m, [&] (int j, int i) {
        double coffset = OCisR ? i2d(co[j]) : OCisC ? i2d(co[i]) : i2d(co[0]);
        double val = ((*beta == 0.0f) ? 0.0 : f2d(*beta) * i2d(C[i + j * ldc]))
            + f2d(*alpha) * dC[i + j * ldc] + coffset;
        C[i + j * ldc] = math::out_round<int32_t>(math::saturate<int32_t>(val));
    });

    free(dA);
    free(dB);
    free(dC);
    return mkldnn_success;
}
}
}
}

using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;

mkldnn_status_t mkldnn_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc) {
    return extended_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

mkldnn_status_t mkldnn_gemm_s8u8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const uint8_t *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *c, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, c, ldc, co);
}

mkldnn_status_t mkldnn_gemm_s8s8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const int8_t *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *c, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, c, ldc, co);
}
