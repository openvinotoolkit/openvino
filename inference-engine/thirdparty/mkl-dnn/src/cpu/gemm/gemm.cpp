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

#include "jit_generator.hpp"

#include "gemm.hpp"

#include "f32/jit_avx512_common_gemm_f32.hpp"
#include "f32/jit_avx_gemm_f32.hpp"
#include "f32/ref_gemm_f32.hpp"

#include "gemm_driver.hpp"
#include "s8x8s32/ref_gemm_s8x8s32.hpp"
#include "s8x8s32/simple_gemm_s8s8s32.hpp"

#include "os_blas.hpp"

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

mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const int *lda,
        const int *ldb, const int *ldc, const float *alpha, const float *beta,
        const bool with_bias) {
    if (utils::any_null(transa, transb, M, N, K, lda, ldb, ldc, alpha, beta))
        return mkldnn_invalid_arguments;
    if (with_bias && *beta != 0)
        return mkldnn_unimplemented;
    bool consistency = true
        && utils::one_of(*transa, 'T', 't', 'N', 'n')
        && utils::one_of(*transb, 'T', 't', 'N', 'n')
        && *M >= 0
        && *N >= 0
        && *K >= 0;

    if (!consistency)
        return mkldnn_invalid_arguments;
    bool isTransA = utils::one_of(*transa, 'T', 't');
    bool isTransB = utils::one_of(*transb, 'T', 't');
    int nrowA = isTransA ? *K : *M;
    int nrowB = isTransB ? *N : *K;
    consistency = true
        && *lda >= nstl::max(1, nrowA)
        && *ldb >= nstl::max(1, nrowB)
        && *ldc >= nstl::max(1, *M);
    if (!consistency)
        return mkldnn_invalid_arguments;

    return mkldnn_success;
}

mkldnn_status_t check_gemm_x8x8x32_input(const char *offsetc,
        const char *transa, const char *transb, const int *M, const int *N,
        const int *K, const int *lda, const int *ldb, const int *ldc,
        const float *alpha, const float *beta, const bool with_bias) {
    if (offsetc == nullptr)
        return mkldnn_invalid_arguments;
    if (!utils::one_of(*offsetc, 'F', 'f', 'C', 'c', 'R', 'r'))
        return mkldnn_invalid_arguments;

    return check_gemm_input(transa, transb, M, N, K, lda, ldb, ldc, alpha,
        beta, with_bias);
}

mkldnn_status_t extended_sgemm(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const float *A, const int *lda, const float *B, const int *ldb,
        const float *beta, float *C, const int *ldc,
        const float *bias, const bool force_jit_nocopy_gemm) {
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K,
            lda, ldb, ldc, alpha, beta, bias != nullptr);
    if (status != mkldnn_success)
        return status;

#ifdef USE_CBLAS
    if (!force_jit_nocopy_gemm) {
        bool trA = *transa == 't' || *transa == 'T';
        bool trB = *transb == 't' || *transb == 'T';
        CBLAS_TRANSPOSE Cblas_trA = trA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE Cblas_trB = trB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasColMajor, Cblas_trA, Cblas_trB,
                *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);

        if (bias) {
            // Add bias if necessary (bias is applied to columns of C)
            cblas_int incx = 1, incy = 1;
            parallel_nd(*N, [&](int n) {
                ptrdiff_t offset = (ptrdiff_t)n * (*ldc);
                cblas_saxpy(*M, 1.0, bias, incx, C + offset, incy);
            });
        }
        return mkldnn_success;
    }
#endif

    if (mayiuse(avx512_mic)) {
        return jit_avx512_common_gemm_f32(transa, transb,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, bias);
    } else if (mayiuse(avx)) {
        float *dummy_ao = NULL;
        float *dummy_bo = NULL;

        return gemm_driver(transa, transb, bias ? "C" : NULL, M, N, K, alpha,
                A, lda, dummy_ao, B, ldb, dummy_bo, beta, C, ldc, bias,
                force_jit_nocopy_gemm);
    } else {
        return ref_gemm<float>(transa, transb,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, bias);
    }
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

#if USE_MKL_IGEMM
        bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
        bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
        bool AisN = (*transa == 'N' || *transa == 'n');
        bool BisN = (*transb == 'N' || *transb == 'n');

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
                *M, *N, *K, *alpha, A, *LDA, *ao, (uint8_t *)B, *LDB, *bo,
                *beta, C, *LDC, co);
        return mkldnn_success;
    } else {
        assert(data_traits<b_dt>::data_type == data_type::s8);
        // TODO CBLAS implementation of gemm_s8s8s32 goes here.
        // mkldnn_gemm_s8s8s32 doesn't support non-zero ao and bo
        if (utils::everyone_is(0, *ao, *bo)) {
            return simple_gemm_s8s8s32(transa, transb, offsetc, M,
                    N, K, alpha, A, LDA, ao, (int8_t *)B, LDB, bo, beta,
                    C, LDC, co);
        } else {
            return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K,
                    alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
        }
    }
#else
    cpu_isa_t isa = isa_any;
    if (mayiuse(avx512_core_vnni)) {
        isa = avx512_core_vnni;
    } else if (mayiuse(avx512_core)) {
        isa = avx512_core;
    }

    if (data_traits<b_dt>::data_type == data_type::u8) {
        switch (isa) {
        case avx512_core:
        case avx512_core_vnni:
            return gemm_driver(transa, transb, offsetc, M,
                    N, K, alpha, A, LDA, ao, (uint8_t *)B, LDB, bo, beta,
                    C, LDC, co, false);
        default:
            return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K,
                    alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
        }
    } else {
        assert(data_traits<b_dt>::data_type == data_type::s8);
        // mkldnn_gemm_s8s8s32 doesn't support non-zero ao and bo
        if ((mayiuse(avx512_core) || mayiuse(avx512_core_vnni))
                && *ao == 0 && *bo == 0) {
            return simple_gemm_s8s8s32(transa, transb, offsetc, M,
                    N, K, alpha, A, LDA, ao, (int8_t *)B, LDB, bo, beta,
                    C, LDC, co);
        } else {
            return ref_gemm_s8x8s32(transa, transb, offsetc, M, N, K,
                    alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
        }
    }
#endif
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
        int32_t *C, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, C, ldc, co);
}

mkldnn_status_t mkldnn_gemm_s8s8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *lda, const int8_t *ao,
        const int8_t *B, const int *ldb, const int8_t *bo, const float *beta,
        int32_t *C, const int *ldc, const int32_t *co) {
    return gemm_s8x8s32(
        transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo,
        beta, C, ldc, co);
}

mkldnn_status_t mkldnn_gemm_bf16bf16f32(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const float *alpha,
        const mkldnn_bfloat16_t *A, const int *lda, const mkldnn_bfloat16_t *B,
        const int *ldb, const float *beta, float *C, const int *ldc) {
    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K, lda,
            ldb, ldc, alpha, beta, false);
    if (status != mkldnn_success)
        return status;

    char *dummyOffsetC = NULL;
    mkldnn_bfloat16_t *dummy_ao = NULL;
    mkldnn_bfloat16_t *dummy_bo = NULL;
    float *dummy_co = NULL;

    if (mayiuse(avx512_core)) {
        return gemm_driver(transa, transb, dummyOffsetC, M, N, K,
                alpha, A, lda, dummy_ao, B, ldb, dummy_bo, beta, C, ldc,
                dummy_co, false);
    } else {
        return mkldnn_unimplemented;
    }
}
