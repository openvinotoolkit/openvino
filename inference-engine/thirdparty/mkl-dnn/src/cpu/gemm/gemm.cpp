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

#include "jit_avx2_gemm_f32.hpp"
#include "jit_avx512_common_gemm_f32.hpp"
#include "ref_gemm.hpp"
#include "../jit_generator.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
using namespace mkldnn::impl::status;
mkldnn_status_t check_gemm_input(const char *transa, const char *transb,
        const int *M, const int *N, const int *K, const int *lda,
        const int *ldb, const int *ldc) {
    if (utils::any_null(transa, transb, M, N, K, lda, ldb, ldc))
        return invalid_arguments;
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
struct gemm_impl_t {
    gemm_impl_t(char transa, char transb, bool zero_beta) {
        //jit kernel has three codepaths: beta is 0, 1 or arbitrary
        //we will generate kernel for 0 and arbitrary beta
        float zero = 0.0f, arbitrary_float = 2.0f;
        if (mayiuse(avx512_common)) {
            isa_ = avx512_common;
            ker_ = (void *)new jit_avx512_common_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float);
        }
        else if (mayiuse(avx2)) {
            isa_ = avx2;
            ker_ = (void *)new jit_avx2_gemm_f32(
                    transa, transb, zero_beta ? zero : arbitrary_float);
        }
    }

    mkldnn_status_t call(const char *transa, const char *transb, const int *M,
            const int *N, const int *K, const float *alpha, const float *A,
            const int *lda, const float *B, const int *ldb, const float *beta,
            float *C, const int *ldc) {
        switch (isa_) {
            case avx2:
                ((jit_avx2_gemm_f32*)ker_)->sgemm(transa, transb, M, N, K,
                    alpha, A, lda, B, ldb, beta, C, ldc);
                break;
            case avx512_common:
                ((jit_avx512_common_gemm_f32*)ker_)->sgemm(transa, transb,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                break;
            default:
                ref_gemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C,
                        ldc);
                break;
        }
        return mkldnn_success;
    }

    void *ker_;
    cpu_isa_t isa_;
};
static gemm_impl_t *gemm_impl[2][2][2];

void initialize() {
    for (int i = 0; i < 2; ++i) {
        gemm_impl[i][0][0] = new gemm_impl_t('n', 'n', (bool)i);
        gemm_impl[i][0][1] = new gemm_impl_t('n', 't', (bool)i);
        gemm_impl[i][1][0] = new gemm_impl_t('t', 'n', (bool)i);
        gemm_impl[i][1][1] = new gemm_impl_t('t', 't', (bool)i);
    }
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
    volatile static int initialized = 0;

    mkldnn_status_t status = check_gemm_input(transa, transb, M, N, K,
            lda, ldb, ldc);
    if (status != mkldnn_success)
        return status;
    if (*M == 0 || *N == 0 || *K == 0)
        return mkldnn_success;

    if (!initialized) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized) {
            mkldnn::impl::cpu::initialize();
            initialized = 1;
        }
    }

    int trA = *transa == 't' || *transa == 'T';
    int trB = *transb == 't' || *transb == 'T';
    return gemm_impl[*beta == 0.f][trA][trB]->call(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
