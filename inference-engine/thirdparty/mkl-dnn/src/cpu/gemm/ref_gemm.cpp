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

#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "../jit_generator.hpp"

#include "gemm_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;

constexpr int unroll_m = 16;
constexpr int unroll_n = 6;
static void copy_A(
        bool isTransA, int K, const float *A, const int lda, float *ws) {
    for (int k = 0; k < K; k++) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < unroll_m; i++) {
            ws[i] = isTransA ? A[i * lda + k] : A[i + k * lda];
        }
        ws += unroll_m;
    }
}

template <bool isTransA, bool isTransB>
static void kernel_mxn(int K, const float *A, const int lda,
        const float *B, const int ldb, float *C, const int ldc,
        const float alpha, const float beta) {
    float c[unroll_m * unroll_n] = { 0. };
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < unroll_n; j++) {
            float b = isTransB ? B[j + k * ldb] : B[k + j * ldb];
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < unroll_m; i++) {
                float a = isTransA ? A[i * lda + k] : A[i + lda * k];
                c[i + unroll_m * j] += a * b;
            }
        }
    }
    for (int j = 0; j < unroll_n; j++) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < unroll_m; i++) {
            C[i + j * ldc] = (beta == 0.0f)
            ? alpha * c[i + unroll_m * j]
            : alpha * c[i + unroll_m * j] + beta * C[i + j * ldc];
        }
    }
}

template <bool isTransA, bool isTransB>
static void block_ker(const int M, const int N, const int K,
        const float *A, const int lda, const float *B, const int ldb, float *C,
        const int ldc, const float alpha, const float beta, float *ws,
        bool do_copy) {
    int Nu = rnd_dn(N, unroll_n), Mu = rnd_dn(M, unroll_m);
    for (int i = 0; i < Mu; i += unroll_m) {
        for (int j = 0; j < Nu; j += unroll_n) {
            const float *b = isTransB ? &B[j] : &B[j * ldb];
            const float *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) {
                    copy_A(isTransA, K, a, lda, ws);
                }
                kernel_mxn<false, isTransB>(
                        K, ws, unroll_m, b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            } else {
                kernel_mxn<isTransA, isTransB>(
                        K, a, lda, b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            }
        }
    }
    // tail processing
    for (int i = 0; i < M; i++) {
        for (int j = Nu; j < N; j++) {
            float c = beta == 0.0f ? 0.0f : beta * C[i + j * ldc];
            for (int p = 0; p < K; p++) {
                float b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                float a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (int i = Mu; i < M; i++) {
        for (int j = 0; j < Nu; j++) {
            float c = beta == 0.0f ? 0.0f : beta * C[i + j * ldc];
            for (int p = 0; p < K; p++) {
                float b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                float a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

template <bool isTransA, bool isTransB>
void gemm_ithr(const int M, const int N, const int K, const float alpha,
        const float *A, const int lda, const float *B, const int ldb,
        const float beta, float *C, const int ldc, bool do_copy, float *ws) {
    int BM = 4032;
    int BN = isTransA ? 96 : 48;
    int BK = isTransB ? 96 : 256;
    const float *curA, *curB;
    float *curC;

    if ((M <= 0) || (N <= 0))
        return;

    if ((K <= 0) || (alpha == 0.0f)) {
        if (beta == 0.0f) {
            for (int j = 0; j < N * M; j++)
                C[j] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < N * M; j++)
                C[j] *= beta;
        }
        return;
    }

    for (int Bk = 0; Bk < K; Bk += BK) {
        int kb = nstl::min(K - Bk, BK);
        for (int Bm = 0; Bm < M; Bm += BM) {
            int mb = nstl::min(M - Bm, BM);
            for (int Bn = 0; Bn < N; Bn += BN) {
                int nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;
                if (Bk == 0) {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, beta, ws, do_copy);
                } else {
                    block_ker<isTransA, isTransB>(mb, nb, kb, curA, lda, curB,
                            ldb, curC, ldc, alpha, 1.0f, ws, do_copy);
                }
            }
        }
    }
}

void ref_gemm(const char *transa_, const char *transb_, const int *M_,
        const int *N_, const int *K_, const float *alpha_, const float *A,
        const int *lda_, const float *B, const int *ldb_, const float *beta_,
        float *C, const int *ldc_, const float *bias) {
    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');
    const int M = *M_, N = *N_, K = *K_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const float alpha = *alpha_, beta = *beta_;

    int max_nthr = mkldnn_in_parallel() ? 1 : mkldnn_get_max_threads();
    int nthr_m, nthr_n, nthr_k;
    int MB, NB, KB;
    // thread balancing over M, N, K & size of blocking dimensions
    gemm_utils::calc_nthr_nocopy_avx(
            M, N, K, max_nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(utils::implication(!mkldnn_thr_syncable(), nthr_k == 1));

    float *c_buffers = nullptr, *ws_buffers = nullptr;
    if (nthr_k > 1) {
        c_buffers = (float *)malloc(nthr_m * nthr_n * (nthr_k - 1) * MB * NB
                * sizeof(float), PAGE_4K);
        if (!c_buffers) {
            nthr_k = 1;
            KB = K;
        }
    }

    bool do_copy = (NB / unroll_n > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr = nthr_mn * nthr_k;
    const size_t ws_elems_per_thr = K * unroll_m;
    const size_t ws_size_per_thr
            = utils::rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (do_copy) {
        ws_buffers = (float *)malloc(nthr * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers)
            do_copy = false;
    }

    parallel(nthr, [&](const int ithr, const int nthr) {
        int ithr_mn = ithr % nthr_mn;
        int ithr_m = ithr_mn % nthr_m;
        int ithr_n = ithr_mn / nthr_m;
        int ithr_k = ithr / nthr_mn;

        int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

        float *ws = do_copy
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(float)
                : nullptr;

        int m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
                k_from = 0, k_to = 0, myK = 0;
        auto get_thr_block = [&](int &from, int &to, int &myN, int NB, int N,
                int ithr) {
            from = NB * (ithr);
            to = NB * (ithr + 1);
            if (to > N)
                to = N;
            myN = to - from;
        };
        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
        get_thr_block(k_from, k_to, myK, KB, K, ithr_k);

        if (myM > 0 && myN > 0) {
            float myBeta, *myC;
            int ld;
            if (ithr_k == 0) {
                myC = &(C[m_from + n_from * ldc]);
                myBeta = beta;
                ld = ldc;
            } else {
                myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                myBeta = 0.0f;
                ld = MB;
            }
            const float *myA = isTransA
                    ? &(A[k_from + m_from * lda])
                    : &(A[m_from + k_from * lda]);
            const float *myB = isTransB
                    ? &(B[n_from + k_from * ldb])
                    : &(B[k_from + n_from * ldb]);

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr<false, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<false, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr<true, false>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<true, true>(myM, myN, myK, alpha, myA, lda, myB,
                            ldb, myBeta, myC, ld, do_copy, ws);
                }
            }
        }

        if (nthr_k > 1) {
            assert(mkldnn_thr_syncable());
            mkldnn_thr_barrier();

            // sum matrices partitioned along K dimension
            int offset = 0, block = 0;
            gemm_utils::partition_unit_diff(ithr_k, nthr_k, myN, &offset,
                    &block);
            for (int ik = 1; ik < nthr_k; ++ik) {
                float *myC = c_buffers + MB * (NB * (cbase + ik - 1) + offset);
                gemm_utils::sum_two_matrices(myM, block, myC, MB,
                        &C[m_from + (n_from + offset) * ldc], ldc);
            }
        }
    });

    if (bias) {
        parallel_nd(N, M, [&](int i, int j) {
            C[i*ldc + j] += bias[j];
        });
    }

    free(ws_buffers);
    free(c_buffers);
}
}
}
}
