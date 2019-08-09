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

#include <cstdint>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "gemm_driver.hpp"

#include "f32/gemm_utils_f32.hpp"
#include "f32/jit_avx512_common_gemm_f32.hpp"
#include "f32/jit_avx_gemm_f32.hpp"
#include "gemm_info.hpp"
#include "jit_generator.hpp"
#include "mkldnn_traits.hpp"
#include "mkldnn_types.h"
#include "nstl.hpp"
#include "s8x8s32/gemv.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef struct {
    int nthrs_m, nthrs_n, nthrs_k;
    int partition;
    int copy_type;
} blas_thread_t;

template <typename c_type>
static inline void round_to_nearest(c_type *rounded_val, double fp_val) {
    if (fp_val >= 0.) {
        fp_val += 0.5;
        if (fp_val > INT32_MAX) {
            fp_val = INT32_MAX;
        }
    } else {
        fp_val -= 0.5;
        if (fp_val < INT32_MIN) {
            fp_val = INT32_MIN;
        }
    }
    *rounded_val = (c_type) fp_val;
}

template <typename c_type>
static inline void add_results(const dim_t m, const dim_t n,
        const float alpha, const float beta, const c_type *c_partial_sum,
        const dim_t ldcp, c_type *c_data, const dim_t ldc, const c_type *co,
        const int offsetc) {

    for (dim_t j = 0; j < n; ++j) {
        for (dim_t i = 0; i < m; ++i) {
            c_type ctemp = c_partial_sum[i + j * ldcp];

            if (alpha == 1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = ctemp;
                } else {
                    double c_float = (double) beta
                        * (double) c_data[i + j * ldc];
                    c_float += (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            } else if (alpha == -1.0f) {
                if (beta == 0.0f) {
                    c_data[i + j * ldc] = -ctemp;
                } else {
                    double c_float = (double) beta
                        * (double) c_data[i + j * ldc];
                    c_float -= (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            } else {
                if (beta == 0.0f) {
                    double c_float = alpha * (double) ctemp;
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                } else {
                    double c_float = alpha * (double) ctemp +
                        beta * (double) c_data[i + j * ldc];
                    round_to_nearest(&c_data[i + j * ldc], c_float);
                }
            }

            if (offsetc == FIX_OFFSET) {
                c_data[i + j * ldc] += co[0];
            } else if (offsetc == ROW_OFFSET) {
                c_data[i + j * ldc] += co[j];
            } else if (offsetc == COL_OFFSET) {
                c_data[i + j * ldc] += co[i];
            }
        }
    }
}

// TODO Find a better place for those functions.
template <typename T>
static inline dim_t ld_padd(const dim_t x) {
    return ((x + ((2048 / sizeof(T)) - 1)) / (2048 / sizeof(T)))
        * (2048 / sizeof(T)) +  (64 / sizeof(T));
}

template <typename a_type, typename b_type, typename c_type>
void gemm_kernel(const dim_t m, const dim_t n, const dim_t k,
        const float alpha, const a_type *a, const b_type *b, float beta,
        c_type *c, const dim_t ldc, const c_type *a_row_sum,
        const c_type *b_col_sum, const c_type *co, const int offsetc,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    // Since m and n are limited by blocking, stack overflow may not happen;
    // it's up to 32kB
#if !defined(_MSC_VER)
    c_type col_offset[m];
    c_type row_offset[n];
#else
    c_type *col_offset = (c_type *) _alloca(sizeof(*col_offset) * m);
    c_type *row_offset = (c_type *) _alloca(sizeof(*row_offset) * n);
#endif

    int col_req = 0;
    int row_req = 0;

    if (data_traits<a_type>::data_type == data_type::s8) {
        a_type ao = arg->ao;
        a_type bo = arg->bo;
        c_type co_0 = offsetc == NO_OFFSET ? 0 : co[0];

        if (bo != 0 || offsetc == COL_OFFSET)
            col_req = 1;
        if (ao != 0 || offsetc == ROW_OFFSET)
            row_req = 1;

        // It needs one of column or row offsets, but it doesn't need both
        if ((ao != 0 && bo != 0) || (offsetc == FIX_OFFSET && co_0 != 0)) {
            if (col_req == 0 && row_req == 0) {
                if (m <= n) {
                    col_req = 1;
                } else {
                    row_req = 1;
                }
            }
        }

        if (col_req) {
            for (dim_t i = 0; i < m; i++)
                col_offset[i] = 0;

            if (offsetc == COL_OFFSET) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += co[i];
            }

            if (bo != 0) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += bo * a_row_sum[i];
            }
        }

        if (row_req) {
            for (dim_t i = 0; i < n; i++)
                row_offset[i] = 0;

            if (offsetc == ROW_OFFSET) {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += co[i];
            }

            if (ao != 0) {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += ao * b_col_sum[i];
            }
        }

        if (offsetc == FIX_OFFSET && co_0 != 0) {
            if (col_req) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += co_0;
            } else {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += co_0;
            }
        }

        if (ao != 0 && bo != 0) {
            if (col_req) {
                for (dim_t i = 0; i < m; i++)
                    col_offset[i] += (c_type) k * ao * bo;
            } else {
                for (dim_t i = 0; i < n; i++)
                    row_offset[i] += (c_type) k * ao * bo;
            }
        }
    }

    bool isBeta0 = beta == 0.0f;
    bool isRowOffset = row_req == 1;
    bool isColOffset = col_req == 1;

    /* Column and row offsets are ignored by non-integer compute kernels.
     * Scaling is done only for bfloat16 kernels.
     */
    arg->kernel[isBeta0][isColOffset][isRowOffset](&m, &n, &k, &alpha, a, b,
            c, ldc, col_offset, row_offset);
}

static inline void *align(void *ptr, size_t alignment) {
    return (void *) utils::rnd_up((uintptr_t) ptr, alignment);
}

template <typename scale_t, typename mat_t>
void scale_matrix(dim_t m, dim_t n, scale_t alpha, mat_t * __restrict p_mat,
        dim_t ld) {
    if (data_traits<mat_t>::data_type == data_type::f32) {
        for (dim_t j = 0; j < n; j++) {
            for (dim_t i = 0; i < m; i++) {
                p_mat[i + j * ld] = (mat_t)
                    ((scale_t) p_mat[i + j * ld] * alpha);
            }
        }
    }
}

template <typename a_type, typename b_type, typename c_type>
static mkldnn_status_t gemm_kernel_driver(const dim_t m, const dim_t n,
        const dim_t k, const a_type *a, const b_type *b, c_type *c,
        const c_type *co, const gemm_info_t<a_type, b_type, c_type> *arg) {
    dim_t lda = arg->lda;
    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    float alpha = *arg->alpha;
    float beta = *arg->beta;

    if (m <= 0 || n <= 0) {
        return mkldnn_success;
    }

    bool isInteger = (data_traits<a_type>::data_type == data_type::s8);

    // Scaling C matrix.
    if (!isInteger && beta != 1.0f && beta != 0.0f) {
        scale_matrix(m, n, beta, c, ldc);
        beta = 1.0f;
    }

    // Quick exit for C = beta * C
    if (!isInteger && alpha == 0.0f) {
        if (beta == 0.0f)
            scale_matrix(m, n, beta, c, ldc);

        return mkldnn_success;
    }

    // Padding along K dimension.
    dim_t k_padd = 0;
    if (k <= arg->bk_traditional) {
        k_padd = utils::rnd_up(k, arg->uk);
        k_padd = nstl::max(128LL, k_padd);
    } else if (k < 2 * arg->bk) {
        k_padd = utils::rnd_up((k + 1) / 2, arg->uk);
    } else {
        k_padd = arg->bk;
    }

    // Padding along M dimension.
    dim_t m_padd = utils::rnd_up(nstl::min(nstl::max(m, arg->um), arg->bm),
            arg->um);

    // Padding along N dimension.
    dim_t n_padd = 0;
    if (k < arg->blocking_small_k) {
        n_padd = utils::rnd_up(nstl::min(nstl::max(n, arg->un),
                    arg->bn_small_k), arg->un);
    } else {
        n_padd = utils::rnd_up(nstl::min(nstl::max(n, arg->un), arg->bn),
                arg->un);
    }

    // Padding for temporary buffer for C
    dim_t ldc_buf = ld_padd<c_type>(m_padd);

    dim_t strideAm = (arg->transa == no_trans)? 1 : lda;
    dim_t strideAn = (arg->transa != no_trans)? 1 : lda;
    dim_t strideBm = (arg->transb == no_trans)? 1 : ldb;
    dim_t strideBn = (arg->transb != no_trans)? 1 : ldb;

    size_t a_buf_nelems = m_padd * k_padd;
    size_t b_buf_nelems = k_padd * n_padd;
    size_t a_row_sum_nelems = m_padd;
    size_t b_col_sum_nelems = n_padd;

    size_t mem_size = a_buf_nelems * sizeof(*a) + PAGE_4K
        + b_buf_nelems * sizeof(*b) + PAGE_4K;

    if (isInteger) {
        mem_size += a_row_sum_nelems * sizeof(*c) + PAGE_4K
            + b_col_sum_nelems * sizeof(*c) + PAGE_4K;
    }

    bool need_c_buffer = isInteger &&
        (alpha != 1.0f || (beta != 1 && beta != 0));

    if (need_c_buffer) {
        size_t c_buf_nelems = ldc_buf * n_padd;
        mem_size += c_buf_nelems * sizeof(*c) + PAGE_4K;
    }

    char *mem = (char *) malloc(mem_size, 128);

    if (!mem) {
        return mkldnn_out_of_memory;
    }

    a_type *bufferA = (a_type *) align(mem, PAGE_4K);
    b_type *bufferB = (b_type *) align(bufferA + a_buf_nelems, PAGE_4K);

    c_type *a_row_sum = NULL;
    c_type *b_col_sum = NULL;
    if (isInteger) {
        a_row_sum = (c_type *) align(bufferB + b_buf_nelems, PAGE_4K);
        b_col_sum = (c_type *) align(a_row_sum + a_row_sum_nelems, PAGE_4K);
    }

    c_type *bufferC = NULL;
    if (need_c_buffer) {
        bufferC = (c_type *) align(b_col_sum + b_col_sum_nelems, PAGE_4K);
    }

    float beta_saved = beta;

    int a_block_copied = 0;
    dim_t sizeM = 0;
    for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
        sizeM = m - Bm;
        if (sizeM > m_padd)
            sizeM = m_padd;

        dim_t sizeK = 0;
        for (dim_t Bk = 0; Bk < k; Bk += sizeK) {
            sizeK = k - Bk;
            if (sizeK > k_padd)
                sizeK = k_padd;

            // Scale C blocks by beta only for the first time
            if (Bk == 0)
                beta = beta_saved;
            else
                beta = 1.0f;

            // Apply C offset when to the last k-block of the partial sum.
            int offsetc = NO_OFFSET;
            if (Bk + sizeK == k && isInteger)
                offsetc = arg->offsetc;

            dim_t sizeN = 0;
            for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN > n_padd)
                    sizeN = n_padd;

                const b_type *b_block = b + Bk * strideBm + Bn * strideBn;
                const float one = 1.0f;

                /* Column sum argument is ignored for non-integer kernels and
                 * scaling factor is ignored by 8-bit and 16-bit copy kernels.
                 */
                arg->copyB(&sizeK, &sizeN, b_block, &ldb, &one, bufferB, NULL,
                        NULL, b_col_sum);

                dim_t sizeUM = 0;
                for (dim_t Um = 0; Um < sizeM; Um += sizeUM) {
                    sizeUM = sizeM - Um;
                    if (sizeUM > arg->um)
                        sizeUM = arg->um;

                    /* Use the whole A buffer only if we have multiple B
                     * blocks for k-dimension, otherwise we are wasting cache
                     * to store B and C blocks.
                     */
                    dim_t Um_forA = 0;
                    if (sizeN < n)
                        Um_forA = Um;

                    const a_type *a_block = a + (Bm + Um) * strideAm
                        + Bk * strideAn;
                    if (!a_block_copied) {
                        /* Row sum argument is ignored for non-integer kernels
                         * and scaling factor is ignored by 8-bit and 16-bit
                         * copy kernels.
                         */
                        arg->copyA(&sizeK, &sizeUM, a_block, &lda, &alpha,
                                bufferA + Um_forA * sizeK, NULL, NULL,
                                a_row_sum + Um_forA);
                    }

                    c_type *c_block = c + (Bm + Um) + Bn * ldc;
                    dim_t co_stride = 0;
                    if (isInteger) {
                        if (offsetc == FIX_OFFSET) {
                            co_stride = 0;
                        } else if (offsetc == ROW_OFFSET) {
                            co_stride = Bn;
                        } else if (offsetc == COL_OFFSET) {
                            co_stride = Bm + Um;
                        }
                    }
                    if (need_c_buffer) {
                        gemm_kernel(sizeUM, sizeN, sizeK, 1.0f,
                                bufferA + Um_forA * sizeK, bufferB, 0.0f,
                                bufferC + Um, ldc_buf, a_row_sum + Um_forA,
                                b_col_sum, (c_type *) NULL, NO_OFFSET, arg);

                        /* Finish the block adding the necessary alpha, beta
                         * and offsets.
                         */
                        add_results(sizeUM, sizeN, alpha, beta, bufferC + Um,
                                ldc_buf, c_block, ldc, co + co_stride,
                                offsetc);
                    } else {
                        gemm_kernel(sizeUM, sizeN, sizeK, alpha,
                                bufferA + Um_forA * sizeK, bufferB, beta,
                                c_block, ldc, a_row_sum + Um_forA, b_col_sum,
                                co + co_stride, offsetc, arg);
                    }
                }
                a_block_copied = 1;
            }
            a_block_copied = 0;
        }
    }

    free(mem);

    return mkldnn_success;
}

template <typename a_type, typename b_type, typename c_type>
static mkldnn_status_t kernel_driver_parallel_acopiedbcopy(const dim_t m,
        const dim_t n, const dim_t k, const a_type *bufferA, const b_type *b,
        const float beta, c_type *c, const int offsetc, const c_type *co,
        const c_type *a_row_sum,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    float alpha = *arg->alpha;

    if (m <= 0 || n <= 0) {
        return mkldnn_success;
    }

    // Padding along N dimension.
    dim_t n_padd = 0;
    if (k < arg->blocking_small_k) {
        n_padd = utils::rnd_up(nstl::min(nstl::max(n, arg->un),
                    arg->bn_small_k), arg->un);
    } else {
        n_padd = utils::rnd_up(nstl::min(nstl::max(n, arg->un), arg->bn),
                arg->un);
    }

    // Padding for temporary buffer for C
    dim_t ldc_buf = ld_padd<c_type>(m);

    dim_t strideBn = (arg->transb != 0)? 1 : ldb;

    size_t b_buf_nelems = k * n_padd;
    size_t b_col_sum_nelems = n_padd;

    size_t mem_size = b_buf_nelems * sizeof(*b) + PAGE_4K;

    bool isInteger = data_traits<a_type>::data_type == data_type::s8;

    if (isInteger) {
        mem_size += b_col_sum_nelems * sizeof(*c) + PAGE_4K;
    }

    bool need_c_buffer = isInteger &&
        (alpha != 1.0f || (beta != 1 && beta != 0));

    if (need_c_buffer) {
        size_t c_buf_nelems = ldc_buf * n_padd;
        mem_size += c_buf_nelems * sizeof(*c) + PAGE_4K;
    }

    char *mem = (char *) malloc(mem_size, 128);

    if (!mem) {
        return mkldnn_out_of_memory;
    }

    b_type *bufferB = (b_type *) align(mem, PAGE_4K);

    c_type *b_col_sum = NULL;
    if (isInteger) {
        b_col_sum = (c_type *) align(bufferB + b_buf_nelems, PAGE_4K);
    }

    c_type *bufferC = NULL;
    if (need_c_buffer) {
        bufferC = (c_type *) align(b_col_sum + b_col_sum_nelems, PAGE_4K);
    }

    dim_t sizeN = 0;
    for (dim_t Bn = 0; Bn < n; Bn += sizeN) {
        sizeN = n - Bn;
        if (sizeN > n_padd)
            sizeN = n_padd;

        const b_type *b_block = b + Bn * strideBn;
        const float one = 1.0f;

        /* Column sum argument is ignored for non-integer kernels and scaling
         * factor is ignored by 8-bit and 16-bit copy kernels.
         */
        arg->copyB(&k, &sizeN, b_block, &ldb, &one, bufferB, NULL, NULL,
                b_col_sum);

        dim_t co_stride = 0;
        if (isInteger) {
            if (offsetc == FIX_OFFSET) {
                co_stride = 0;
            } else if (offsetc == ROW_OFFSET) {
                co_stride = Bn;
            } else if (offsetc == COL_OFFSET) {
                co_stride = 0;
            }
        }

        c_type *c_block = c + Bn * ldc;
        if (need_c_buffer) {
            gemm_kernel(m, sizeN, k, 1.0f, bufferA, bufferB, 0.0f, bufferC,
                    ldc_buf, a_row_sum, b_col_sum, (c_type *) NULL, NO_OFFSET,
                    arg);

            // Finish the block adding the necessary alpha, beta and offsets.
            add_results(m, sizeN, alpha, beta, bufferC, ldc_buf, c_block, ldc,
                    co + co_stride, offsetc);
        } else {
            gemm_kernel(m, sizeN, k, alpha, bufferA, bufferB, beta, c_block,
                    ldc, a_row_sum, b_col_sum, co + co_stride, offsetc, arg);
        }
    }

    free(mem);

    return mkldnn_success;

}

static inline int nocopy_checker_avx2(const int nthr, const int transa,
        const int transb, const dim_t m, const dim_t n, const dim_t k,
        const dim_t lda, const dim_t ldb, const dim_t ldc) {
    static const dim_t BM_NOCOPY_AVX2 = 64;
    static const dim_t MN_NOCOPY_AVX2 = 128;
    static const dim_t N_TRANSB_PER_THR = 1;
    static const dim_t K_TRANSB_PER_THR = 1;
    static const dim_t N_NOTRANSB_PER_THR = 16;
    static const dim_t K_NOTRANSB_PER_THR = 2;
    static const double FORCE_NOCOPY_THRESH = 0.0038;

    // Crude threshold to nocopy kernels if copy overhead is significant.
    if (1.0 / m + 1.0 / n >= FORCE_NOCOPY_THRESH) {
        return 1;
    }

    if (m <= 378 && n <= 378 && k >= nthr * 378) return 0;

    if (m >= nthr * 378 && k >= nthr * 378) return 0;

    if (transb == no_trans) {
        if (m <= MN_NOCOPY_AVX2 && n <= MN_NOCOPY_AVX2) return 1;
        if (n <= nthr * N_NOTRANSB_PER_THR) return 1;
        if (k <= nthr * K_NOTRANSB_PER_THR) return 1;
        if (m <= BM_NOCOPY_AVX2 && n >= nthr * N_NOTRANSB_PER_THR) return 1;
    } else {
        if (m <= MN_NOCOPY_AVX2 && n <= MN_NOCOPY_AVX2) return 1;
        if (n <= nthr * N_TRANSB_PER_THR) return 1;
        if (k <= nthr * K_TRANSB_PER_THR) return 1;
    }

    return 0;
}

static inline int nocopy_checker_avx512(int nthr, const int transa,
        const int transb, const dim_t m, const dim_t n, const dim_t k,
        const dim_t lda, const dim_t ldb, const dim_t ldc) {
    // Constants definition
    static const dim_t BAD_LD_MULT = 256;
    static const dim_t M_TRANSB_PER_THR = 28;
    static const dim_t N_TRANSB_PER_THR = 28;
    static const dim_t K_TRANSB_PER_THR = 1;
    static const dim_t MN_NOTRANSB_PER_THR = 28;
    static const dim_t K_NOTRANSB_PER_THR = 1;
    static const double FORCE_NOCOPY_THRESH = 0.00196;

    // Crude threshold to nocopy kernels if copy overhead is significant.
    if (1.0 / m + 1.0 / n >= FORCE_NOCOPY_THRESH) {
        return 1;
    }

    // Do not use no copy kernels on "bad" leading dimensions, which are
    // multiples of 256 if M or N is too small, then skip this leading
    // dimension check (no-copy is still helpful there).
    // For LSTM use cases, seems that for N=16 no-copy is still beneficial
    // with bad leading dimension when K is not too large and A non
    // transpose and M != 4096
    if (m >= 32 &&
        (n > 16 || (n == 16 && (k >= 6400 || transa == 0 || m == 4096))) &&
        (lda % BAD_LD_MULT == 0 || ldb % BAD_LD_MULT == 0
         || ldc % BAD_LD_MULT == 0))
        return 0;

    if (m <= 378 && n <= 378 && k >= nthr * 378) return 0;

    if (m >= nthr * 378 && k >= nthr * 378) return 0;

    if (transb == no_trans) {
        if (m <= nthr * MN_NOTRANSB_PER_THR) return 1;
        if (n <= nthr * MN_NOTRANSB_PER_THR) return 1;
        if (k <= nthr * K_NOTRANSB_PER_THR) return 1;
    } else {
        if (m <= nthr * M_TRANSB_PER_THR && m >= n) return 1;
        if (n <= nthr * N_TRANSB_PER_THR) return 1;
        if (k <= nthr * K_TRANSB_PER_THR) return 1;
    }
    return 0;
}

static inline int nocopy_checker(const int nthr, const int transa,
        const int transb, const dim_t m, const dim_t n, const dim_t k,
        const dim_t lda, const dim_t ldb, const dim_t ldc) {

    if (mayiuse(avx512_core)) {
        return nocopy_checker_avx512(nthr, transa, transb, m, n, k, lda, ldb,
                ldc);
    } else if (mayiuse(avx2)) {
        return nocopy_checker_avx2(nthr, transa, transb, m, n, k, lda, ldb,
                ldc);
    } else {
        return 1;
    }
}


#define N2D_MAX 384
#define M2D_MIN 384
template <typename a_type, typename b_type, typename c_type>
static inline void set_thread_opts(int *p_nthrs, blas_thread_t *thread_info,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    int nthrs = *p_nthrs;
    int transa = arg->transa;
    int transb = arg->transb;
    dim_t m = arg->m;
    dim_t n = arg->n;
    dim_t k = arg->k;
    dim_t lda = arg->lda;
    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    thread_info->nthrs_m = 0;
    thread_info->nthrs_n = 0;
    thread_info->nthrs_k = 0;
    thread_info->copy_type = COPY_NONE; // By default don't do parallel copy.

    bool isInteger = data_traits<a_type>::data_type == data_type::s8;
    bool isSgemm = data_traits<a_type>::data_type == data_type::f32;

    if (isSgemm &&
            nocopy_checker(nthrs, transa, transb, m, n, k, lda, ldb, ldc)) {
        thread_info->copy_type = NO_COPY;
        int nthrs_m = 0;
        int nthrs_n = 0;
        int nthrs_k = 0;
        int BM = 0;
        int BN = 0;
        int BK = 0;

        if (mayiuse(avx512_core)) {
            gemm_utils::calc_nthr_nocopy_avx512_common(
                    (int) m, (int) n, (int) k, nthrs,
                    &nthrs_m, &nthrs_n, &nthrs_k, &BM, &BN, &BK);
        } else {
            gemm_utils::calc_nthr_nocopy_avx(
                    (int) m, (int) n, (int) k, nthrs,
                    &nthrs_m, &nthrs_n, &nthrs_k, &BM, &BN, &BK);
        }

        // Block information is being ignored. We will create patitioning
        // later.

        thread_info->nthrs_m = nthrs_m;
        thread_info->nthrs_n = nthrs_n;
        thread_info->nthrs_k = nthrs_k;

        // Reset the total number of threads that will be used.
        *p_nthrs = nthrs_m * nthrs_n * nthrs_k;
    }

    // TODO Check if we can use dynamic scheduling for sgemm.

    // TODO Check if we should use 3D blocking.

    int condition_2D_bsrc = -1;
    if (isSgemm) {
        // If m is large and n is small then do 1D partitioning for AVX2.
        if (!mayiuse(avx512_core) && n <= N2D_MAX && (m >= nthrs * M2D_MIN)) {
            condition_2D_bsrc = 0;
        } else {
            condition_2D_bsrc = ((n > nthrs * N2D_MAX) ||
                    (n <= nthrs * N2D_MAX / 2)) && (m >= 2 * M2D_MIN);
        }
    } else {
        condition_2D_bsrc = (256 * m > nthrs * n) && (nthrs * m < 256 * n);
    }

    // TODO Check if we shoud use k-partitioning.

    int condition_1D_copya = 0;
    if (mayiuse(avx512_core)) {
        const dim_t thresh = isSgemm ? N2D_MAX / 4 : 68;
        if (m >= 1000 && (n >= nthrs * thresh)) {
            condition_2D_bsrc = 0;
            condition_1D_copya = 1;
        }
    } else { // AVX2 code path
        if (m >= 1000 && n >= 4000) {
            condition_2D_bsrc = 0;
            condition_1D_copya = 1;
        }
    }

    // If A or B offset are non-zero, we need to keep 1D_copya to reduce update
    // overhead for integer case.
    if (isInteger && (arg->ao != 0 || arg->bo != 0)) {
        condition_2D_bsrc = 0;
        condition_1D_copya = 1;
    }

    if (condition_2D_bsrc == 1) {
        int nthrs_m = 1;
        int nthrs_n = nthrs;

        while ((nthrs_n % 2 == 0) &&
                (n / nthrs > N2D_MAX || n / nthrs_n <= N2D_MAX / 2) &&
                (m / nthrs_m >= 2 * M2D_MIN) &&
                (nthrs_m < 4)) {
            nthrs_m *= 2;
            nthrs_n /= 2;
        }

        thread_info->nthrs_m = nthrs_m;
        thread_info->nthrs_n = nthrs_n;
        thread_info->partition = PARTITION_2D;

        // Reset the total number of threads that will be used.
        *p_nthrs = nthrs_m * nthrs_n;

    } else if (condition_1D_copya && mkldnn_thr_syncable()) {
        // Use parallel copy A algorithm
        thread_info->copy_type = COPY_A;
        thread_info->partition = PARTITION_1D_COL;
    } else {
        int veclen = 0;
        if (mayiuse(avx512_core)) {
            veclen = cpu_isa_traits<avx512_core>::vlen / (int) sizeof(c_type);
        } else {
            veclen = cpu_isa_traits<avx2>::vlen / (int) sizeof(c_type);
        }

        if (m > n && (m >= nthrs * veclen || n < nthrs)) {
            thread_info->partition = PARTITION_1D_ROW;
        } else {
            thread_info->partition = PARTITION_1D_COL;
        }
    }
}
#undef N2D_MAX
#undef M2D_MIN

static inline void partition_1d(const int ithr, const int nthrs, const dim_t n,
        dim_t *t_offset, dim_t *t_block) {

    dim_t band = n / nthrs;

    dim_t tail = n - (nthrs - 1) * band;
    if (tail > (band + 1))
        band++;
    tail = n - (nthrs - 1) * band;

    if (ithr < (nthrs - 1))
        *t_block = band;
    else
        *t_block = tail;

    *t_offset = ithr * band;

    if (*t_offset >= n) {
        *t_block = 0;
        *t_offset = 0;
    } else if ((*t_offset + *t_block) > n) {
        *t_block = n - *t_offset;
    }
}

static inline void partition_2d(const int ithr, int *nthrs, const int ithr_i,
        const int ithr_j, const int nthrs_m, const int nthrs_n, const dim_t m,
        const dim_t n, dim_t *p_m_disp, dim_t *p_m_band, dim_t *p_n_disp,
        dim_t *p_n_band) {

    dim_t m_disp = 0, n_disp = 0;
    dim_t m_band = 0, n_band = 0;

    int mdiv = nthrs_m;
    int ndiv = nthrs_n;

    dim_t m_bandt = m / mdiv; /* size per thread */
    dim_t n_bandt = n / ndiv; /* size per thread */
    int firstmgroup = mdiv - 1;
    int firstngroup = ndiv - 1;
    dim_t firstmval = m_bandt;
    dim_t firstnval = n_bandt;

    int mthr_used = mdiv;
    if (m - (mdiv - 1) * m_bandt > m_bandt + 1) {
        if (m - (mdiv - 1) * m_bandt > mdiv)
            ++m_bandt;

        firstmval = m_bandt + 1;
        mthr_used = (int) (m / firstmval);

        if (mthr_used * firstmval < m)
            ++mthr_used;

        firstmgroup = mthr_used - 1;
    }

    int nthr_used = ndiv;
    if (n - (ndiv - 1) * n_bandt > n_bandt + 1) {
        firstnval = n_bandt + 1;
        nthr_used = (int) (n / firstnval);

        if (nthr_used * firstnval < n)
            ++nthr_used;

        firstngroup = nthr_used - 1;
    }

    *nthrs = mthr_used * nthr_used;

    if (ithr < *nthrs) {
        if (ithr_i < firstmgroup) {
            m_band = firstmval;
            m_disp = ithr_i * firstmval;
        } else if (ithr_i <= mthr_used - 2) {
            m_band = m_bandt;
            m_disp = firstmgroup * firstmval + (ithr_i - firstmgroup) * m_bandt;
        } else {
            m_disp = firstmgroup * firstmval
                + (mthr_used - 1 - firstmgroup) * m_bandt;
            m_band = nstl::max(0LL, m - m_disp);
        }

        if (ithr_j < firstngroup) {
            n_band = firstnval;
            n_disp = ithr_j * firstnval;
        } else if (ithr_j <= nthr_used - 2) {
            n_band = n_bandt;
            n_disp = firstngroup * firstnval + (ithr_j - firstngroup) * n_bandt;
        } else {
            n_disp = firstngroup * firstnval
                + (nthr_used - 1 - firstngroup) * n_bandt;
            n_band = nstl::max(0LL, n - n_disp);
        }
        m_disp = nstl::max(nstl::min(m_disp, m - 1), 0LL);
        n_disp = nstl::max(nstl::min(n_disp, n - 1), 0LL);
    }

    if (ithr < *nthrs) {
        *p_m_disp = m_disp;
        *p_n_disp = n_disp;
        *p_m_band = m_band;
        *p_n_band = n_band;
    } else {
        *p_m_disp = 0;
        *p_n_disp = 0;
        *p_m_band = 0;
        *p_n_band = 0;
    }

    return;
}

template <typename a_type, typename b_type, typename c_type>
static inline void decompose_matrices(const int ithr, int *nthrs, dim_t *m,
        dim_t *n, dim_t *k, const a_type **a, const b_type **b, c_type **c,
        const c_type **co, const blas_thread_t *thread_info,
        const gemm_info_t<a_type, b_type, c_type> *arg) {

    dim_t strideAm = (arg->transa == no_trans)? 1 : arg->lda;
    dim_t strideBn = (arg->transb != no_trans)? 1 : arg->ldb;
    int offsetc = arg->offsetc;

    bool isInteger = data_traits<a_type>::data_type == data_type::s8;

    switch (thread_info->partition) {
    case PARTITION_1D_ROW:
        {
            dim_t offset = 0;
            dim_t block = 0;
            partition_1d(ithr, *nthrs, arg->m, &offset, &block);

            *m = block;
            *n = arg->n;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a + offset * strideAm;

            // Set matrix B.
            *b = arg->b;

            // Set matrix C.
            *c = arg->c + offset;

            // Set offset vector for C matrix.
            if (isInteger) {
                dim_t co_stride = 0;
                if (offsetc == FIX_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == ROW_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == COL_OFFSET) {
                    co_stride = offset;
                }
                *co = arg->co + co_stride;
            }
            break;
        }

    case PARTITION_1D_COL:
        {
            dim_t offset = 0;
            dim_t block = 0;
            partition_1d(ithr, *nthrs, arg->n, &offset, &block);

            *m = arg->m;
            *n = block;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a;

            // Set matrix B.
            *b = arg->b + offset * strideBn;

            // Set matrix C.
            *c = arg->c + offset * arg->ldc;

            // Set offset vector for C matrix
            if (isInteger) {
                dim_t co_stride = 0;
                if (offsetc == FIX_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == ROW_OFFSET) {
                    co_stride = offset;
                } else if (offsetc == COL_OFFSET) {
                    co_stride = 0;
                }
                *co = arg->co + co_stride;
            }
            break;
        }

    case PARTITION_2D_COL_MAJOR:
        {
            int nthrs_m = thread_info->nthrs_m;
            int nthrs_n = thread_info->nthrs_n;
            int ithr_i = ithr % nthrs_m;
            int ithr_j = ithr / nthrs_m;

            dim_t m_disp = 0;
            dim_t m_band = 0;
            dim_t n_disp = 0;
            dim_t n_band = 0;

            partition_2d(ithr, nthrs, ithr_i, ithr_j, nthrs_m, nthrs_n,
                    arg->m, arg->n, &m_disp, &m_band, &n_disp, &n_band);

            *m = m_band;
            *n = n_band;
            *k = arg->k;

            // Set matrix A.
            *a = arg->a + m_disp * strideAm;

            // Set matrix B.
            *b = arg->b + n_disp * strideBn;

            // Set matrix C.
            *c = arg->c + m_disp + n_disp * arg->ldc;

            // Set offset vector for C matrix
            if (isInteger) {
                dim_t co_stride = 0;
                if (offsetc == FIX_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == ROW_OFFSET) {
                    co_stride = n_disp;
                } else if (offsetc == COL_OFFSET) {
                    co_stride = m_disp;
                }
                *co = arg->co + co_stride;
            }
            break;
        }
    }
}

#define MULTIPLIER 10
template <typename a_type, typename b_type, typename c_type>
static mkldnn_status_t parallel_a_copy(const int ithr, const int nthrs,
        const dim_t m, const dim_t n, const dim_t k, const a_type *a,
        const b_type *b, c_type *c, const c_type *co,
        const gemm_info_t<a_type, b_type, c_type> *arg, char **p_shared_mem) {

    const dim_t lda = arg->lda;
    const dim_t ldb = arg->ldb;
    const dim_t ldc = arg->ldc;
    const dim_t strideAm = (arg->transa == no_trans)? 1 : lda;
    const dim_t strideAn = (arg->transa != no_trans)? 1 : lda;
    const dim_t strideBm = (arg->transb == no_trans)? 1 : ldb;

    float alpha = *arg->alpha;
    float beta = *arg->beta;

    bool isInteger = (data_traits<a_type>::data_type == data_type::s8);

    // Scaling C matrix.
    if (!isInteger && beta != 1.0f && beta != 0.0f) {
        scale_matrix(m, n, beta, c, ldc);
        beta = 1.0f;
    }

    // Padding along M dimension.
    dim_t m_padd = utils::rnd_up(nstl::min(nstl::max(m, arg->um), arg->bm),
            arg->um);

    // Padding along K dimension.
    dim_t k_padd = 0;
    if (k <= arg->bk_traditional) {
        k_padd = utils::rnd_up(k, arg->uk);
        k_padd = nstl::max(128LL, k_padd);
    } else if (k < 2 * arg->bk) {
        k_padd = utils::rnd_up(k / 2, arg->uk);
    } else {
        k_padd = arg->bk;
    }

    m_padd *= nthrs > MULTIPLIER ? MULTIPLIER : nthrs;
    if (m_padd > m) {
        m_padd = utils::rnd_up(m, arg->um);
    }

    size_t a_buf_nelems = m_padd * k_padd;

    // Allocate shared memory for A and its row sum buffers in master thread.
    if (ithr == 0) { // If thread master
        size_t mem_size = (a_buf_nelems * sizeof(*a) + PAGE_4K);

        if (isInteger) {
            size_t a_row_sum_nelems = m_padd;
            mem_size += a_row_sum_nelems * sizeof(*c) + PAGE_4K;
        }

        *p_shared_mem = (char *) malloc(mem_size, 128);

    }
    mkldnn_thr_barrier();

    char *mem = *p_shared_mem;
    a_type *bufferA = (a_type *) align(mem, PAGE_4K);

    c_type *a_row_sum = NULL;
    if (isInteger) {
        a_row_sum = (c_type *) align(bufferA + a_buf_nelems, PAGE_4K);
    }

    if (!mem) {
        return mkldnn_out_of_memory;
    }

    mkldnn_status_t result = mkldnn_success; // Return status

    float beta_saved = beta;

    dim_t sizeK = 0;
    for (dim_t Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK > k_padd)
            sizeK = k_padd;

        // Scale C blocks by beta only for the first term of partial sum.
        if (Bk == 0)
            beta = beta_saved;
        else
            beta = 1.0f;

        // Apply C offset for the last k-block of the partial sum.
        int offsetc = NO_OFFSET;
        if (Bk + sizeK == k && isInteger)
            offsetc = arg->offsetc;

        dim_t sizeM = 0;
        for (dim_t Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM > m_padd)
                sizeM = m_padd;

            if (ithr < nthrs) {
                dim_t band = (sizeM + nthrs - 1) / nthrs;
                band = utils::rnd_up(band, arg->um);

                dim_t offset = band * ithr;

                // If offset is too large don't use that thread for copying.
                if (offset >= sizeM) {
                    offset = 0;
                    band = 0;
                }

                // Handle the tail of the copy.
                if (offset + band > sizeM) {
                    band = sizeM - offset;
                }

                if (band > 0) {
                    const a_type *a_block = a + (Bm + offset) * strideAm
                        + Bk * strideAn;

                    /* Row sum argument is ignored for non-integer kernels and
                     * scaling factor is ignored by 8-bit and 16-bit copy
                     * kernels.
                     */
                    arg->copyA(&sizeK, &band, a_block, &lda, &alpha,
                            bufferA + offset * sizeK, NULL, NULL,
                            a_row_sum + offset);
                }
            }
            mkldnn_thr_barrier(); // Wait for finishing parallel copy.

            const b_type *b_block = b + Bk * strideBm;
            c_type *c_block = c + Bm;

            dim_t co_stride = 0;
            if (isInteger) {
                if (offsetc == FIX_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == ROW_OFFSET) {
                    co_stride = 0;
                } else if (offsetc == COL_OFFSET) {
                    co_stride = Bm;
                }
            }

            result = kernel_driver_parallel_acopiedbcopy(sizeM, n, sizeK,
                    bufferA, b_block, beta, c_block, offsetc, co + co_stride,
                    a_row_sum, arg);

            mkldnn_thr_barrier(); // Wait for kernel computations to finish.
        }
    }

    // Free memory allocated in master thread
    if (ithr == 0) {
        free(mem);
    }

    return result;
}
#undef MULTIPLIER

template <typename T>
static inline void get_omp_thread_count(dim_t m, dim_t n, dim_t k, int *nthrs) {
    const double omp_overhead_small_core = 3.0e+3;
    const double omp_intercept_big_core = 4.0e+3;
    const double omp_slope_big_core = 5.0e+2;

    int veclen = 0;
    if (mayiuse(avx512_core)) {
        veclen = cpu_isa_traits<avx512_core>::vlen / (int) sizeof(T);
    } else {
        veclen = cpu_isa_traits<avx2>::vlen / (int) sizeof(T);
    }
    const double fp_per_cycle = 2.0 * 2.0 * veclen;

    double gemm_cycles = m * n * k / fp_per_cycle;
    if (data_traits<T>::data_type == data_type::f32) {
        gemm_cycles *= 2.0;
    } else {
        gemm_cycles *= 8.0;
    }

    int i = *nthrs;

    // Use a different model for omp overheads if nthrs is <= 4
    if (*nthrs <= 4 && omp_overhead_small_core > 0) {
        double omp_cycles = omp_overhead_small_core;
        if (gemm_cycles < omp_cycles) {
            *nthrs = 1;
            return;
        } else {
            while (i > 1) {
                if (omp_cycles * i < gemm_cycles * (i - 1)) break;
                --i;
            }
        }
    } else {
        if (gemm_cycles < (omp_intercept_big_core + 2 * omp_slope_big_core)) {
            *nthrs = 1;
            return;
        }

        // adaptive decrement to march faster·
        while (i > 1) {
            double omp_cycles = omp_intercept_big_core + i * omp_slope_big_core;
            if (omp_cycles * i < gemm_cycles * (i - 1))
                break;

            if (i < 10)
                i -= 2;
            else if (i < 30)
                i -= 4;
            else
                i -= 8;
        }
    }

    if (i < 1)
        i = 1;

    *nthrs = i;
}

static mkldnn_status_t call_no_copy_sgemm(const int transa, const int transb,
        const dim_t m, const dim_t n, const dim_t k, const float *alpha,
        const float *a, const dim_t lda, const float *b, const dim_t ldb,
        const float *beta, float *c, dim_t ldc, const float *bias) {
    int m_s32 = (int) m;
    int n_s32 = (int) n;
    int k_s32 = (int) k;
    int lda_s32 = (int) lda;
    int ldb_s32 = (int) ldb;
    int ldc_s32 = (int) ldc;

    if (mayiuse(avx512_core))
        return jit_avx512_common_gemm_f32(
                transa == no_trans ? "N" : "T",
                transb == no_trans ? "N" : "T",
                &m_s32, &n_s32, &k_s32, alpha,
                a, &lda_s32,
                b, &ldb_s32,
                beta, c, &ldc_s32, bias);
    else
        return jit_avx_gemm_f32(
                transa == no_trans ? "N" : "T",
                transb == no_trans ? "N" : "T",
                &m_s32, &n_s32, &k_s32, alpha,
                a, &lda_s32,
                b, &ldb_s32,
                beta, c, &ldc_s32, bias);
}

#define CACHE_LINE_SIZE 64
template <typename a_type, typename b_type, typename c_type>
static mkldnn_status_t gemm_threading_driver(
        gemm_info_t<a_type, b_type, c_type> *arg) {

    if ((arg->m <= 0) || (arg->n <= 0))
        return mkldnn_success;

    if (arg->force_nocopy) {
        return call_no_copy_sgemm(arg->transa, arg->transb,
                arg->m, arg->n, arg->k, arg->alpha,
                (float *) arg->a, arg->lda,
                (float *) arg->b, arg->ldb,
                arg->beta, (float *) arg->c, arg->ldc,
                (float *) arg->co);
    }

    if (data_traits<a_type>::data_type == data_type::s8) {
        if (gemm_s8u8s32_jump_to_gemv_s8u8s32(arg)) {
            return mkldnn_success;
        }
    }

    int nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();

    // Check if thread is beneficial.
    if (mayiuse(avx2) && !mayiuse(avx512_core)) {
        if (arg->m > 10 * arg->n && arg->n < nthr) {
            const int veclen = cpu_isa_traits<avx2>::vlen / (int)sizeof(c_type);
            if (arg->m / nthr < veclen * 3) {
                nthr = (int) nstl::max(arg->m / veclen / 3, 1LL);
            }
        }
    }
    get_omp_thread_count<c_type>(arg->m, arg->n, arg->k, &nthr);

    if (nthr == 1) {
        return gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a, arg->b,
                arg->c, arg->co, arg);
    }

    if ((data_traits<a_type>::data_type == data_type::f32) &&
            nocopy_checker(nthr, arg->transa, arg->transb, arg->m, arg->n,
                arg->k, arg->lda, arg->ldb, arg->ldc))
        return call_no_copy_sgemm(arg->transa, arg->transb,
                arg->m, arg->n, arg->k, arg->alpha,
                (float *) arg->a, arg->lda,
                (float *) arg->b, arg->ldb,
                arg->beta, (float *) arg->c, arg->ldc, NULL);

    mkldnn_status_t *results = (mkldnn_status_t *) malloc(
            sizeof(*results) * nthr * CACHE_LINE_SIZE, PAGE_4K);

    if (!results) {
        return mkldnn_out_of_memory;
    }

    for (int i = 0; i < nthr; i++) {
        results[i * CACHE_LINE_SIZE] = mkldnn_success; // Initialize to success
    }

    char *shared_mem = NULL;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int nthrs = nthr;
        if (nthrs == 1) {
            results[0] = gemm_kernel_driver(arg->m, arg->n, arg->k, arg->a,
                arg->b, arg->c, arg->co, arg);
        } else {
            blas_thread_t thread_info;
            set_thread_opts(&nthrs, &thread_info, arg);

            const a_type *a = NULL;
            const b_type *b = NULL;
            c_type *c = NULL;
            const c_type *co = NULL;
            dim_t m = -1;
            dim_t n = -1;
            dim_t k = -1;
            decompose_matrices(ithr, &nthrs, &m, &n, &k, &a, &b, &c, &co,
                &thread_info, arg);

            if (ithr < nthrs) {
                switch (thread_info.copy_type) {
                case COPY_A:
                    results[ithr * CACHE_LINE_SIZE] =
                        parallel_a_copy(ithr, nthrs, m, n, k, a, b, c, co, arg,
                                &shared_mem);
                    break;

                default:
                case COPY_NONE:
                    results[ithr * CACHE_LINE_SIZE] =
                        gemm_kernel_driver(m, n, k, a, b, c, co, arg);
                    break;

                case NO_COPY:
                    if (data_traits<a_type>::data_type == data_type::f32) {
                        if (mayiuse(avx512_core)) {
                            avx512_common_gemm_f32::sgemm_nocopy_driver(
                                    arg->transa == no_trans ? "N" : "T",
                                    arg->transb == no_trans ? "N" : "T",
                                    (int) m, (int) n, (int) k, arg->alpha,
                                    (float *) a, arg->lda,
                                    (float *) b, arg->ldb,
                                    arg->beta, (float *) c, arg->ldc,
                                    NULL, NULL);
                        } else {
                            avx_gemm_f32::sgemm_nocopy_driver(
                                    arg->transa == no_trans ? "N" : "T",
                                    arg->transb == no_trans ? "N" : "T",
                                    (int) m, (int) n, (int) k, arg->alpha,
                                    (float *) a, arg->lda,
                                    (float *) b, arg->ldb,
                                    arg->beta, (float *)c, arg->ldc,
                                    NULL, NULL);
                        }
                        results[ithr * CACHE_LINE_SIZE] = mkldnn_success;
                    }
                    break;

                }
            }
        }
    });

    mkldnn_status_t result = mkldnn_success;  // Initialize to success
    for (int i = 0; i < nthr; i++) {
        if (results[i * CACHE_LINE_SIZE] != mkldnn_success) {
            result = results[i * CACHE_LINE_SIZE];
            break;
        }
    }

    mkldnn::impl::free(results);

    return result;
}
#undef CACHE_LINE_SIZE

template <typename a_type, typename b_type, typename c_type>
mkldnn_status_t gemm_driver(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const a_type *a, const int *lda, const a_type *oa,
        const b_type *b, const int *ldb, const a_type *ob,
        const float *beta, c_type *c, const int *ldc, const c_type *oc,
        const bool force_nocopy) {

    // gemm_driver supports bfloat16 gemm for avx512_core and above.
    assert(IMPLICATION(data_traits<a_type>::data_type == data_type::bf16,
                mayiuse(avx512_core) && !force_nocopy));

    // gemm_driver supports 8-bit integer for avx512_core and above.
    assert(IMPLICATION(data_traits<a_type>::data_type == data_type::s8,
                mayiuse(avx512_core) && !force_nocopy));

    // gemm_driver supports sgemm for avx.
    assert(IMPLICATION(data_traits<a_type>::data_type == data_type::f32,
            mayiuse(avx)));

    gemm_info_t<a_type, b_type, c_type> args(transA, transB, offsetC, m, n, k,
            alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc, force_nocopy);

    // Check if copy algorithm kernels were generated on supported ISAs.
    assert(args.hasKernels());

    return gemm_threading_driver(&args);
}

template // Instantiate gemm_bf16bf16f32
mkldnn_status_t gemm_driver<mkldnn_bfloat16_t, mkldnn_bfloat16_t, float>(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k, const float *alpha,
        const mkldnn_bfloat16_t *a, const int *lda, const mkldnn_bfloat16_t *oa,
        const mkldnn_bfloat16_t *b, const int *ldb, const mkldnn_bfloat16_t *ob,
        const float *beta, float *c, const int *ldc, const float *oc,
        const bool force_nocopy);

template // Instantiate gemm_s8u8s32
mkldnn_status_t gemm_driver<int8_t, uint8_t, int32_t>(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const int8_t *a, const int *lda, const int8_t *oa,
        const uint8_t *b, const int *ldb, const int8_t *ob,
        const float *beta, int32_t *c, const int *ldc, const int32_t *oc,
        const bool force_nocopy);

template // Instantiate sgemm
mkldnn_status_t gemm_driver<float, float, float>(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const float *a, const int *lda, const float *oa,
        const float *b, const int *ldb, const float *ob,
        const float *beta, float *c, const int *ldc, const float *oc,
        const bool force_nocopy);

}
}
}
