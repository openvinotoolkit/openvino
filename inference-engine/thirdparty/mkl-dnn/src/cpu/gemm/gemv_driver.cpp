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

#include <cstdint>

#include "gemv_driver.hpp"

#include "cpu_isa_traits.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_types.h"
#include "gemm_info.hpp"
#include "jit_generator.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename a_t, typename b_t, typename c_t>
static void gemv_n_kernel(const dim_t m, const dim_t n, float alpha,
        const a_t *__restrict a, const dim_t lda, const b_t *__restrict x,
        const dim_t inc, c_t *__restrict y,
        const gemm_info_t<a_t, b_t, c_t> *arg) {
    if (inc == 1) {
        for (dim_t i = 0; i < n; i++) {
            for (dim_t j = 0; j < m; j++) {
                y[j] += alpha * x[i] * a[j + i * lda];
            }
        }
    } else {
        dim_t idx = inc < 0 ? (1 - n) * inc : 0;
        for (dim_t i = 0; i < n; i++) {
            for (dim_t j = 0; j < m; j++) {
                y[j] += alpha * x[idx] * a[j + i * lda];
            }
            idx += inc;
        }
    }
}

template <typename a_t, typename b_t, typename c_t>
static void gemv_t_kernel(const dim_t m, const dim_t n, float alpha,
        const a_t *__restrict a, const dim_t lda, const b_t *__restrict x,
        const dim_t incy, c_t *__restrict y,
        const gemm_info_t<a_t, b_t, c_t> *arg) {

    if (mayiuse(sse42)) {
        arg->gemv_kernel[do_trans](&m, &n, &alpha, a, &lda, x, &incy, y);
    } else {
        if (incy == 1) {
            for (dim_t i = 0; i < n; i++) {
                c_t temp = (c_t)0;
                for (dim_t j = 0; j < m; j++) {
                    temp += x[j] * a[j + i * lda];
                }
                y[i] += temp * alpha;
            }
        } else {
            dim_t idy = incy < 0 ? (1 - n) * incy : 0;
            for (dim_t i = 0; i < n; i++) {
                c_t temp = (c_t)0;
                for (dim_t j = 0; j < m; j++) {
                    temp += x[j] * a[j + i * lda];
                }
                y[idy] += temp * alpha;

                idy += incy;
            }
        }
    }
}

#define M_BLK 512
template <typename a_t, typename b_t, typename c_t>
static inline void gemv_kernel_driver(const int trans, const dim_t m,
        const dim_t n, const float alpha, const a_t *a, const dim_t lda,
        const b_t *x, const dim_t incx, const float beta, c_t *y,
        const dim_t incy, const gemm_info_t<a_t, b_t, c_t> *arg) {
    // Quick exit.
    if (m == 0 || n == 0 || (alpha == 0.0f && beta == 1.0f)) { return; }

    // Set dimensions of X and Y vectors based on transpose type.
    dim_t x_dim = 0;
    dim_t y_dim = 0;
    if (trans == no_trans) {
        x_dim = n;
        y_dim = m;
    } else {
        x_dim = m;
        y_dim = n;
    }

    // Set the indices for y and x vectors based on incx/incy
    dim_t idx_x = incx < 0 ? (1 - x_dim) * incx : 0;
    dim_t idx_y = incy < 0 ? (1 - y_dim) * incy : 0;

    // Scale the Y vector
    if (beta != 1.0f) {
        if (incy == 1) {
            if (beta == 0.0f) {
                for (dim_t i = 0; i < y_dim; i++) {
                    y[i] = (c_t)0.0f;
                }
            } else {
                for (dim_t i = 0; i < y_dim; i++) {
                    y[i] *= beta;
                }
            }
        } else {
            if (beta == 0.0f) {
                for (dim_t i = 0, inc = idx_y; i < y_dim; i++) {
                    y[inc] = (c_t)0.0f;
                    inc += incy;
                }
            } else {
                for (dim_t i = 0, inc = idx_y; i < y_dim; i++) {
                    y[inc] *= beta;
                    inc += incy;
                }
            }
        }
    }

    if (alpha == 0.0f) { return; }

    if (trans == no_trans) { // A is not transpose.
        if (incy == 1) {
            gemv_n_kernel(m, n, alpha, a, lda, x, incx, y, arg);
        } else {
            // Allocate temporary buffer for y vector.
            c_t *ytmp = (c_t *)malloc(M_BLK * sizeof(*ytmp), PAGE_4K);

            if (!ytmp) {
                for (dim_t j = 0; j < n; j++) {
                    for (dim_t i = 0, inc = idx_y; i < m; i++) {
                        y[inc] += alpha * x[idx_x] * a[i + j * lda];
                        inc += incy;
                    }
                    idx_x += incx;
                }
                return;
            }

            dim_t m_blk = 0;
            for (dim_t i = 0; i < m; i += m_blk) {
                m_blk = m - i;
                if (m_blk > M_BLK) m_blk = M_BLK;

                // Copy a block of y vector to temporary buffer.
                for (dim_t j = 0, inc = idx_y; j < m_blk; j++) {
                    ytmp[j] = y[inc];
                    inc += incy;
                }

                // Call unit-stride kernel.
                gemv_n_kernel(m_blk, n, alpha, a, lda, x, incx, ytmp, arg);

                // Copy computed result back to y vector.
                for (dim_t j = 0, inc = idx_y; j < m_blk; j++) {
                    y[inc] = ytmp[j];
                    inc += incy;
                }
                a += m_blk;
                y += m_blk * incy;
            }

            free(ytmp);
        }
    } else { // Matrix A is transpose.
        if (incx == 1) {
            gemv_t_kernel(m, n, alpha, a, lda, x, incy, y, arg);
        } else {
            // Allocate temporary buffer for x vector.
            c_t *xtmp = (c_t *)malloc(M_BLK * sizeof(*xtmp), PAGE_4K);

            // If memory is not available, jump to naive code path
            if (!xtmp) {
                for (dim_t j = 0; j < n; j++) {
                    c_t acc = (c_t)0.0f;
                    for (dim_t i = 0, inc = idx_x; i < m; i++) {
                        acc += x[inc] * a[i + j * lda];
                        inc += incx;
                    }
                    y[idx_y] += acc * alpha;

                    idx_y += incy;
                }
                return;
            }

            dim_t m_blk = 0;
            for (dim_t i = 0; i < m; i += m_blk) {
                m_blk = m - i;
                if (m_blk > M_BLK) m_blk = M_BLK;

                // Copy a block of x vector to temporary buffer.
                for (dim_t j = 0, inc = idx_x; j < m_blk; j++) {
                    xtmp[j] = x[inc];
                    inc += incx;
                }

                // Call unit-stride kernel.
                gemv_t_kernel(m_blk, n, alpha, a, lda, xtmp, incy, y, arg);

                a += m_blk;
                x += m_blk * incx;
            }
            free(xtmp);
        }
    }

    return;
}
#undef M_BLK

#define M_MIN 128
#define N_MIN 128
#define BAND_MIN 32
#define MN_MIN_N 1536
#define MN_MIN_T 2048
#define M_LARGE 20000
#define N_LARGE 20000
#define M_SMALL 200
#define N_SMALL 200
#define CONST1_AVX2 288
#define CONST2_AVX2 41700
// Check if threading is beneficial.
static inline dim_t thread_checker(const dim_t m, const dim_t n) {
    dim_t nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();

    // Threshold based on performance measurement with warm and cold cache
    // to decide when threading is beneficial.
    if (mayiuse(avx2)) {
        if (m * n + CONST1_AVX2 * n < CONST2_AVX2) { return 1; }
    } else {
        if (m < M_MIN && n < N_MIN) {
            // Execute in sequential mode for small n and m.
            return 1;
        }
    }

    if (m >= M_LARGE && n <= N_SMALL) {
        // Execute in parallel mode.
        return nthr;
    }

    dim_t bandt = n / nthr; // size per thread.

    if (nthr <= 12 && bandt < BAND_MIN) {
        if (m * bandt < MN_MIN_T) { return 1; }
    } else if (nthr <= 12 && m * bandt < 2 * MN_MIN_T) {
        return 1;
    } else if (nthr > 12 && bandt * m < 2 * MN_MIN_T) {
        if (bandt == 0) {
            return 1;
        } else {
            return nstl::min(
                    dim_t(nstl::max((n * m) / dim_t(2 * MN_MIN_N), dim_t(1))),
                    nthr);
        }
    }

    return nthr;
}
#undef M_MIN
#undef N_MIN
#undef BAND_MIN
#undef MN_MIN_N
#undef MN_MIN_T
#undef M_LARGE
#undef N_LARGE
#undef M_SMALL
#undef N_SMALL
#undef CONST1_AVX2
#undef CONST2_AVX2

template <typename T>
static inline void decompose_vector(const dim_t m, const dim_t nthr,
        const dim_t ithr, T *addr, dim_t *offset, dim_t *size) {
    dim_t loffset = 0;
    dim_t lsize = 0;

    if (addr == NULL) {
        dim_t xthr = m % nthr;
        dim_t width = m / nthr;

        if (ithr < xthr) {
            lsize = width + 1;
            loffset = ithr * lsize;
        } else {
            lsize = width;
            loffset = m - (nthr - ithr) * lsize;
        }
    }

    *offset = loffset;
    *size = lsize;
}

template <typename a_t, typename b_t, typename c_t>
static inline void gemv_threading_driver(const int trans, const dim_t m,
        const dim_t n, const float alpha, const a_t *a, const dim_t lda,
        const b_t *x, const dim_t incx, const float beta, c_t *y,
        const dim_t incy, const gemm_info_t<a_t, b_t, c_t> *arg) {

    // Quick return if possible.
    if (m <= 0 || n <= 0) { return; }

    dim_t nthr = thread_checker(m, n);

    if (nthr == 1) {
        gemv_kernel_driver(
                trans, m, n, alpha, a, lda, x, incx, beta, y, incy, arg);
        return;
    }

    // Execute in parallel mode
    parallel_nd((dim_t)nthr, [&](const dim_t ithr) {
        dim_t band, disp;
        decompose_vector(n, nthr, ithr, (c_t *)NULL, &disp, &band);

        dim_t ydisp = disp * incy;
        if (incy < 0) ydisp = ydisp + (-n + band) * incy;

        disp = disp * lda;

        auto a_loc = a + disp;
        auto x_loc = x;
        auto y_loc = y + ydisp;
        gemv_kernel_driver(trans, m, band, alpha, a_loc, lda, x_loc, incx, beta,
                y_loc, incy, arg);
    });

    return;
}

template <>
mkldnn_status_t jump_to_gemv(const gemm_info_t<int8_t, uint8_t, int32_t> *arg) {
    return mkldnn_unimplemented;
}

template <>
mkldnn_status_t jump_to_gemv(const gemm_info_t<int8_t, int8_t, int32_t> *arg) {
    return mkldnn_unimplemented;
}

template <>
mkldnn_status_t jump_to_gemv(
        const gemm_info_t<mkldnn_bfloat16_t, mkldnn_bfloat16_t, float> *arg) {
    return mkldnn_unimplemented;
}

template <typename a_t, typename b_t, typename c_t>
mkldnn_status_t jump_to_gemv(const gemm_info_t<a_t, b_t, c_t> *arg) {
    int transa = arg->transa;
    int transb = arg->transb;

    dim_t m = arg->m;
    dim_t n = arg->n;
    dim_t k = arg->k;

    dim_t lda = arg->lda;
    dim_t ldb = arg->ldb;
    dim_t ldc = arg->ldc;

    float alpha = arg->alpha;
    float beta = arg->beta;

    const a_t *a = arg->a;
    const b_t *b = arg->b;
    c_t *c = arg->c;

    if (k == 0) return mkldnn_success;

    if (n == 1 && transa == do_trans) {
        gemv_threading_driver(do_trans, k, m, alpha, a, lda, b,
                transb == no_trans ? 1 : ldb, beta, c, 1, arg);
        return mkldnn_success;
    }

    if (m == 1 && transb == no_trans) {
        gemv_threading_driver(do_trans, k, n, alpha, b, ldb, a,
                transa == no_trans ? lda : 1, beta, c, ldc, arg);
        return mkldnn_success;
    }

    return mkldnn_unimplemented;
}

template // Instatiate gemv_f32
        mkldnn_status_t
        jump_to_gemv<float, float, float>(
                const gemm_info_t<float, float, float> *arg);

} // namespace cpu
} // namespace impl
} // namespace mkldnn
