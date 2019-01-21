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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "nspc_batch_normalization.hpp"
#include "type_helpers.hpp"

// clang 6 and 7 generate incorrect code with OMP_SIMD in some particular cases
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define SAFE_TO_USE_OMP_SIMD 0
#else
#define SAFE_TO_USE_OMP_SIMD 1
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

typedef float data_t;
nspc_batch_normalization_fwd_t::nspc_batch_normalization_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), stats_reduction_(nullptr),
    tmp_mean_(nullptr), tmp_variance_(nullptr), conf_(*pd) {
    if (!conf_.stats_is_src()) {
        this->stats_reduction_ = (data_t *)malloc(
                nstl::max(conf_.C(), 16) * mkldnn_get_max_threads() * sizeof(data_t), 64);
        this->tmp_mean_ = (data_t *)malloc(mkldnn_get_max_threads() *
                nstl::max(conf_.C(), 16) * sizeof(data_t), 64);
        this->tmp_variance_
                = (data_t *)malloc(mkldnn_get_max_threads() *
                       nstl::max(conf_.C(), 16) * sizeof(data_t), 64);
    }
}
nspc_batch_normalization_fwd_t::~nspc_batch_normalization_fwd_t() {
    if (!conf_.stats_is_src()) {
        free(this->stats_reduction_);
        free(this->tmp_mean_);
        free(this->tmp_variance_);
    }
}

void nspc_batch_normalization_fwd_t::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    const bool save_stats = conf_.is_training();
    const bool is_training = conf_.is_training();
    const bool fuse_bn_relu = conf_.fuse_bn_relu();
    const bool calculate_stats = !conf_.stats_is_src();
    const bool with_relu = conf_.with_relu_post_op();
    data_t *mean, *variance;
    if (!calculate_stats) {
        mean = reinterpret_cast<data_t *>(
                const_cast<char *>(this->input_memory(1)));
        variance = reinterpret_cast<data_t *>(
                const_cast<char *>(this->input_memory(2)));
    } else {
        if (save_stats) {
            mean = reinterpret_cast<data_t *>(this->memory(1));
            variance = reinterpret_cast<data_t *>(this->memory(2));
        } else {
            mean = this->tmp_mean_;
            variance = this->tmp_variance_;
        }
    }
    auto idx_scaleshift = 1 + 2 * conf_.stats_is_src();
    auto scaleshift = reinterpret_cast<const data_t *>(
            this->input_memory(idx_scaleshift));

    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto ws = reinterpret_cast<uint8_t *>(this->memory(conf_.ws_idx()));
    auto ws_reduce = this->stats_reduction_;

    const int N = conf_.MB();
    const int C = conf_.C();
    const int SP = conf_.H() * conf_.W() * conf_.D();

    const float eps = conf_.desc()->batch_norm_epsilon;
    const bool use_scaleshift = conf_.use_scaleshift();
    auto maybe_post_op
            = [&](data_t res) { return (with_relu && res < 0) ? 0 : res; };

    assert(mkldnn_thr_syncable());
    parallel(0, [&](const int ithr, const int nthr) {
        int N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);
        data_t *mean_loc = this->tmp_mean_ + nstl::max(C, 16)*ithr;
        data_t *variance_loc = this->tmp_variance_ + nstl::max(C,16)*ithr;

        if (calculate_stats) {
            for (int c = 0; c < C; c++)
                ws_reduce[C * ithr + c] = 0.;

            for (int n = N_s; n < N_e; n++)
                for (int sp = 0; sp < SP; sp++)
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++)
                        ws_reduce[C * ithr + c] += src[(size_t)n * SP * C
                            + sp * C + c];

            mkldnn_thr_barrier();

            for (int c = C_s; c < C_e; c++) {
                mean[c] = 0;
                for (int n = 0; n < nthr; n++)
                    mean[c] += ws_reduce[C * n + c];
                mean[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (int c = 0; c < C; c++) {
                mean_loc[c] = mean[c];
                ws_reduce[C * ithr + c] = 0.;
            }

            for (int n = N_s; n < N_e; n++)
                for (int sp = 0; sp < SP; sp++)
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++) {
                        data_t m = src[(size_t)n * SP * C + sp * C + c]
                            - mean_loc[c];
                        ws_reduce[C * ithr + c] += m * m;
                    }

            mkldnn_thr_barrier();

            for (int c = C_s; c < C_e; c++) {
                variance[c] = 0;
                for (int n = 0; n < nthr; n++)
                    variance[c] += ws_reduce[C * n + c];
                variance[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (int c = 0; c < C; c++)
                variance_loc[c] = variance[c];
        } else {
            variance_loc = variance;
            mean_loc = mean;
        }

        for (int n = N_s; n < N_e; n++) {
            for (int sp = 0; sp < SP; sp++) {
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (int c = 0; c < C; c++) {
                    data_t sqrt_variance = static_cast<data_t>(
                            1.0f / sqrtf(variance_loc[c] + eps));
                    data_t sm = use_scaleshift ? scaleshift[c] : 1;
                    data_t sv = use_scaleshift ? scaleshift[C + c] : 0;
                    data_t bn_res
                            = sm * (src[(size_t)n * SP * C + sp * C + c]
                                    - mean_loc[c]) * sqrt_variance + sv;
                    if (fuse_bn_relu) {
                        if (bn_res <= 0) {
                            bn_res = 0;
                            if (is_training)
                                ws[(size_t)n * SP * C + sp * C + c] = 0;
                        } else {
                            if (is_training)
                                ws[(size_t)n * SP * C + sp * C + c] = 1;
                        }
                    }
                    dst[(size_t)n * SP * C + sp * C + c] = maybe_post_op(bn_res);
                }
            }
        }
    });
}

nspc_batch_normalization_bwd_t::nspc_batch_normalization_bwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
    this->stats_reduction_ = (data_t *)malloc(
            conf_.C() * 2 * mkldnn_get_max_threads() * sizeof(data_t), 64);
    this->tmp_diff_scaleshift_
            = (data_t *)malloc((mkldnn_get_max_threads() + 1) * conf_.C() * 2 *
                    sizeof(data_t), 64);
}
nspc_batch_normalization_bwd_t::~nspc_batch_normalization_bwd_t() {
    free(this->stats_reduction_);
    free(this->tmp_diff_scaleshift_);
}


void nspc_batch_normalization_bwd_t::execute_backward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto variance = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scaleshift = reinterpret_cast<const data_t *>(this->input_memory(4));
    auto ws = reinterpret_cast<const uint8_t *>(
            this->input_memory(conf_.ws_idx()));

    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_scaleshift = (this->memory(1)) ?
            reinterpret_cast<data_t *>(this->memory(1)) :
            this->tmp_diff_scaleshift_;

    const int N = conf_.MB();
    const int C = conf_.C();
    const int SP = conf_.D() * conf_.H() * conf_.W();
    data_t *diff_gamma = diff_scaleshift, *diff_beta = diff_scaleshift + C;
    data_t *ws_reduce = this->stats_reduction_;

    const float eps = conf_.desc()->batch_norm_epsilon;
    const bool use_scaleshift = conf_.use_scaleshift();
    const bool calculate_diff_stats = !conf_.omit_stats();
    const bool fuse_bn_relu = conf_.fuse_bn_relu();

    assert(mkldnn_thr_syncable());
    parallel(0, [&](const int ithr, const int nthr) {
        int N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);

        data_t *diff_gamma_loc = this->tmp_diff_scaleshift_ + 2*C + C*ithr;
        data_t *diff_beta_loc = this->tmp_diff_scaleshift_ + 2*C + C*nthr
            + C*ithr;

        for (int c = 0; c < C; c++) {
            ws_reduce[C * ithr + c] = 0.;
            ws_reduce[C * nthr + C * ithr + c] = 0.;
        }

        for (int n = N_s; n < N_e; n++)
            for (int sp = 0; sp < SP; sp++)
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (int c = 0; c < C; c++) {
                    const size_t d_off = (size_t)n * SP * C + sp * C + c;
                    data_t dd;
                    if (fuse_bn_relu)
                        dd = (!ws[d_off]) ? 0 : diff_dst[d_off];
                    else
                        dd = diff_dst[d_off];
                    ws_reduce[C * ithr + c] += (src[d_off] - mean[c]) * dd;
                    ws_reduce[C * nthr + C * ithr + c] += dd;
                }

        mkldnn_thr_barrier();

        for (int c = C_s; c < C_e; c++) {
            data_t sqrt_variance
                    = static_cast<data_t>(1.0f / sqrtf(variance[c] + eps));
            diff_gamma[c] = 0;
            diff_beta[c] = 0;
            for (int n = 0; n < nthr; n++) {
                diff_gamma[c] += ws_reduce[C * n + c];
                diff_beta[c] += ws_reduce[C * nthr + C * n + c];
            }
            diff_gamma[c] *= sqrt_variance;
        }

        mkldnn_thr_barrier();

        for (int c = 0; c < C; c++) {
            diff_gamma_loc[c] = diff_gamma[c];
            diff_beta_loc[c] = diff_beta[c];
        }

        for (int n = N_s; n < N_e; n++) {
            for (int sp = 0; sp < SP; sp++) {
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (int c = 0; c < C; c++) {
                    const size_t d_off = (size_t)n * SP * C + sp * C + c;
                    data_t gamma = use_scaleshift ? scaleshift[c] : 1;
                    data_t sqrt_variance
                            = static_cast<data_t>(1.0f / sqrtf(variance[c] + eps));
                    data_t v_diff_src;
                    if (fuse_bn_relu)
                        v_diff_src = (!ws[d_off]) ? 0 : diff_dst[d_off];
                    else
                        v_diff_src = diff_dst[d_off];
                    if (calculate_diff_stats) {
                        v_diff_src -= diff_beta_loc[c] / (SP * N)
                                + (src[d_off] - mean[c]) * diff_gamma_loc[c]
                                        * sqrt_variance / (SP * N);
                    }
                    v_diff_src *= gamma * sqrt_variance;
                    diff_src[d_off] = v_diff_src;
                }
            }
        }
    });
}
}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
