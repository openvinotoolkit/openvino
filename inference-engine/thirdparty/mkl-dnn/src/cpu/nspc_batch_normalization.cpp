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
#include "type_helpers.hpp"

#include "cpu_batch_normalization_utils.hpp"

#include "bfloat16_utils.hpp"
#include "nspc_batch_normalization.hpp"

// clang 6 and 7 generate incorrect code with OMP_SIMD in some particular cases
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define SAFE_TO_USE_OMP_SIMD 0
#else
#define SAFE_TO_USE_OMP_SIMD 1
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

template <data_type_t data_type>
void nspc_batch_normalization_fwd_t<data_type>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));

    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();
    const bool calculate_stats = !pd()->stats_is_src();
    const bool with_relu = pd()->with_relu_post_op();

    auto scratchpad = this->scratchpad();
    auto tmp_mean = scratchpad.template get<acc_data_t>(key_bnorm_tmp_mean);
    auto tmp_var = scratchpad.template get<acc_data_t>(key_bnorm_tmp_var);

    acc_data_t *mean, *variance;
    if (!calculate_stats) {
        mean = reinterpret_cast<acc_data_t *>(
                const_cast<char *>(this->input_memory(1)));
        variance = reinterpret_cast<acc_data_t *>(
                const_cast<char *>(this->input_memory(2)));
    } else {
        if (save_stats) {
            mean = reinterpret_cast<acc_data_t *>(this->memory(1));
            variance = reinterpret_cast<acc_data_t *>(this->memory(2));
        } else {
            mean = tmp_mean;
            variance = tmp_var;
        }
    }
    auto idx_scaleshift = 1 + 2 * pd()->stats_is_src();
    auto scaleshift = reinterpret_cast<const acc_data_t *>(
            this->input_memory(idx_scaleshift));

    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto ws = reinterpret_cast<uint8_t *>(this->memory(pd()->ws_idx()));
    auto *ws_reduce = scratchpad.template get<acc_data_t>(key_bnorm_reduction);
    acc_data_t *tmp_data_ = data_type == data_type::bf16
        ? scratchpad.template get<acc_data_t>(key_bnorm_bf16cvt)
        : nullptr;

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const int simd_w = 16;
    const dim_t C_align = utils::rnd_up(C, simd_w);
    const dim_t SP = pd()->H() * pd()->W() * pd()->D();

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    auto maybe_post_op
            = [&](acc_data_t res) { return (with_relu && res < 0) ? 0 : res; };

    assert(mkldnn_thr_syncable());
    parallel(0, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);
        acc_data_t *mean_loc = tmp_mean + nstl::max(C, (dim_t)16) * ithr;
        acc_data_t *variance_loc = tmp_var + nstl::max(C, (dim_t)16) * ithr;

        if (calculate_stats) {
            for (dim_t c = 0; c < C; c++)
                ws_reduce[C * ithr + c] = 0.;

            for (dim_t n = N_s; n < N_e; n++) {
                for (dim_t sp = 0; sp < SP; sp++) {
                    const acc_data_t *_src;
                    const size_t s_off = (size_t)n * SP * C + sp * C;
                    if (data_type == data_type::bf16) {
                        // convert src from b16 to f32
                        acc_data_t *tmp_src = tmp_data_ + ithr * C_align;
                        bf16_cvt_utils::cvt_bfloat16_to_float(tmp_src,
                                (mkldnn_bfloat16_t *)src + s_off, C);
                        _src = tmp_src;
                    } else {
                        _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                    }
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++) {
                        ws_reduce[C * ithr + c] += _src[c];
                    }
                }
            }

            mkldnn_thr_barrier();

            for (dim_t c = C_s; c < C_e; c++) {
                mean[c] = 0;
                for (dim_t n = 0; n < nthr; n++)
                    mean[c] += ws_reduce[C * n + c];
                mean[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (dim_t c = 0; c < C; c++) {
                mean_loc[c] = mean[c];
                ws_reduce[C * ithr + c] = 0.;
            }

            for (dim_t n = N_s; n < N_e; n++) {
                for (dim_t sp = 0; sp < SP; sp++) {
                    const acc_data_t *_src;
                    const size_t s_off = (size_t)n * SP * C + sp * C;
                    if (data_type == data_type::bf16) {
                        // convert src from b16 to f32
                        acc_data_t *tmp_src = tmp_data_ + ithr * C_align;
                        bf16_cvt_utils::cvt_bfloat16_to_float(tmp_src,
                                (mkldnn_bfloat16_t *)src + s_off, C);
                        _src = tmp_src;
                    } else {
                        _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                    }
                    PRAGMA_OMP_SIMD()
                    for (int c = 0; c < C; c++) {
                        acc_data_t m = _src[c] - mean_loc[c];
                        ws_reduce[C * ithr + c] += m * m;
                    }
                }
            }

            mkldnn_thr_barrier();

            for (dim_t c = C_s; c < C_e; c++) {
                variance[c] = 0;
                for (dim_t n = 0; n < nthr; n++)
                    variance[c] += ws_reduce[C * n + c];
                variance[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (dim_t c = 0; c < C; c++)
                variance_loc[c] = variance[c];
        } else {
            variance_loc = variance;
            mean_loc = mean;
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                acc_data_t *_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (data_type == data_type::bf16) {
                    // store dst to f32 buffer
                    _dst = tmp_data_ + ithr * C_align;
                    // convert src from b16 to f32
                    acc_data_t *tmp_src = tmp_data_ + (nthr + ithr) * C_align;
                    bf16_cvt_utils::cvt_bfloat16_to_float(tmp_src,
                            (mkldnn_bfloat16_t *)src + s_off, C);
                    _src = tmp_src;
                } else {
                    _dst = reinterpret_cast<acc_data_t *>(dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (int c = 0; c < C; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t sqrt_variance = static_cast<acc_data_t>(
                            sqrtf(variance_loc[c] + eps));
                    acc_data_t sm = (use_scaleshift ? scaleshift[c] : 1.0f) / sqrt_variance;
                    acc_data_t sv = use_scaleshift ? scaleshift[C + c] : 0;
                    acc_data_t bn_res = sm * (_src[c] - mean_loc[c]) + sv;
                    if (fuse_bn_relu) {
                        if (bn_res <= 0) {
                            bn_res = 0;
                            if (is_training)
                                ws[c_off] = 0;
                        } else {
                            if (is_training)
                                ws[c_off] = 1;
                        }
                    }
                    _dst[c] = maybe_post_op(bn_res);
                }
                if (data_type == data_type::bf16) {
                    // convert dst from f32 to b16
                    bf16_cvt_utils::cvt_float_to_bfloat16((mkldnn_bfloat16_t *)dst + s_off, _dst, C);
                }
            }
        }
    });
}

template struct nspc_batch_normalization_fwd_t<data_type::f32>;
template struct nspc_batch_normalization_fwd_t<data_type::bf16>;

template <data_type_t data_type>
void nspc_batch_normalization_bwd_t<data_type>::execute_backward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const acc_data_t *>(this->input_memory(1));
    auto variance = reinterpret_cast<const acc_data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scaleshift
            = reinterpret_cast<const acc_data_t *>(this->input_memory(4));
    auto ws = reinterpret_cast<const uint8_t *>(
            this->input_memory(pd()->ws_idx()));

    auto scratchpad = this->scratchpad();
    auto tmp_diff_ss = scratchpad.template get<acc_data_t>(key_bnorm_tmp_diff_ss);

    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_scaleshift = this->memory(1)
        ? reinterpret_cast<acc_data_t *>(this->memory(1)) : tmp_diff_ss;

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const int simd_w = 16;
    const dim_t C_align = utils::rnd_up(C, simd_w);
    const dim_t SP = pd()->D() * pd()->H() * pd()->W();
    acc_data_t *diff_gamma = diff_scaleshift, *diff_beta = diff_scaleshift + C;
    acc_data_t *ws_reduce = scratchpad.template get<acc_data_t>(key_bnorm_reduction);
    acc_data_t *tmp_data_ = data_type == data_type::bf16
        ? scratchpad.template get<acc_data_t>(key_bnorm_bf16cvt)
        : nullptr;

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    assert(mkldnn_thr_syncable());
    parallel(0, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);

        acc_data_t *diff_gamma_loc = tmp_diff_ss + 2 * C + C * ithr;
        acc_data_t *diff_beta_loc = tmp_diff_ss + 2 * C + C * (nthr + ithr);

        for (dim_t c = 0; c < C; c++) {
            ws_reduce[C * ithr + c] = 0.;
            ws_reduce[C * nthr + C * ithr + c] = 0.;
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                const acc_data_t *_diff_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (data_type == data_type::bf16) {
                    // convert diff_dst from b16 to f32
                    acc_data_t *tmp_diff_dst = tmp_data_ + ithr * C_align;
                    bf16_cvt_utils::cvt_bfloat16_to_float(tmp_diff_dst,
                            (mkldnn_bfloat16_t *)diff_dst + s_off, C);
                    _diff_dst = tmp_diff_dst;
                    // convert src from b16 to f32
                    acc_data_t *tmp_src = tmp_data_ + (nthr + ithr) * C_align;
                    bf16_cvt_utils::cvt_bfloat16_to_float(tmp_src,
                            (mkldnn_bfloat16_t *)src + s_off, C);
                    _src = tmp_src;
                } else {
                    _diff_dst = reinterpret_cast<const acc_data_t *>(diff_dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t dd;
                    if (fuse_bn_relu && !ws[c_off])
                        dd = 0;
                    else
                        dd = _diff_dst[c];
                    ws_reduce[C * ithr + c] += (_src[c] - mean[c]) * dd;
                    ws_reduce[C * nthr + C * ithr + c] += dd;
                }
            }
        }

        mkldnn_thr_barrier();

        for (dim_t c = C_s; c < C_e; c++) {
            acc_data_t sqrt_variance
                    = static_cast<acc_data_t>(1.0f / sqrtf(variance[c] + eps));
            diff_gamma[c] = 0;
            diff_beta[c] = 0;
            for (dim_t n = 0; n < nthr; n++) {
                diff_gamma[c] += ws_reduce[C * n + c];
                diff_beta[c] += ws_reduce[C * nthr + C * n + c];
            }
            diff_gamma[c] *= sqrt_variance;
        }

        mkldnn_thr_barrier();

        for (dim_t c = 0; c < C; c++) {
            diff_gamma_loc[c] = diff_gamma[c];
            diff_beta_loc[c] = diff_beta[c];
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
                acc_data_t *_diff_src;
                const acc_data_t *_diff_dst;
                const acc_data_t *_src;
                const size_t s_off = (size_t)n * SP * C + sp * C;
                if (data_type == data_type::bf16) {
                    // store diff_src to f32 buffer
                    _diff_src = tmp_data_ + ithr * C_align;
                    // convert diff_dst from b16 to f32
                    acc_data_t *tmp_diff_dst = tmp_data_ + ithr * C_align;
                    bf16_cvt_utils::cvt_bfloat16_to_float(tmp_diff_dst,
                            (mkldnn_bfloat16_t *)diff_dst + s_off, C);
                    _diff_dst = tmp_diff_dst;
                    if (calculate_diff_stats) {
                        // convert src from b16 to f32
                        acc_data_t *tmp_src = tmp_data_ + (2 * nthr + ithr) * C_align;
                        bf16_cvt_utils::cvt_bfloat16_to_float(tmp_src,
                                (mkldnn_bfloat16_t *)src + s_off, C);
                        _src = tmp_src;
                    } else
                        _src = nullptr; // to avoid compiler warning w/ gcc483
                } else {
                    _diff_src = reinterpret_cast<acc_data_t *>(diff_src + s_off);
                    _diff_dst = reinterpret_cast<const acc_data_t *>(diff_dst + s_off);
                    _src = reinterpret_cast<const acc_data_t *>(src + s_off);
                }
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    const size_t c_off = s_off + c;
                    acc_data_t gamma = use_scaleshift ? scaleshift[c] : 1;
                    acc_data_t sqrt_variance = static_cast<acc_data_t>(
                            1.0f / sqrtf(variance[c] + eps));
                    acc_data_t v_diff_src;
                    if (fuse_bn_relu && !ws[c_off])
                        v_diff_src = 0;
                    else
                        v_diff_src = _diff_dst[c];
                    if (calculate_diff_stats) {
                        v_diff_src -= diff_beta_loc[c] / (SP * N)
                                + (_src[c] - mean[c])
                                        * diff_gamma_loc[c] * sqrt_variance
                                        / (SP * N);
                    }
                    v_diff_src *= gamma * sqrt_variance;
                    _diff_src[c] = v_diff_src;
                }
                if (data_type == data_type::bf16) {
                    // convert diff_src from f32 to b16
                    bf16_cvt_utils::cvt_float_to_bfloat16((mkldnn_bfloat16_t *)diff_src + s_off, _diff_src, C);
                }
            }
        }
    });
}

template struct nspc_batch_normalization_bwd_t<data_type::f32>;
template struct nspc_batch_normalization_bwd_t<data_type::bf16>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
