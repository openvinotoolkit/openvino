/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"

#include "bfloat16_utils.hpp"
#include "ref_batch_normalization.hpp"

#define DECLARE_DATA_OFFSET \
    auto data_offset = [&](const memory_desc_wrapper &data_d, int n, int c, \
            int d, int h, int w) { \
        if (has_spatial) { \
            if (is_3d) \
                return data_d.off(n, c, d, h, w); \
            else \
                return data_d.off(n, c, h, w); \
        } else { \
            return data_d.off(n, c); \
        } \
    }

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

typedef float acc_data_t;

template <typename T>
inline float maybe_up_convert(T x) {
    return x;
}

template <>
inline float maybe_up_convert<mkldnn_bfloat16_t>(mkldnn_bfloat16_t x) {
    return bf16_cvt_utils::cvt_bfloat16_to_float(x);
}

}

template <data_type_t data_type>
void ref_batch_normalization_fwd_t<data_type>::execute_forward()
const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    /* FIXME: check this */
    acc_data_t *mean = pd()->stats_is_src() ?
        const_cast<acc_data_t *>(reinterpret_cast<const acc_data_t *>(
               this->input_memory(1))) :
        reinterpret_cast<acc_data_t *>(this->memory(1));

    acc_data_t *variance = pd()->stats_is_src() ?
        const_cast<acc_data_t *>(reinterpret_cast<const acc_data_t *>(
                this->input_memory(2))) :
        reinterpret_cast<acc_data_t *>(this->memory(2));

    auto idx_scaleshift = 1 + 2 * pd()->stats_is_src();
    auto scaleshift =
        reinterpret_cast<const acc_data_t *>(this->input_memory(idx_scaleshift));

    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto ws = reinterpret_cast<uint8_t *>(this->memory(pd()->ws_idx()));

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) return;

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper scaleshift_d(pd()->weights_pd());

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    dim_t H = 1, W = 1, D = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 4, 5);
    if (has_spatial) {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
    }

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();;
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();
    const bool calculate_stats = !pd()->stats_is_src();

    const bool with_relu = pd()->with_relu_post_op();
    auto maybe_post_op = [&](acc_data_t res) {
        return (with_relu && res < 0.0f) ? 0.0f : res;
    };
    const bool is_3d = data_d.ndims() == 5;

    //auto data_offset(const memory_desc_wrapper &, int, int, int, int, int)
    DECLARE_DATA_OFFSET;

    parallel_nd(C, [&](int c) {
        acc_data_t v_mean = calculate_stats ? 0 : mean[c];
        acc_data_t v_variance = calculate_stats ? 0 : variance[c];

        if (calculate_stats) {
            for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                v_mean += maybe_up_convert(src[data_offset(data_d, n, c, d, h, w)]);
            }
            v_mean /= W * N * H * D;

            for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
            for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                acc_data_t m = maybe_up_convert(src[data_offset(data_d, n, c, d, h, w)]) - v_mean;
                v_variance += m * m;
            }
            v_variance /= W * H * N * D;
        }

        acc_data_t sqrt_variance = sqrtf(v_variance + eps);
        acc_data_t sm = (use_scaleshift
            ? scaleshift[scaleshift_d.off(0, c)]
            : 1.0f) / sqrt_variance;
        acc_data_t sv = use_scaleshift ? scaleshift[scaleshift_d.off(1, c)] : 0;

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            auto d_off = data_offset(data_d, n, c, d, h, w);
            acc_data_t bn_res = sm * (maybe_up_convert(src[d_off]) - v_mean) + sv;
            if (fuse_bn_relu) {
                if (bn_res <= 0) {
                    bn_res = 0;
                    if (is_training)
                        ws[d_off] = 0;
                } else {
                    if (is_training)
                        ws[d_off] = 1;
                }
            }
            if (data_type == data_type::s8) {
                dst[d_off] = qz_a1b0<float, data_t>()(
                        maybe_post_op(bn_res), round_mode::nearest);
            } else if (data_type == data_type::bf16) {
                const float bn_res_p = maybe_post_op(bn_res);
                bf16_cvt_utils::cvt_float_to_bfloat16(
                        (mkldnn_bfloat16_t *)&dst[d_off], &bn_res_p);
            } else {
                dst[d_off] = static_cast<data_t>(maybe_post_op(bn_res));
            }
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[c] = v_mean;
                variance[c] = v_variance;
            }
        }
    });
}

template struct ref_batch_normalization_fwd_t<data_type::s8>;
template struct ref_batch_normalization_fwd_t<data_type::f32>;
template struct ref_batch_normalization_fwd_t<data_type::bf16>;

template <data_type_t data_type>
void ref_batch_normalization_bwd_t<data_type>::execute_backward()
const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto mean = reinterpret_cast<const acc_data_t *>(this->input_memory(1));
    auto variance = reinterpret_cast<const acc_data_t *>(this->input_memory(2));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(3));
    auto scaleshift = reinterpret_cast<const acc_data_t *>(this->input_memory(4));
    auto ws = reinterpret_cast<const uint8_t *>(
            this->input_memory(pd()->ws_idx()));

    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_scaleshift = reinterpret_cast<acc_data_t *>(this->memory(1));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());
    const memory_desc_wrapper scaleshift_d(pd()->weights_pd());
    const memory_desc_wrapper diff_scaleshift_d(pd()->diff_weights_pd());
    const memory_desc_wrapper mean_d(pd()->mean_pd());
    const memory_desc_wrapper variance_d(pd()->variance_pd());

    const dim_t C = pd()->C();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (diff_scaleshift) {
            for (dim_t c = 0; c < C; ++c) {
                diff_scaleshift[diff_scaleshift_d.off(0, c)] = 0;
                diff_scaleshift[diff_scaleshift_d.off(1, c)] = 0;
            }
        }
        return;
    }

    const dim_t N = pd()->MB();
    dim_t H = 1, W = 1, D = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 4, 5);
    if (has_spatial) {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
    }

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    const bool is_3d = data_d.ndims() == 5;

    //auto data_offset(const memory_desc_wrapper &, int, int, int, int, int)
    DECLARE_DATA_OFFSET;

    parallel_nd(C, [&](int c) {
        acc_data_t v_mean = mean[mean_d.off(c)];
        acc_data_t v_variance = variance[variance_d.off(c)];
        acc_data_t sqrt_variance = static_cast<acc_data_t>(1.0f / sqrtf(v_variance + eps));
        acc_data_t gamma = use_scaleshift ? scaleshift[scaleshift_d.off(0, c)] : 1;
        acc_data_t diff_gamma = acc_data_t(0);
        acc_data_t diff_beta = acc_data_t(0);

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = data_offset(data_d, n, c, d, h, w);
            acc_data_t dd;
            if (fuse_bn_relu && !ws[s_off])
                dd = 0;
            else
                dd = maybe_up_convert(
                    diff_dst[data_offset(diff_data_d, n, c, d, h, w)]);
            diff_gamma += (maybe_up_convert(src[s_off]) - v_mean) * dd;
            diff_beta += dd;
        }
        diff_gamma *= sqrt_variance;

        if (diff_scaleshift) {
            diff_scaleshift[diff_scaleshift_d.off(0, c)] = diff_gamma;
            diff_scaleshift[diff_scaleshift_d.off(1, c)] = diff_beta;
        }

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = data_offset(data_d, n, c, d, h, w);
            const size_t dd_off = data_offset(diff_data_d, n, c, d, h, w);
            acc_data_t dd;
            if (fuse_bn_relu && !ws[s_off])
                dd = 0;
            else
                dd = maybe_up_convert(diff_dst[dd_off]);
            acc_data_t v_diff_src = dd;
            if (calculate_diff_stats) {
                v_diff_src -= diff_beta / (D * W * H * N) +
                    (maybe_up_convert(src[s_off]) - v_mean) * diff_gamma * sqrt_variance / (D * W * H * N);
            }
            v_diff_src *= gamma * sqrt_variance;
            if (data_type == data_type::bf16) {
                bf16_cvt_utils::cvt_float_to_bfloat16(
                        (mkldnn_bfloat16_t *)&diff_src[dd_off], &v_diff_src);
            } else {
                diff_src[dd_off] = static_cast<data_t>(v_diff_src);
            }
        }
    });
}

template struct ref_batch_normalization_bwd_t<data_type::f32>;
template struct ref_batch_normalization_bwd_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
