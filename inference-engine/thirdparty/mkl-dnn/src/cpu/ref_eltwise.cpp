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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "bfloat16_utils.hpp"

#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;
using namespace math;

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(alg_kind_t alg, float alpha,
        float beta): alg_(alg), alpha_(alpha), beta_(beta) {
    assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
                eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                eltwise_clamp, eltwise_exp, eltwise_not));
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        const post_ops_t::entry_t::eltwise_t &eltwise)
    : ref_eltwise_scalar_fwd_t(eltwise.alg, eltwise.alpha, eltwise.beta) {}

float ref_eltwise_scalar_fwd_t::compute_scalar(float s) {
    switch (alg_) {
        case eltwise_relu: return relu_fwd(s, alpha_);
        case eltwise_tanh: return tanh_fwd(s);
        case eltwise_elu: return elu_fwd(s, alpha_);
        case eltwise_square: return square_fwd(s);
        case eltwise_abs: return abs_fwd(s);
        case eltwise_sqrt: return sqrt_fwd(s);
        case eltwise_linear: return linear_fwd(s, alpha_, beta_);
        case eltwise_bounded_relu: return bounded_relu_fwd(s, alpha_);
        case eltwise_soft_relu: return soft_relu_fwd(s);
        case eltwise_logistic: return logistic_fwd(s);
        case eltwise_clamp: return clamp_fwd(s, alpha_, beta_);
        case eltwise_exp: return exp_fwd(s);
        case eltwise_not: return not_fwd(s);
        default: assert(!"unknown eltwise alg_kind");
    }

    return 0.f;
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_nCspBc_padded() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const blocking_desc_t &blk = data_d.blocking_desc();
    const int block = blk.block_dims[1];

    const int MB = pd()->MB();
    const int C = pd()->C() / block;
    const int C_PADDED = blk.padding_dims[1] / block;
    const int tail = pd()->C() % block;
    const int SP = pd()->D() * pd()->H() * pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    auto ker = [=] (data_t &d, data_t s) {
        switch (alg_kind) {
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_bounded_relu:
                d = bounded_relu_fwd(s, alpha); break;
            case eltwise_soft_relu: d = soft_relu_fwd(s); break;
            case eltwise_logistic: d = logistic_fwd(s); break;
            case eltwise_clamp: d = clamp_fwd(s, alpha, beta); break;
            case eltwise_exp: d = exp_fwd(s); break;
            case eltwise_not: d = not_fwd(s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    };

    // FIXME: integer overflow?

    parallel_nd(MB, C_PADDED, SP,
        [&](int n, int c, int sp) {
        auto d_off = (n*C_PADDED*SP + c*SP + sp) * block;
        if (c < C) {
            for (int v = 0; v < block; v++)
                ker(dst[d_off + v], src[d_off + v]);
        } else {
            for (int v = 0; v < tail; v++)
                ker(dst[d_off + v], src[d_off + v]);
        }
    });
}

template <>
void ref_eltwise_fwd_t<data_type::bf16>::execute_forward_nCspBc_padded() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const blocking_desc_t &blk = data_d.blocking_desc();
    const int block = blk.block_dims[1];

    const int MB = pd()->MB();
    const int C = pd()->C() / block;
    const int C_PADDED = blk.padding_dims[1] / block;
    const int tail = pd()->C() % block;
    const int SP = pd()->D() * pd()->H() * pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    auto ker = [=] (data_t &d, data_t s) {
        float s_ = 0.0f, d_ = 0.0f;
        bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &s);
        switch (alg_kind) {
            case eltwise_linear: d_ = linear_fwd(s_, alpha, beta); break;
            case eltwise_bounded_relu:
                d_ = bounded_relu_fwd(s_, alpha); break;
            case eltwise_soft_relu: d_ = soft_relu_fwd(s_); break;
            case eltwise_logistic: d_ = logistic_fwd(s_); break;
            default: assert(!"unknown eltwise alg_kind");
        }
        bf16_cvt_utils::cvt_float_to_bfloat16(&d, &d_);
    };

    // FIXME: integer overflow?

    parallel_nd(MB, C_PADDED, SP,
        [&](int n, int c, int sp) {
        auto d_off = (n*C_PADDED*SP + c*SP + sp) * block;
        if (c < C) {
            for (int v = 0; v < block; v++)
                ker(dst[d_off + v], src[d_off + v]);
        } else {
            for (int v = 0; v < tail; v++)
                ker(dst[d_off + v], src[d_off + v]);
        }
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_generic() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    const memory_desc_wrapper data_d(pd()->src_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int id, int h, int w) {
        auto d_off = is_3d
            ? data_d.off(n, c, id, h, w) : data_d.off(n, c, h, w);
        data_t s = src[d_off];
        data_t &d = dst[d_off];
        switch (alg_kind) {
            case eltwise_relu: d = relu_fwd(s, alpha); break;
            case eltwise_tanh: d = tanh_fwd(s); break;
            case eltwise_elu: d = elu_fwd(s, alpha); break;
            case eltwise_square: d = square_fwd(s); break;
            case eltwise_abs: d = abs_fwd(s); break;
            case eltwise_sqrt: d = sqrt_fwd(s); break;
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_bounded_relu:
                d = bounded_relu_fwd(s, alpha); break;
            case eltwise_soft_relu: d = soft_relu_fwd(s); break;
            case eltwise_logistic: d = logistic_fwd(s); break;
            case eltwise_clamp: d = clamp_fwd(s, alpha, beta); break;
            case eltwise_exp: d = exp_fwd(s); break;
            case eltwise_not: d = not_fwd(s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <>
void ref_eltwise_fwd_t<data_type::bf16>::execute_forward_generic() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    const memory_desc_wrapper data_d(pd()->src_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int id, int h, int w) {
        auto d_off = is_3d
            ? data_d.off(n, c, id, h, w) : data_d.off(n, c, h, w);
        data_t s = src[d_off];
        data_t &d = dst[d_off];
        float s_ = 0.0f, d_ = 0.0f;
        bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &s);
        switch (alg_kind) {
            case eltwise_relu: d_ = relu_fwd(s_, alpha); break;
            case eltwise_tanh: d_ = tanh_fwd(s_); break;
            case eltwise_elu: d_ = elu_fwd(s_, alpha); break;
            case eltwise_square: d_ = square_fwd(s_); break;
            case eltwise_abs: d_ = abs_fwd(s_); break;
            case eltwise_sqrt: d_ = sqrt_fwd(s_); break;
            case eltwise_linear: d_ = linear_fwd(s_, alpha, beta); break;
            case eltwise_bounded_relu:
                d_ = bounded_relu_fwd(s_, alpha); break;
            case eltwise_soft_relu: d_ = soft_relu_fwd(s_); break;
            case eltwise_logistic: d_ = logistic_fwd(s_); break;
            default: assert(!"unknown eltwise alg_kind");
        }
        bf16_cvt_utils::cvt_float_to_bfloat16(&d, &d_);
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_dense() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta  = pd()->desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    if (alg_kind == eltwise_relu) {
        // a fast path for relu as the most popular activation
        parallel_nd(nelems, [&](ptrdiff_t e) {
            dst[e] = relu_fwd(src[e], alpha);
        });
        return;
    }

    parallel_nd(nelems, [&](ptrdiff_t e) {
        const data_t s = src[e];
        data_t &d = dst[e];

        switch (alg_kind) {
        case eltwise_tanh: d = tanh_fwd(s); break;
        case eltwise_elu: d = elu_fwd(s, alpha); break;
        case eltwise_square: d = square_fwd(s); break;
        case eltwise_abs: d = abs_fwd(s); break;
        case eltwise_sqrt: d = sqrt_fwd(s); break;
        case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
        case eltwise_bounded_relu: d = bounded_relu_fwd(s, alpha); break;
        case eltwise_soft_relu: d = soft_relu_fwd(s); break;
        case eltwise_logistic: d = logistic_fwd(s); break;
        case eltwise_clamp: d = clamp_fwd(s, alpha, beta); break;
        case eltwise_exp: d = exp_fwd(s); break;
        case eltwise_not: d = not_fwd(s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <>
void ref_eltwise_fwd_t<data_type::bf16>::execute_forward_dense() const {
    auto src = reinterpret_cast<const mkldnn_bfloat16_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<mkldnn_bfloat16_t *>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta  = pd()->desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    if (alg_kind == eltwise_relu) {
        // a fast path for relu as the most popular activation
        parallel_nd(nelems, [&](ptrdiff_t e) {
            float s_ = 0.0f;
            bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &src[e]);
            float d_ = relu_fwd(s_, alpha);
            bf16_cvt_utils::cvt_float_to_bfloat16(&dst[e], &d_);
        });
        return;
    }

    parallel_nd(nelems, [&](ptrdiff_t e) {
        float s_ = 0.0f, d_ = 0.0f;
        bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &src[e]);
        switch (alg_kind) {
        case eltwise_tanh: d_ = tanh_fwd(s_); break;
        case eltwise_elu: d_ = elu_fwd(s_, alpha); break;
        case eltwise_square: d_ = square_fwd(s_); break;
        case eltwise_abs: d_ = abs_fwd(s_); break;
        case eltwise_sqrt: d_ = sqrt_fwd(s_); break;
        case eltwise_linear: d_ = linear_fwd(s_, alpha, beta); break;
        case eltwise_bounded_relu: d_ = bounded_relu_fwd(s_, alpha); break;
        case eltwise_soft_relu: d_ = soft_relu_fwd(s_); break;
        case eltwise_logistic: d_ = logistic_fwd(s_); break;
        default: assert(!"unknown eltwise alg_kind");
        }
        bf16_cvt_utils::cvt_float_to_bfloat16(&dst[e], &d_);
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_generic() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int d, int h, int w) {
        auto data_off = is_3d
            ? data_d.off(n, c, d, h, w) : data_d.off(n, c, h, w);
        auto diff_data_off = is_3d
            ? diff_data_d.off(n, c, d, h, w)
            : diff_data_d.off(n, c, h, w);
        data_t s = src[data_off];
        data_t dd = diff_dst[diff_data_off];
        data_t &ds = diff_src[diff_data_off];
        switch (alg_kind) {
            case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
            case eltwise_tanh: ds = tanh_bwd(dd, s); break;
            case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
            case eltwise_square: ds = square_bwd(dd, s); break;
            case eltwise_abs: ds = abs_bwd(dd, s); break;
            case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
            case eltwise_linear:
                ds = linear_bwd(dd, s, alpha, beta); break;
            case eltwise_bounded_relu:
                ds = bounded_relu_bwd(dd, s, alpha); break;
            case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
            case eltwise_logistic: ds = logistic_bwd(dd, s); break;
            case eltwise_clamp: ds = clamp_bwd(dd, s, alpha, beta); break;
            case eltwise_exp: ds = exp_bwd(dd, s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <>
void ref_eltwise_bwd_t<data_type::bf16>::execute_backward_generic() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int d, int h, int w) {
        auto data_off = is_3d
            ? data_d.off(n, c, d, h, w) : data_d.off(n, c, h, w);
        auto diff_data_off = is_3d
            ? diff_data_d.off(n, c, d, h, w)
            : diff_data_d.off(n, c, h, w);

        float dd_ = 0.0f, s_ = 0.0f, ds_ = 0.0f;
        bf16_cvt_utils::cvt_bfloat16_to_float(&dd_, &diff_dst[diff_data_off]);
        bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &src[data_off]);
        switch (alg_kind) {
            case eltwise_relu: ds_ = relu_bwd(dd_, s_, alpha); break;
            case eltwise_tanh: ds_ = tanh_bwd(dd_, s_); break;
            case eltwise_elu: ds_ = elu_bwd(dd_, s_, alpha); break;
            case eltwise_square: ds_ = square_bwd(dd_, s_); break;
            case eltwise_abs: ds_ = abs_bwd(dd_, s_); break;
            case eltwise_sqrt: ds_ = sqrt_bwd(dd_, s_); break;
            case eltwise_linear:
                ds_ = linear_bwd(dd_, s_, alpha, beta); break;
            case eltwise_bounded_relu:
                ds_ = bounded_relu_bwd(dd_, s_, alpha); break;
            case eltwise_soft_relu: ds_ = soft_relu_bwd(dd_, s_); break;
            case eltwise_logistic: ds_ = logistic_bwd(dd_, s_); break;
            default: assert(!"unknown eltwise alg_kind");
        }
        bf16_cvt_utils::cvt_float_to_bfloat16(&diff_src[diff_data_off], &ds_);
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_dense() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

    parallel_nd(nelems, [&](ptrdiff_t e) {
        const data_t dd = diff_dst[e];
        const data_t s = src[e];
        data_t &ds = diff_src[e];

        switch (alg_kind) {
        case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
        case eltwise_tanh: ds = tanh_bwd(dd, s); break;
        case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
        case eltwise_square: ds = square_bwd(dd, s); break;
        case eltwise_abs: ds = abs_bwd(dd, s); break;
        case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
        case eltwise_linear: ds = linear_bwd(dd, s, alpha, beta); break;
        case eltwise_bounded_relu: ds = bounded_relu_bwd(dd, s, alpha); break;
        case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
        case eltwise_logistic: ds = logistic_bwd(dd, s); break;
        case eltwise_clamp: ds = clamp_bwd(dd, s, alpha, beta); break;
        case eltwise_exp: ds = exp_bwd(dd, s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <>
void ref_eltwise_bwd_t<data_type::bf16>::execute_backward_dense() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

    parallel_nd(nelems, [&](ptrdiff_t e) {
        float dd_ = 0.0f, s_ = 0.0f, ds_ = 0.0f;
        bf16_cvt_utils::cvt_bfloat16_to_float(&dd_, &diff_dst[e]);
        bf16_cvt_utils::cvt_bfloat16_to_float(&s_, &src[e]);

        switch (alg_kind) {
        case eltwise_relu: ds_ = relu_bwd(dd_, s_, alpha); break;
        case eltwise_tanh: ds_ = tanh_bwd(dd_, s_); break;
        case eltwise_elu: ds_ = elu_bwd(dd_, s_, alpha); break;
        case eltwise_square: ds_ = square_bwd(dd_, s_); break;
        case eltwise_abs: ds_ = abs_bwd(dd_, s_); break;
        case eltwise_sqrt: ds_ = sqrt_bwd(dd_, s_); break;
        case eltwise_linear: ds_ = linear_bwd(dd_, s_, alpha, beta); break;
        case eltwise_bounded_relu: ds_ = bounded_relu_bwd(dd_, s_, alpha); break;
        case eltwise_soft_relu: ds_ = soft_relu_bwd(dd_, s_); break;
        case eltwise_logistic: ds_ = logistic_bwd(dd_, s_); break;
        default: assert(!"unknown eltwise alg_kind");
        }
        bf16_cvt_utils::cvt_float_to_bfloat16(&diff_src[e], &ds_);
    });
}

template struct ref_eltwise_fwd_t<data_type::f32>;
template struct ref_eltwise_fwd_t<data_type::bf16>;
template struct ref_eltwise_fwd_t<data_type::s32>;
template struct ref_eltwise_fwd_t<data_type::s16>;
template struct ref_eltwise_fwd_t<data_type::s8>;
template struct ref_eltwise_fwd_t<data_type::u8>;

template struct ref_eltwise_bwd_t<data_type::f32>;
template struct ref_eltwise_bwd_t<data_type::bf16>;
template struct ref_eltwise_bwd_t<data_type::s32>;
template struct ref_eltwise_bwd_t<data_type::s16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
