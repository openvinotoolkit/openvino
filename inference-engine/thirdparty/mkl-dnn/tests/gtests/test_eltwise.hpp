/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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
#ifndef TEST_ELTWISE_HPP
#define TEST_ELTWISE_HPP

#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename T, typename A> inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(s * alpha);
}
template <typename T, typename A> inline T relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : static_cast<T>(dd * alpha);
}
template <typename T> T tanh_fwd(T s) {
    return static_cast<T>(::tanhf((float)s));
}
template <typename T> T tanh_bwd(T dd, T s) {
    const float th = ::tanhf((float)s);
    return static_cast<T>(dd * (1 - th) * (1 + th));
}

template <typename T, typename A> T elu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(alpha * (::expf(s) - 1));
}
template <typename T, typename A> T elu_bwd(T dd, T s, A alpha) {
    return static_cast<T>(dd * (s > 0 ? 1 : alpha * ::expf(s)));
}

template <typename T>
T square_fwd(T s) {
    return s * s;
}

template <typename T>
T square_bwd(T dd, T s) {
    return dd * 2*s;
}

template <typename T>
T abs_fwd(T s) {
    return s > 0 ? s : -s;;
}

template <typename T>
T abs_bwd(T dd, T s) {
    return dd * (s > 0 ? 1 : s < 0 ? -1 : 0);
}

template <typename T>
T sqrt_fwd(T s) {
    return s > 0 ? ::sqrtf(s) : 0;
}

template <typename T>
T sqrt_bwd(T dd, T s) {
    return s > 0 ? dd / (2 * ::sqrtf(s)) : 0;
}

template <typename T, typename A>
T linear_fwd(T s, A alpha, A beta) {
    return alpha * s + beta;
}

template <typename T, typename A>
T linear_bwd(T dd, T s, A alpha, A beta) {
    (void) s;
    (void) beta;
    return dd * alpha;
}

template <typename T, typename A>
T bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? alpha : s;
}

template <typename T, typename A>
T bounded_relu_bwd(T dd, T s, A alpha) {
    return dd * ((0 < s && s < alpha) ? 1 : 0);
}

template <typename T>
T soft_relu_fwd(T s) {
    return s < (T)logf(FLT_MAX) ? log1pf(::expf(s)) : s;
}

template <typename T>
T soft_relu_bwd(T dd, T s) {
    return dd / (1 + ::expf(-s));
}

template <typename T>
T logistic_fwd(T s) {
    T v = (T)(::expf(- (float)s));
    return 1 / (1 + v);
}

template <typename T>
T logistic_bwd(T dd, T s) {
    T v = logistic_fwd<T>(s);
    return dd * v * (1 - v);
}

template <typename T, typename A>
inline T clamp_fwd(T s, A alpha, A beta) {
    return s > alpha ? alpha : s < beta ? beta : s;
}

template <typename T, typename A>
inline T clamp_bwd(T dd, T s, A alpha, A beta) {
    return dd * (beta < s && s < alpha ? 1 : 0);
}

template <typename T>
inline T exp_fwd(T s) {
    return (T)(::expf((float)s));
}

template <typename T>
 inline T exp_bwd(T dd, T s) {
    return (T)(::expf((float)s));
}

template <typename T>
inline T not_fwd(T s) {
    return (T)(!s);
}

struct eltwise_test_params {
    engine::kind engine_kind;
    algorithm alg_kind;
    memory::format data_format;
    memory::format diff_format;
    float alpha, beta;
    memory::dims dims;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

size_t n_elems(const memory::desc &md) {
    size_t p = 1;
    const ptrdiff_t *pdims = md.data.layout_desc.blocking.padding_dims;
    for (int i = 0; i < md.data.ndims; ++i)
        p *= (size_t)(pdims[i]);
    return p;
}

template <typename data_t>
void ref_eltwise_fwd(const eltwise_test_params &p,
        const memory::desc &md, const memory &src, const memory &dst)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    size_t n = n_elems(md);
    for (size_t i = 0; i < n; ++i) {
        data_t s = src_data[i];
        data_t ref_d = 0;
        switch (p.alg_kind) {
        case eltwise_relu:        ref_d = relu_fwd(s, p.alpha);           break;
        case eltwise_tanh:        ref_d = tanh_fwd(s);                    break;
        case eltwise_elu:         ref_d = elu_fwd(s, p.alpha);            break;
        case eltwise_square:      ref_d = square_fwd(s);                  break;
        case eltwise_abs:         ref_d = abs_fwd(s);                     break;
        case eltwise_sqrt:        ref_d = sqrt_fwd(s);                    break;
        case eltwise_linear:      ref_d = linear_fwd(s, p.alpha, p.beta); break;
        case eltwise_bounded_relu: ref_d = bounded_relu_fwd(s, p.alpha);  break;
        case eltwise_soft_relu:   ref_d = soft_relu_fwd(s);               break;
        case eltwise_logistic:    ref_d = logistic_fwd(s);                break;
        case eltwise_clamp:       ref_d = clamp_fwd(s, p.alpha, p.beta);  break;
        case eltwise_exp:         ref_d = exp_fwd(s);                     break;
        case eltwise_not:         ref_d = not_fwd(s);                     break;
        default: assert(!"unknown alg_kind");
        }
        dst_data[i] = ref_d;
    }
}

template <typename data_t>
void compare_eltwise_fwd(const eltwise_test_params &p,
        const memory::desc &md, const memory &dst, const memory &ref_dst,
        const float eps)
{
    data_t *ref_dst_data = (data_t *)ref_dst.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    size_t n = n_elems(md);
    for (size_t i = 0; i < n; ++i) {
        float diff_err = dst_data[i] - ref_dst_data[i];
        float rel_err = std::abs(
                (std::min)(std::abs((float)ref_dst_data[i]), std::abs(diff_err)) > 1e-5
                ? diff_err / ref_dst_data[i]
                : diff_err);
        if (p.alg_kind == eltwise_soft_relu){
            EXPECT_NEAR(rel_err, 0, 2 * eps);
        }
        else{
            EXPECT_NEAR(rel_err, 0, eps);
        }
    }
}


template <typename data_t>
void check_eltwise_bwd(const eltwise_test_params &p,
        const memory::desc &md, const memory &src, const memory &diff_dst,
        const memory &diff_src, const float eps)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *diff_dst_data = (data_t *)diff_dst.get_data_handle();
    data_t *diff_src_data = (data_t *)diff_src.get_data_handle();

    const memory::desc data_d = src.get_primitive_desc().desc();
    const memory::desc diff_data_d = diff_src.get_primitive_desc().desc();

    ASSERT_EQ(md.data.data_type, memory::data_type::f32);

    size_t n = n_elems(md);
    for (size_t i = 0; i < n; ++i) {
        data_t ref_s = src_data[map_index(data_d, i)];
        data_t ref_dd = diff_dst_data[map_index(diff_data_d, i)];
        data_t ref_ds = 0;
        switch (p.alg_kind) {
        case eltwise_relu:   ref_ds = relu_bwd(ref_dd, ref_s, p.alpha); break;
        case eltwise_tanh:   ref_ds = tanh_bwd(ref_dd, ref_s); break;
        case eltwise_elu:    ref_ds = elu_bwd(ref_dd, ref_s, p.alpha); break;
        case eltwise_square: ref_ds = square_bwd(ref_dd, ref_s); break;
        case eltwise_abs:    ref_ds = abs_bwd(ref_dd, ref_s); break;
        case eltwise_sqrt:   ref_ds = sqrt_bwd(ref_dd, ref_s); break;
        case eltwise_linear:
            ref_ds = linear_bwd(ref_dd, ref_s, p.alpha, p.beta);
            break;
        case eltwise_bounded_relu:
            ref_ds = bounded_relu_bwd(ref_dd, ref_s, p.alpha);
            break;
        case eltwise_soft_relu:
            ref_ds = soft_relu_bwd(ref_dd, ref_s);
            break;
        case eltwise_logistic: ref_ds = logistic_bwd(ref_dd, ref_s); break;
        case eltwise_clamp: ref_ds = clamp_bwd(ref_dd, ref_s, p.alpha, p.beta); break;
        case eltwise_exp: ref_ds = exp_bwd(ref_dd, ref_s); break;
        default: assert(!"unknown alg_kind");
        }
        float diff_err = diff_src_data[map_index(diff_data_d, i)] - ref_ds;
        float rel_err = std::abs(
                (std::min)(std::abs((float)ref_ds), std::abs(diff_err)) > 1e-6
                ? diff_err / ref_ds
                : diff_err);
        EXPECT_NEAR(rel_err, 0, eps);
    }
}

}

#endif
