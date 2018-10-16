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

#include "ref_softmax.hpp"

#ifdef USE_MKL
#include "mkl_vml_functions.h"
#include "mkl_cblas.h"
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_dense() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

#   pragma omp parallel for schedule(static)
    for (int ou = 0; ou < outer_size_; ou++) {
        const data_t *src_data = src + ou * channels_;
        data_t *dst_data = dst + ou * channels_;
        data_t scalar = 0;

        _max(channels_, src_data, &scalar);
        _sub(channels_, scalar, src_data, dst_data);
        _exp(channels_, dst_data, dst_data);
        _sum(channels_, dst_data, &scalar);
        _scal(channels_, data_t(1)/scalar, dst_data);
    }
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_generic() {
    typedef typename prec_traits<data_type>::type data_t;
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const size_t dim = channels_ * inner_size_;

    outer_size_ = utils::array_product(conf_.src_pd()->desc()->dims, conf_.desc()->softmax_axis);

    for (int ou = 0; ou < outer_size_; ou++) {
        utils::array_set(max_, -nstl::numeric_limits<data_t>::max(), inner_size_);
        utils::array_set(denom_, 0, inner_size_);

        for (int c = 0; c < channels_; c++) {
            for(int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                max_[in] = nstl::max(max_[in], src[off]);
            }
        }

        for (int c = 0; c < channels_; c++) {
            for(int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                denom_[in] += dst[off] = exp(src[off] - max_[in]);
            }
        }

        for (int c = 0; c < channels_; c++) {
            for (int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                dst[off] /= denom_[in];
            }
        }
    }
}


template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::_max(int n, const data_t *x,
        data_t *max_data) {
    max_data[0] = x[0];
    for (int c = 1; c < n; ++c)
        max_data[0] = nstl::max(max_data[0], x[c]);
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::_sub(int n, data_t alpha, const data_t *x,
        data_t *y) {
    for (int c = 0; c < n; ++c)
        y[c] = x[c] - alpha;
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::_exp(int n, const data_t *a, data_t *r) {
#ifdef USE_MKL
    if (data_type == data_type::f32) {
        vsExp(n, a, r);
        return;
    }
#endif
#   pragma omp parallel for
    for (int c = 0; c < n; ++c)
        r[c] = expf(a[c]);
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::_sum(int n, const data_t *x,
        data_t *sum_data) {
    sum_data[0] = 0;
    for (int c = 0; c < n; ++c)
        sum_data[0] += x[c];
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::_scal(int n, data_t alpha, data_t *x) {
#ifdef USE_MKL
    if (data_type == data_type::f32) {
        cblas_sscal(n, alpha, x, 1);
        return;
    }
#endif
#   pragma omp parallel for
    for (int c = 0; c < n; ++c)
        x[c] *= alpha;
}

template struct ref_softmax_fwd_t<data_type::f32>;


// NC/NCHW softmax for along final axe (1 for NC, 3 for NCHW)
template <impl::data_type_t data_type>
void ref_softmax_bwd_t<data_type>::execute_backward_dense() {
    auto data = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

#   pragma omp parallel for schedule(static)
    for (int ou = 0; ou < outer_size_; ou++) {
        data_t sbr = 0;
        size_t off = channels_*ou;
        for (int c = 0; c < channels_; c++) {
            size_t loff = off + c;
            data_t ldata = data[loff];
            sbr += diff_dst[loff]*ldata;
            diff_src[loff] = ldata;
        }

        for(int c=0; c < channels_ ; ++c) {
          size_t loff = off + c;
          diff_src[loff] *= (diff_dst[loff] - sbr);
        }
    }
}

template <impl::data_type_t data_type>
void ref_softmax_bwd_t<data_type>::execute_backward_generic() {
    const size_t dim = channels_ * inner_size_;
    auto data = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));
    const memory_desc_wrapper diff_d(conf_.diff_src_pd());
    const memory_desc_wrapper data_d(conf_.dst_pd());

#   pragma omp parallel for schedule(static)
    for (int ou = 0; ou < outer_size_; ou++) {
        for (int in = 0; in < inner_size_; in++) {
            data_t sbr = 0;
            for (int c = 0; c < channels_; c++) {
                size_t off_diff = diff_d.off_l(ou * dim + c * inner_size_ + in);
                size_t off_data = diff_d.off_l(ou * dim + c * inner_size_ + in);
                sbr += diff_dst[off_diff]*data[off_data];
            }

            for(int c=0; c < channels_ ; ++c) {
              size_t off_diff = diff_d.off_l(ou * dim + c * inner_size_ + in);
              size_t off_data = data_d.off_l(ou * dim + c * inner_size_ + in);
              diff_src[off_diff] = data[off_data]*(diff_dst[off_diff] - sbr);
            }
        }
    }
}

template struct ref_softmax_bwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
