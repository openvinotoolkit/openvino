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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "ref_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::data_type;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
         data_type_t acc_type>
void ref_inner_product_fwd_t<src_type, wei_type, dst_type, acc_type>
        ::execute_forward() {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC();

    const bool src_has_spatial = utils::one_of(src_d.ndims(), 4, 5);

    const bool is_3d = src_d.ndims() == 5;

    auto ker_has_spatial = [=](acc_data_t &d, int mb, int oc) {
        const int KD = conf_.KD();
        const int KH = conf_.KH();
        const int KW = conf_.KW();
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        if (is_3d)
                            d += (acc_data_t)src[src_d.off(mb, ic, kd, kh, kw)]
                                * weights[weights_d.off(oc, ic, kd, kh, kw)];
                        else
                            d += (acc_data_t)src[src_d.off(mb, ic, kh, kw)]
                                * weights[weights_d.off(oc, ic, kh, kw)];
                    }
                }
            }
        }
    };

    auto ker_no_spatial = [=](acc_data_t &d, int mb, int oc) {
        for (int ic = 0; ic < IC; ++ic) {
            d += (acc_data_t)src[src_d.off(mb, ic)]
                * weights[weights_d.off(oc, ic)];
        }
    };

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC; ++oc) {
            acc_data_t a = bias ? bias[bias_d.off(oc)] : (dst_data_t)0;
            if (src_has_spatial) {
                ker_has_spatial(a, mb, oc);
            } else {
                ker_no_spatial(a, mb, oc);
            }
            dst[dst_d.off(mb, oc)] = (dst_data_t)a;
        }
    }
}

template struct ref_inner_product_fwd_t<f32>;
template struct ref_inner_product_fwd_t<s16, s16, s32, s32>;
template struct ref_inner_product_fwd_t<u8, s8, u8, s32>;

template <data_type_t diff_src_type, data_type_t wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void ref_inner_product_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
     acc_type>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC();

    const bool diff_src_has_spatial = utils::one_of(diff_src_d.ndims(), 4, 5);

    const bool is_3d = diff_src_d.ndims() == 5;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int ic = 0; ic < IC; ++ic) {
            if (diff_src_has_spatial) {
                const int KD = conf_.KD();
                const int KH = conf_.KH();
                const int KW = conf_.KW();
                for (int kd = 0; kd < KD; ++kd)
                for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    acc_data_t ds = acc_data_t(0);
                    for (int oc = 0; oc < OC; ++oc) {
                        if (is_3d)
                            ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                                * weights[weights_d.off(oc, ic, kd, kh, kw)]);
                        else
                            ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                                * weights[weights_d.off(oc, ic, kh, kw)]);
                    }
                    if (is_3d) diff_src[diff_src_d.off(mb, ic, kd, kh, kw)] =
                        (diff_src_data_t)ds;
                    else diff_src[diff_src_d.off(mb, ic, kh, kw)] =
                        (diff_src_data_t)ds;
                }
            } else {
                acc_data_t ds = acc_data_t(0);
                for (int oc = 0; oc < OC; ++oc) {
                    ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)] *
                        weights[weights_d.off(oc, ic)]);
                }
                diff_src[diff_src_d.off(mb, ic)] = (diff_src_data_t)ds;
            }
        }
    }
}

template struct ref_inner_product_bwd_data_t<f32, f32, f32, f32>;
template struct ref_inner_product_bwd_data_t<s32, s16, s16, s32>;

template <impl::data_type_t data_type>
void ref_inner_product_bwd_weights_t<data_type>::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t*>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    diff_dst += diff_dst_d.blocking_desc().offset_padding;

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC();

    const bool src_has_spatial = utils::one_of(src_d.ndims(), 4 ,5);

    const bool is_3d = src_d.ndims() == 5;

#   pragma omp parallel for collapse(2) schedule(static)
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            if (src_has_spatial) {
                const int KD = conf_.KD();
                const int KH = conf_.KH();
                const int KW = conf_.KW();
                for (int kd = 0; kd < KD; ++kd) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            data_t *dw = is_3d
                                ? &diff_weights[
                                diff_weights_d.off(oc, ic, kd, kh, kw)]
                                : &diff_weights[
                                diff_weights_d.off(oc, ic, kh, kw)];
                            *dw = data_t(0);
                            for (int mb = 0; mb < MB; ++mb) {
                                if (is_3d)
                                    *dw += diff_dst[diff_dst_d.off(mb, oc)] *
                                        src[src_d.off(mb, ic, kd, kh, kw)];
                                else
                                    *dw += diff_dst[diff_dst_d.off(mb, oc)] *
                                        src[src_d.off(mb, ic, kh, kw)];
                            }
                        }
                    }
                }
            } else {
                data_t *dw = &diff_weights[diff_weights_d.off(oc, ic)];
                *dw = data_t(0);
                for (int mb = 0; mb < MB; ++mb) {
                    *dw += diff_dst[diff_dst_d.off(mb, oc)] *
                        src[src_d.off(mb, ic)];
                }
            }
        }
    }

    if (diff_bias) {
        diff_bias += diff_bias_d.blocking_desc().offset_padding;
        constexpr int blksize = 8;
        int OC_blocks = OC / blksize;
        int rem_OC = OC % blksize;
#       pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int oc_st{0}, oc_e{0};
            balance211(OC_blocks, nthr, ithr, oc_st, oc_e);
            oc_st = oc_st * blksize;
            oc_e = oc_e * blksize;

            PRAGMA_OMP_SIMD()
            for (int oc = oc_st; oc < oc_e; ++oc) {
                diff_bias[oc] = diff_dst[oc];
            }

            for (int mb = 1; mb < MB; ++mb) {
                PRAGMA_OMP_SIMD()
                for (int oc = oc_st; oc < oc_e; ++oc) {
                    diff_bias[oc] += diff_dst[mb * OC + oc];
                }
            }

            if (rem_OC != 0 && ithr == nthr-1) {
                for (int oc = OC_blocks * blksize; oc < OC; oc++)
                    diff_bias[oc] = diff_dst[oc];
                for (int mb = 1; mb < MB; ++mb) {
                    for (int oc = OC_blocks * blksize; oc < OC; oc++) {
                        diff_bias[oc] += diff_dst[mb * OC + oc];
                    }
                }
            }
        }
    }
}

template struct ref_inner_product_bwd_weights_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
