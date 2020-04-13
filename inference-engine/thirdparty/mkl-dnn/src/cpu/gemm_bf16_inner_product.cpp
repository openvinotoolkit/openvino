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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "gemm_bf16_inner_product.hpp"
#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;
using namespace memory_tracking::names;
using namespace mkldnn::impl::cpu::bf16_cvt_utils;

template <data_type_t dst_data_type>
void gemm_bf16_inner_product_fwd_t<dst_data_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const int M = pd()->OC();
    const int N = pd()->MB();
    const int K = pd()->IC_total_padded();

    bool wei_tr = !utils::one_of(pd()->weights_pd()->desc()->format,
             hwio, dhwio, io);

    acc_data_t *acc = pd()->dst_is_acc_
        ? (acc_data_t *)dst
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K,
            &alpha, weights, wei_tr ? &K : &M, src, &K, &beta, acc, &M);

    const float *scales = pd()->attr()->output_scales_.scales_;
    if (postops_in_ip_)
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end);
        });
}

template <data_type_t diff_src_data_type>
void gemm_bf16_inner_product_bwd_data_t<diff_src_data_type>::
    execute_backward_data() const
{
    auto diff_dst =
        reinterpret_cast<const diff_dst_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const int M = pd()->IC_total_padded();
    const int N = pd()->MB();
    const int K = pd()->OC();

    bool wei_tr = utils::one_of(pd()->weights_pd()->desc()->format,
             hwio, dhwio, io);

    acc_data_t *acc = pd()->diff_src_is_acc_
        ? (acc_data_t *)diff_src
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha,
            weights, wei_tr ? &K : &M, diff_dst, &K, &beta, acc, &M);

    if (!pd()->diff_src_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((mkldnn_bfloat16_t *)&diff_src[start],
                    (const float *)&acc[start],
                    end - start);
        });
    }
}

template <data_type_t diff_wei_data_type>
void gemm_bf16_inner_product_bwd_weights_t<diff_wei_data_type>::
    execute_backward_weights() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst =
        reinterpret_cast<const diff_dst_data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<diff_wei_data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<char *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_pd(1));

    diff_dst += diff_dst_d.blocking_desc().offset_padding;

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    bool wei_tr = utils::one_of(pd()->diff_weights_pd()->desc()->format,
             hwio, dhwio, io);

    const int M = wei_tr ? OC : IC;
    const int N = wei_tr ? IC : OC;
    const int K = MB;

    acc_data_t *acc = pd()->diff_wei_is_acc_
        ? (acc_data_t *)diff_weights
        : scratchpad().template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    gemm_bf16bf16f32("N", "T", &M, &N, &K, &alpha,
            wei_tr ? diff_dst : src, &M, wei_tr ? src : diff_dst, &N, &beta,
            acc, &M);

    if (!pd()->diff_wei_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)M * N, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((mkldnn_bfloat16_t *)&diff_weights[start],
                    (const float *)&acc[start],
                    end - start);
        });
    }

    if (pd()->with_bias()) {
        const size_t bias_dt_size = types::data_type_size(
                pd()->desc()->diff_bias_desc.data_type);
        diff_bias += bias_dt_size * diff_bias_d.blocking_desc().offset_padding;
        constexpr int blksize = 16;
        const int OC_blocks = OC / blksize;
        const int rem_OC = OC % blksize;
        float *ddst_ws = (float *)scratchpad().template get<acc_data_t>(
            key_iprod_dst_bf16_convert_wsp);
        float *diff_bias_acc = pd()->diff_bias_is_acc_
                ? (float *)diff_bias
                : (float *)scratchpad().template get<acc_data_t>(
                          key_iprod_bias_bf16_convert_wsp);
        parallel(0, [&](const int ithr, const int nthr) {
            int oc_st{0}, oc_e{0};
            balance211(OC_blocks, nthr, ithr, oc_st, oc_e);
            oc_st = oc_st * blksize;
            oc_e = oc_e * blksize;

            PRAGMA_OMP_SIMD()
            for (int oc = oc_st; oc < oc_e; ++oc)
                diff_bias_acc[oc] = 0.0f;

            for (int mb = 0; mb < MB; ++mb) {
                if (oc_e > oc_st)
                    cvt_bfloat16_to_float(&ddst_ws[oc_st],
                        (const mkldnn_bfloat16_t *)&diff_dst[mb * OC + oc_st],
                        oc_e - oc_st);

                PRAGMA_OMP_SIMD()
                for (int oc = oc_st; oc < oc_e; ++oc)
                    diff_bias_acc[oc] += ddst_ws[oc];
            }
            if (!pd()->diff_bias_is_acc_ && oc_st < oc_e)
                cvt_float_to_bfloat16(
                    &((mkldnn_bfloat16_t *)diff_bias)[oc_st],
                    &((const float *)diff_bias_acc)[oc_st],
                    oc_e - oc_st);

            if (rem_OC != 0 && ithr == nthr-1) {
                oc_st = OC_blocks * blksize;
                oc_e = OC;
                for (int oc = oc_st; oc < oc_e; oc++)
                    diff_bias_acc[oc] = 0.0f;
                for (int mb = 0; mb < MB; ++mb) {
                    cvt_bfloat16_to_float(&ddst_ws[oc_st],
                        (const mkldnn_bfloat16_t *)&diff_dst[mb * OC + oc_st],
                        oc_e - oc_st);

                    for (int oc = oc_st; oc < oc_e; ++oc)
                        diff_bias_acc[oc] += ddst_ws[oc];
                }

                if (!pd()->diff_bias_is_acc_ && oc_st < oc_e)
                    cvt_float_to_bfloat16(
                        &((mkldnn_bfloat16_t *)diff_bias)[oc_st],
                        &((const float *)diff_bias_acc)[oc_st],
                        oc_e - oc_st);
            }
        });
    }
}

template struct gemm_bf16_inner_product_fwd_t<data_type::f32>;
template struct gemm_bf16_inner_product_fwd_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::bf16>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
