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

#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"
#include "gemm_u8s8s32x_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace math;
using namespace memory_format;

template <data_type_t dst_type>
void gemm_u8s8s32x_inner_product_fwd_t<dst_type>::execute_forward() {
#if USE_MKL_IGEMM
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const int MB = conf_.MB();
    const int OC = conf_.OC();

    bool wei_tr = utils::one_of(conf_.weights_pd()->desc()->format,
             oihw, oidhw, oi);

    const int M = OC;
    const int N = MB;
    const int K = conf_.IC_total_padded();
    const int8_t off_a = 0, off_b = 0;
    const int32_t off_c = 0;

    const int scale_idx_mult = conf_.attr()->output_scales_.mask_ == (1 << 1);
    const float *scales = conf_.attr()->output_scales_.scales_;
    const auto rmode = conf_.attr()->round_mode_;

    const auto &post_ops = conf_.attr()->post_ops_;
    const bool do_relu = post_ops.len_ == 1;
    const float nslope = do_relu ? post_ops.entry_[0].eltwise.alpha : 0.f;

    acc_data_t *acc = this->dst_is_acc_
        ? (acc_data_t *)dst
        : (acc_data_t *)this->scratchpad_->get();

    auto get_bias = [=, &bias](size_t off) -> acc_data_t {
#       define CASE(dt) case dt: return (acc_data_t)\
        (*((const prec_traits<dt>::type *)bias + off))
        switch (conf_.desc()->bias_desc.data_type) {
        CASE(data_type::s8);
        CASE(data_type::u8);
        CASE(data_type::s32);
        CASE(data_type::f32);
        default: assert(!"unimplemented");
        }
#       undef CASE
        return 0;
    };

    cblas_gemm_s8u8s32(CblasColMajor, wei_tr ? CblasTrans : CblasNoTrans,
            CblasNoTrans, CblasFixOffset, M, N, K, 1., weights,
            wei_tr ? K : M, off_a, src, K, off_b, 0., acc, M, &off_c);

    parallel_nd(MB, OC, [&](int mb, int oc) {
        size_t dst_off = mb * OC + oc;
        float d = (float)acc[dst_off];
        if (bias)
            d += get_bias(oc);
        d *= scales[oc * scale_idx_mult];
        if (do_relu && d < 0)
            d *= nslope;
        dst[dst_off] = qz_a1b0<float, dst_data_t>()(d, rmode);
    });
#endif
}

using namespace data_type;

template struct gemm_u8s8s32x_inner_product_fwd_t<f32>;
template struct gemm_u8s8s32x_inner_product_fwd_t<s32>;
template struct gemm_u8s8s32x_inner_product_fwd_t<s8>;
template struct gemm_u8s8s32x_inner_product_fwd_t<u8>;
}
}
}
